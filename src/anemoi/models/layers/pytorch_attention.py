import torch 
import einops
import math

from flash_attn import flash_attn_func
from torch.nn.functional import scaled_dot_product_attention

#q = torch.arange(1, 9).reshape(4,2)
#k = torch.arange(1, 13).reshape(6,2)
#q_kt = torch.einsum('xd,yd->xy', q, k)
#kt = torch.einsum('xd->dx', k)
#print(f"{q=},\n{k=},\n{kt=},\n{q_kt=}") #,\n{q_k=}")

#batch size, seq len, num_heads(1), embedding size
q = torch.randn(1, 8, 64, dtype=torch.float16, device="cuda")
k = torch.randn(1, 8, 64, dtype=torch.float16, device="cuda")
v = torch.randn(1, 8, 64, dtype=torch.float16, device="cuda")
w = 2


print(f"setup: {w=}, \n{q.shape=} \n")

#batch size, heads, seq len, embedding size
#q = torch.randn(1, 16, 8, 768)
#k = torch.randn(1, 16, 8, 768)
#v = torch.randn(1, 16, 8, 768)

#input: q: (batch_size, seqlen, 1, headdim)
#output shape (batch_size, seqlen, headdim)

def pytorch_attention(q, k, v, w):

    def _chunk(hidden_states, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap = w"""
        chunk_size = [
            hidden_states.size(0), #bs
            torch.div(hidden_states.size(1), window_overlap, rounding_mode="trunc") - 1, #n_chunks
            window_overlap * 2,
            hidden_states.size(2),
        ]

        overlapping_chunks = torch.empty(chunk_size, device=hidden_states.device, dtype=q.dtype)
        for chunk in range(chunk_size[1]):
            overlapping_chunks[:, chunk, :, :] = hidden_states[
                :, chunk * window_overlap : chunk * window_overlap + 2 * window_overlap, :
            ]
        return overlapping_chunks

    def _create_mask(window_overlap):
        """Create a mask to zero out diagonals."""
        chunk_size = 2 * window_overlap
        mask = torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.float32)
        mask = torch.triu(mask, diagonal=window_overlap)  # upper diagonal
        mask = mask + torch.tril(mask, diagonal=-window_overlap)  # lower diagonal
        mask = mask == 0  # True where we want to mask
        return mask

    def _unchunk(chunked_hidden_states, window_overlap, seq_len):
        """Reconstruct the sequence from overlapping chunks"""
        batch_size, n_chunks, chunk_size, embed_size = chunked_hidden_states.size()

        # Initialize the output tensor
        output_hidden_states = torch.zeros((batch_size, seq_len, embed_size), device=chunked_hidden_states.device, dtype=q.dtype)
        overlap_count = torch.zeros((batch_size, seq_len, embed_size), device=chunked_hidden_states.device, dtype=q.dtype)

        for chunk in range(n_chunks):
            start = chunk * window_overlap
            end = start + 2 * window_overlap
            output_hidden_states[:, start:end, :] += chunked_hidden_states[:, chunk, :, :]
            overlap_count[:, start:end, :] += 1  # Track the number of overlapping chunks

        # Handle overlaps by averaging
        output_hidden_states /= overlap_count
        return output_hidden_states.unsqueeze(2)

    if (w != 0):
        query = _chunk(q, window_overlap=w)
        key   = _chunk(k, window_overlap=w)
        value = _chunk(v, window_overlap=w)
    else:
        query = q
        key = k
        value = v
    print(f"{query.shape=}, {key.shape=}, {value.shape=}")

    if (w != 0):
        diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))
        
        # Create a mask to zero out diagonals
        mask = _create_mask(window_overlap=w)
        print(f"Mask shape: {mask.shape}")

        # Apply the mask to zero out diagonal values
        diagonal_chunked_attention_scores.masked_fill_(mask, float('-inf'))  # Mask out undesired positions

    else:
        diagonal_chunked_attention_scores = torch.einsum("bxd,byd->bxy", (query, key))
    print(f"after masking {diagonal_chunked_attention_scores.shape=}")

    #scale_factor = 1 / math.sqrt(query.size(-1))

    #can fuse softmax with root
    #weights = torch.softmax(diagonal_chunked_attention_scores / key.shape[1] ** 0.5, dim=-1)
    max_score = torch.max(diagonal_chunked_attention_scores, dim=-1, keepdim=True)[0]
    stabilized_scores = diagonal_chunked_attention_scores - max_score
    weights = torch.softmax(stabilized_scores / (query.shape[-1] ** 0.5), dim=-1)

    #weights = torch.softmax(diagonal_chunked_attention_scores / (query.shape[-1] ** 0.5), dim=-1)
    #weights = torch.softmax(diagonal_chunked_attention_scores * scale_factor, dim=-1)
    print(f"{weights.shape=}")
    print(f"{value.shape=}")
    attention = torch.matmul(weights, value)

    #if (w != 0):
        # Unchunk the result to get back to the original sequence length
        #seq_len = q.size(1)  # Get the original sequence length from q
        #attention = _unchunk(attention, window_overlap=w, seq_len=seq_len)

    return attention

pytorch_attn_out = pytorch_attention(q,k,v,w)
pytorch_attn_out = torch.einsum('btsd -> bst d', pytorch_attn_out) #go from [1, 1, 8, 64] to [1, 8, 1, 64]
print(f"{pytorch_attn_out.shape=}")
print(f"{pytorch_attn_out[0][0]=}")

#exit()

#change shape from i.e. (1, 8, 768) to (1, 8, 1, 768)
#the extra dimenson represents numbers of heads
#now vectors are in the form "batch, seq_len, heads, embedding size"

#query, key, value = (
#    einops.rearrange(t, "batch heads grid vars -> batch grid heads vars") for t in (query, key, value)
#)
if (w != 0):
    #window_size=(w//2,w//2) #TODO make this safe
    window_size=(w,w) #TODO make this safe
else:
    window_size=(-1,-1)
flash_attn_out = flash_attn_func(q.unsqueeze(2), k.unsqueeze(2), v.unsqueeze(2), causal=False, window_size=window_size, dropout_p=0.0)
print(f"{flash_attn_out.shape=}") #flash_attn_out.shape=torch.Size([1, 8, 1, 64])
print(f"{flash_attn_out[0][0]=}")
#flash_attn_out = einops.rearrange(flash_attn_out, "batch grid heads vars -> batch heads grid vars")

if (w == 0):
    sdpa_attn_out = scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=False, dropout_p=0.0,)  # expects (batch heads grid variable) format
    sdpa_attn_out = torch.einsum('btsd -> bst d', sdpa_attn_out)
    print(f"{sdpa_attn_out.shape=}")
    print(f"{sdpa_attn_out[0][0]=}")
    are_close = torch.allclose(pytorch_attn_out, sdpa_attn_out, atol=1e-6, rtol=1e-5)
    if are_close:
        print("pytorch_attn_out and sdpa_attn_out are numerically close.")
    else:
        print("pytorch_attn_out and sdpa_attn_out differ.")


else:
    print("Sliding window not supported in Pytorch SDPA")

are_close = torch.allclose(pytorch_attn_out, flash_attn_out, atol=1e-6, rtol=1e-5)
if are_close:
    print("pytorch_attn_out and flash_attn_out are numerically close.")
else:
    print("pytorch_attn_out and flash_attn_out differ.")
