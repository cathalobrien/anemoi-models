import torch 
import einops
import math

from flash_attn import flash_attn_func
from torch.nn.functional import scaled_dot_product_attention

def generate_diagonal_mask_indices(seq_len, window_size):
    indices = []
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        for j in range(start, end):
            indices.append((i, j))  # Record the (row, col) indices
    return indices

def compute_attn_at_indices(q, k, mask_indices):
    """
    Compute the attention scores at specific diagonal indices for the given query and key tensors.
    
    Args:
        q (torch.Tensor): The query tensor of shape [batch_size, seq_len, embed_dim].
        k (torch.Tensor): The key tensor of shape [batch_size, seq_len, embed_dim].
        mask_indices (list): A list of tuples of the form (row_index, col_index) indicating where to compute attention scores.
    
    Returns:
        results (torch.Tensor): A 1D tensor containing the attention scores at the given mask indices.
    """
    batch_size, seq_len, embed_dim = q.shape
    results = []

    # Loop through the mask_indices and calculate only the corresponding attention scores
    for (i, j) in mask_indices:
        # Compute the dot product for the (i, j) pair
        attn_score = torch.einsum("bd,bd->b", q[:, i, :], k[:, j, :]) / (embed_dim ** 0.5)
        results.append(attn_score)
    
    # Stack the results into a tensor
    return torch.stack(results, dim=1)  # Shape: [batch_size, len(mask_indices)]

def recombine_attn_scores(attn_score_chunks, window_size, seq_len):
    batch_size, num_chunks, chunk_size, _ = attn_score_chunks.shape

    # Initialize an empty tensor for full attention scores of shape [batch_size, seq_len, seq_len]
    attn_scores_full = torch.zeros((batch_size, seq_len, seq_len), device=attn_score_chunks.device, dtype=torch.float16)
    
    # Initialize a counter tensor to average overlapping regions
    overlap_count = torch.zeros((batch_size, seq_len, seq_len), device=attn_score_chunks.device, dtype=torch.float16)

    # Fill in the overlapping regions from chunks back into the full attention scores
    for i in range(num_chunks):
        start = i * window_size
        end = start + chunk_size
        
        # Accumulate the attention scores in the full matrix
        attn_scores_full[:, start:end, start:end] += attn_score_chunks[:, i, :, :]

        # Track the overlap counts for each element in the sequence
        overlap_count[:, start:end, start:end] += 1

    # Avoid division by zero in non-overlapping positions
    overlap_count = overlap_count.masked_fill(overlap_count == 0, 1.0)

    # Average the overlapping regions by dividing the accumulated scores by the counts
    attn_scores_full /= overlap_count

    return attn_scores_full




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

def swa2(q,k,v,window_size):
    batch_size, seq_len, embed_dim = q.shape
    
    mask_indices = generate_diagonal_mask_indices(seq_len, window_size)

    # Compute attention scores at the masked indices
    #TODO transpose of K?
    attn_results = torch.softmax(compute_attn_at_indices(q, k, mask_indices) / (embed_dim ** 0.5), dim=-1)
    
    
    #compute values subset
    batch_size = v.shape[0]
    filtered_values = []
    
    # Loop through the mask indices and extract the corresponding values
    for (i, j) in mask_indices:
        # Assuming you want to extract the element from the first batch (batch index = 0)
        filtered_values.append(v[0, i, j])
    
    # Stack the filtered values into a 1D tensor
    v_subset = torch.stack(filtered_values, dim=0)
    print(f"{v_subset=}\n{attn_results=}")
    
    return attn_results * v_subset

def sliding_window_attention(q, k, v, window_size):
    """
    q: Queries tensor of shape [batch_size, seq_len, embedding_dim]
    k: Keys tensor of shape [batch_size, seq_len, embedding_dim]
    v: Values tensor of shape [batch_size, seq_len, embedding_dim]
    window_size: The size of the sliding window (i.e., how many tokens each token attends to)
    """
    batch_size, seq_len, embed_dim = q.shape
    
    if window_size == -1:
        # Step 1: Global attention (standard attention)
        attn_scores = torch.einsum("bsd, btd -> bst", q, k) / (embed_dim ** 0.5)  # shape [batch_size, seq_len, seq_len]
        print(f"{attn_scores.shape=}") #attn_scores.shape=torch.Size([1, 8, 8])
        attn_weights = torch.softmax(attn_scores, dim=-1)
        #return attn_weights @ v
        attn_output = torch.einsum("bst, btd -> bsd", attn_weights, v)
        return attn_output
    
    # Sliding window attention case
    chunk_size = 2 * window_size  # Set chunk size to 2 * window_size (to account for overlap)
    num_chunks = (seq_len // window_size) - 1

    # Chunk q and k with overlap
    chunk_q = torch.empty((batch_size, num_chunks, chunk_size, embed_dim), device=q.device, dtype=torch.float16)
    chunk_k = torch.empty_like(chunk_q)

    # Create overlapping chunks of q and k
    for i in range(num_chunks):
        chunk_q[:, i, :, :] = q[:, i * window_size : i * window_size + chunk_size, :]
        chunk_k[:, i, :, :] = k[:, i * window_size : i * window_size + chunk_size, :]
    
    print(f"{chunk_q.shape=}")

    # Compute attention scores for chunks
    attn_score_chunks = torch.einsum("bnqd, bnkd -> bnqk", chunk_q, chunk_k) / (embed_dim ** 0.5)
    print(f"{attn_score_chunks.shape=}") #attn_score_chunks.shape=torch.Size([1, 3, 4, 4])

    attn_scores_full = recombine_attn_scores(attn_score_chunks, window_size, seq_len)
    print(f"{attn_scores_full.shape}")
    
    attn_weights = torch.softmax(attn_scores_full, dim=-1)
    return attn_weights @ v


pytorch_attn_out = swa2(q,k,v,w)
#pytorch_attn_out = torch.einsum('btsd -> bst d', pytorch_attn_out) #go from [1, 1, 8, 64] to [1, 8, 1, 64]
print(f"{pytorch_attn_out.shape=}")
#print(f"{pytorch_attn_out[0][0]=}")
print(f"{pytorch_attn_out=}")

#change shape from i.e. (1, 8, 768) to (1, 8, 1, 768)
#the extra dimenson represents numbers of heads
#now vectors are in the form "batch, seq_len, heads, embedding size"

#query, key, value = (
#    einops.rearrange(t, "batch heads grid vars -> batch grid heads vars") for t in (query, key, value)
#)
window_size=(w,w) #TODO make this safe
flash_attn_out = flash_attn_func(q.unsqueeze(2), k.unsqueeze(2), v.unsqueeze(2), causal=False, window_size=window_size, dropout_p=0.0)
print(f"{flash_attn_out.shape=}") #flash_attn_out.shape=torch.Size([1, 8, 1, 64])
print(f"{flash_attn_out[0][0]=}")
#flash_attn_out = einops.rearrange(flash_attn_out, "batch grid heads vars -> batch heads grid vars")

if (w == -1):
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
