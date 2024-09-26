import torch 
import einops
import math

from flash_attn import flash_attn_func
from torch.nn.functional import scaled_dot_product_attention

def slidingWindowAttn(query, key, value, window_size):
    # Inputs:
    #   query  -> tensor of shape (batch_size, seq_len, embed_dim)
    #   key    -> tensor of shape (batch_size, seq_len, embed_dim)
    #   value  -> tensor of shape (batch_size, seq_len, embed_dim)
    #   window_size -> the size of the local sliding window
    #
    # Output:
    #   output -> tensor of shape (batch_size, seq_len, embed_dim)

    batch_size, seq_len, embed_dim = query.shape

    # Global attention 
    if window_size == -1:
        attn_scores = torch.einsum("bsd, btd -> bst", query, key) / (embed_dim ** 0.5)  # shape [batch_size, seq_len, seq_len]
        attn_weights = torch.softmax(attn_scores, dim=-1) #TODO replace with scaled SM for stability
        attn_output = torch.einsum("bst, btd -> bsd", attn_weights, v)
        return attn_output

    # Sw attention

    output = torch.zeros(batch_size, seq_len, embed_dim, dtype=torch.float16, device="cuda")
        #attn_scores = torch.zeros(batch_size, seq_len, 2 * window_size + 1, device=query.device, dtype=torch.float16)

    for i in range(seq_len):
        # Define the local window indices for the query at position i
        # The window is centered around i with size `window_size`.
        start_idx = max(0, i - window_size)
        end_idx = min(seq_len, i + window_size + 1)  # window_size on both sides

        # Extract local key and value slices corresponding to the window
        local_keys = key[:, start_idx:end_idx, :]    # Shape: (batch_size, window_len, embed_dim)
        local_values = value[:, start_idx:end_idx, :]  # Shape: (batch_size, window_len, embed_dim)

        # Compute attention scores between query[i] and the local keys
        # Attention scores are computed as a dot product between query[i] and each key in the window
        #attn_scores = Dot product of query[:, i, :] with local_keys along embed_dim  # Shape: (batch_size, window_len)
        attn_scores = torch.einsum('bqd,bkd->bqk', query[:, i:i+1, :], local_keys) #could replace with query[i]
        
        # Scale attention scores by the square root of the embedding dimension
        attn_scores = attn_scores / (embed_dim ** 0.5)

        # Apply softmax to get attention weights
        # first, subtract the max value for numerical stability (preventing overflow in softmax)
        attn_scores_scaled = attn_scores - torch.max(attn_scores, dim=-1, keepdim=True)[0]
        attn_weights = torch.softmax(attn_scores_scaled, dim=-1)  # Shape: (batch_size, window_len)

        # Compute the attention output as a weighted sum of the local values
        #weighted_sum = Matrix multiplication of attn_weights and local_values  # Shape: (batch_size, embed_dim)
        weighted_sum = attn_weights @ local_values

        # Store the result in the output tensor
        output[:, i, :] = weighted_sum

    # Return the final attention output
    return output


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

pytorch_attn_out = slidingWindowAttn(q,k,v,w)
#pytorch_attn_out = torch.einsum('btsd -> bst d', pytorch_attn_out) #go from [1, 1, 8, 64] to [1, 8, 1, 64]
print(f"{pytorch_attn_out.shape=}")
print(f"{pytorch_attn_out[0][0]=}")
#print(f"{pytorch_attn_out=}")

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

are_close = torch.allclose(pytorch_attn_out.unsqueeze(1), flash_attn_out, atol=1e-6, rtol=1e-5)
if are_close:
    print("pytorch_attn_out and flash_attn_out are numerically close.")
else:
    print("pytorch_attn_out and flash_attn_out differ.")
