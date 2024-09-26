import torch 
import einops
import math

from flash_attn import flash_attn_func
from torch.nn.functional import scaled_dot_product_attention

#@torch.compile(mode="max-autotune")
@torch.compile()
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
        attn_scores_scaled = attn_scores - torch.max(attn_scores, dim=-1, keepdim=True)[0] #scaled softmax for numerical stability
        attn_weights = torch.softmax(attn_scores_scaled, dim=-1)
        attn_output = torch.einsum("bst, btd -> bsd", attn_weights, v)
        return attn_output

    # sliding window attention
    output = torch.zeros(batch_size, seq_len, embed_dim, dtype=torch.float16, device="cuda")

    # Loop over each element in sequence, calculate its local winow and compute the resulting attention score for that element
    for i in range(seq_len):
        # Define the local window indices for the query at position i
        # the window_len is [window_size + 1 + window_size] big
        start_idx = max(0, i - window_size)
        end_idx = min(seq_len, i + window_size + 1)  # window_size on both sides

        # Extract local key and value slices corresponding to the window
        local_keys = key[:, start_idx:end_idx, :]    # Shape: (batch_size, window_len, embed_dim)
        local_values = value[:, start_idx:end_idx, :]  # Shape: (batch_size, window_len, embed_dim)

        # Compute attention scores between query[i] and the local keys covered by the window
        attn_scores = torch.einsum('bqd,bkd->bqk', query[:, i:i+1, :], local_keys)
       
        #scale and softmax the attn_scores
        attn_scores = attn_scores / (embed_dim ** 0.5)
        attn_scores_scaled = attn_scores - torch.max(attn_scores, dim=-1, keepdim=True)[0] # subtract the max value for numerical stability
        attn_weights = torch.softmax(attn_scores_scaled, dim=-1)  # Shape: (batch_size, window_len)

        #store results in output buffer
        output[:, i, :] = attn_weights @ local_values

    # Return the final attention output
    return output

#@torch.compile(mode="max-autotune")
@torch.compile()
def multiheadAttn(query, key, value, window_size):
    # Inputs:
    #   query  -> tensor of shape (batch_size, seq_len, nheads, head_dim)
    #   key    -> tensor of shape (batch_size, seq_len, nheads, head_dim)
    #   value  -> tensor of shape (batch_size, seq_len, nheads, head_dim)
    #   window_size -> the size of the local sliding window
    #
    # Output:
    #   output -> tensor of shape (batch_size, seq_len, nheads, head_dim)

    # Get the dimensions
    batch_size, seq_len, nheads, head_dim = query.shape

    # allocate output buffer
    attn_output = torch.zeros(batch_size, seq_len, nheads, head_dim, dtype=torch.float16, device="cuda")

    # loop over heads one at a time, writing the result into the output buffer 
    for head in range(nheads):
        q = query[:, :, head, :]  # Shape: (batch_size, seq_len, head_dim)
        k = key[:, :, head, :]
        v = value[:, :, head, :]

        single_head_attn = slidingWindowAttn(q, k, v, window_size)

        attn_output[:, :, head, :] = single_head_attn

    return attn_output

#batch size, seq len, num_heads(1), embedding size
q = torch.randn(1, 8, 2, 64, dtype=torch.float16, device="cuda")
k = torch.randn(1, 8, 2, 64, dtype=torch.float16, device="cuda")
v = torch.randn(1, 8, 2, 64, dtype=torch.float16, device="cuda")
w = 2

print(f"setup: {w=}, \n{q.shape=} \n")

pytorch_attn_out = multiheadAttn(q,k,v,w)
print(f"{pytorch_attn_out.shape=}")
print(f"{pytorch_attn_out[0][0][0]=}")

window_size=(w,w) 
flash_attn_out = flash_attn_func(q, k, v, causal=False, window_size=window_size, dropout_p=0.0)
print(f"{flash_attn_out.shape=}") #flash_attn_out.shape=torch.Size([1, 8, 1, 64])
print(f"{flash_attn_out[0][0][0]=}")

are_close = torch.allclose(pytorch_attn_out, flash_attn_out,
        atol=1e-3, rtol=1e-2)
if are_close:
    print("pytorch_attn_out and flash_attn_out are numerically close.")
else:
    print("pytorch_attn_out and flash_attn_out differ.")

# also compare against pytorch sdpa, if we aren't using a sliding window
if (w == -1):
    print("Need to convert from Flash attn format to torch sdpa format")
    raise ValueError
    sdpa_attn_out = scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=False, dropout_p=0.0,)  # expects (batch heads grid variable) format
    sdpa_attn_out = torch.einsum('btsd -> bst d', sdpa_attn_out)
    print(f"{sdpa_attn_out.shape=}")
    print(f"{sdpa_attn_out[0][0]=}")
    are_close = torch.allclose(pytorch_attn_out, sdpa_attn_out, atol=1e-3, rtol=1e-2)
    if are_close:
        print("pytorch_attn_out and sdpa_attn_out are numerically close.")
    else:
        print("pytorch_attn_out and sdpa_attn_out differ.")
else:
    print("Sliding window not supported in Pytorch SDPA")

