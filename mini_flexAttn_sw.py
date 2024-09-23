import torch

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

import functools

if __name__ == "__main__":

    seq_lens= {
        "o96" : 40320,
        "o32" : 5248,
    }

    num_channels=256 #1024 works, 256 doesnt at fp16 on 16 heads
    num_heads=4
    head_dim= num_channels // num_heads

    B, H, SEQ_LEN, HEAD_DIM = 1, num_heads, seq_lens['o96'], head_dim
    WINDOW_SIZE = 512
    #PRECISION=torch.float32 #always works
    PRECISION=torch.float16
    DEVICE="cpu"

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=PRECISION, requires_grad=True)

    q, k, v = make_tensor(), make_tensor(), make_tensor()
    gradOut = torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=DEVICE, dtype=PRECISION)

    def sliding_window(b, h, q_idx, kv_idx):
        return torch.abs(q_idx - kv_idx) <= WINDOW_SIZE

    block_mask = create_block_mask(
        sliding_window, B=None, H=None, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN, _compile=True, device=DEVICE

    )
    #self.attention=flex_attention
    attention = functools.partial(flex_attention, block_mask=block_mask) #cache the block mask so its not remade

    attention = torch.compile(attention)

    out = attention(q, k, v, block_mask=block_mask)
    print(f"Shape of output tensor: {list(out.shape)}")

    out.backward(gradOut, retain_graph=True)
    print(f"Shape of output tensor after bw: {list(out.shape)}")
