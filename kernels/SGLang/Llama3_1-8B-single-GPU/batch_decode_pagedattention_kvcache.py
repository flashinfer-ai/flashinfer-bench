"""
The implementation of paged attention using flashinfer and native PyTorch.
Based on real SGLang trace data from Llama3.1 inference.

Shapes:
- q: (batch_size, num_qo_heads, head_dim)
- k_cache: (total_pages, page_size, num_kv_heads, head_dim) 
- v_cache: (total_pages, page_size, num_kv_heads, head_dim)
- kv_indptr: (batch_size + 1,) - cumulative page counts per sequence
- kv_indices: (total_used_pages,) - actual page indices being used
- kv_last_page_len: (batch_size,) - number of tokens in last page per sequence
"""

import os
import torch
import math
from typing import Optional, Tuple
import flashinfer
from torch.nn.functional import scaled_dot_product_attention


def forward_flashinfer(
    q: torch.Tensor,
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    page_size: int,
    sm_scale: float,
    logits_soft_cap: float = 0.0,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    FlashInfer paged attention implementation.
    """
    batch_size, num_qo_heads, head_dim = q.shape
    
    _, _, num_kv_heads, _ = k_cache.shape
    
    # https://github.com/sgl-project/sglang/blob/93b6785d78d1225b65672979795566a9d73b6e47/python/sglang/global_config.py#L37
    workspace_buffer = torch.empty(348 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout="NHD",
        use_tensor_cores=False, # from tracing
    )
    
    wrapper.plan(
        indptr=kv_indptr,
        indices=kv_indices,
        last_page_len=kv_last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode="NONE",
        window_left=-1,
        logits_soft_cap=logits_soft_cap,
        q_data_type=q.dtype,
        kv_data_type=k_cache.dtype,
        sm_scale=sm_scale,
        non_blocking=True,
    )
    
    output = wrapper.run(
        q=q,
        paged_kv_cache=(k_cache, v_cache),
        q_scale=k_scale,
        k_scale=k_scale, 
        v_scale=v_scale,
        return_lse=False,
    )
    
    return output


def forward_pytorch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor, 
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    page_size: int,
    sm_scale: float,
    logits_soft_cap: float = 0.0,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Adapted from SGLang's https://github.com/sgl-project/sglang/blob/93b6785d78d1225b65672979795566a9d73b6e47/python/sglang/srt/layers/attention/torch_native_backend.py
    """
    batch_size, num_qo_heads, head_dim = q.shape
    
    num_kv_heads = k_cache.shape[2]
    
    output = torch.empty_like(q)
    
    for seq_idx in range(batch_size):
        page_start = kv_indptr[seq_idx].item()
        page_end = kv_indptr[seq_idx + 1].item()
        seq_pages = kv_indices[page_start:page_end]
        
        num_pages = len(seq_pages)
        if num_pages == 0:
            output[seq_idx] = 0
            continue
        
        last_page_len = kv_last_page_len[seq_idx].item()
        seq_len = (num_pages - 1) * page_size + last_page_len
        
        seq_k_pages = k_cache[seq_pages]
        seq_v_pages = v_cache[seq_pages]
        
        # (seq_len, num_kv_heads, head_dim)
        seq_k = seq_k_pages.reshape(-1, num_kv_heads, head_dim)[:seq_len]
        seq_v = seq_v_pages.reshape(-1, num_kv_heads, head_dim)[:seq_len]
        seq_q = q[seq_idx]
        
        if num_qo_heads != num_kv_heads:
            kv_head_ratio = num_qo_heads // num_kv_heads
            # (seq_len, num_qo_heads, head_dim)
            seq_k = seq_k.repeat_interleave(kv_head_ratio, dim=1)
            seq_v = seq_v.repeat_interleave(kv_head_ratio, dim=1)
        
        seq_q = seq_q.unsqueeze(0).unsqueeze(2)     # (1, num_qo_heads, 1, head_dim)
        seq_k = seq_k.transpose(0, 1).unsqueeze(0)  # (1, num_qo_heads, seq_len, head_dim)
        seq_v = seq_v.transpose(0, 1).unsqueeze(0)  # (1, num_qo_heads, seq_len, head_dim)
        
        effective_sm_scale = sm_scale
        if k_scale is not None:
            effective_sm_scale *= k_scale
        
        seq_output = scaled_dot_product_attention(
            seq_q,
            seq_k, 
            seq_v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=effective_sm_scale,
        )
        
        seq_output = seq_output.squeeze(0).squeeze(1)        
        if v_scale is not None:
            seq_output *= v_scale
            
        output[seq_idx] = seq_output
    
    return output


def kernel_descriptions():
    return """
paged_attention(q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, page_size, sm_scale)

Shape Variables:
- batch_size: number of sequences in batch
- num_qo_heads: number of query/output heads  
- num_kv_heads: number of key/value heads
- head_dim: dimension of each attention head
- total_pages: total number of pages in KV cache
- total_used_pages: number of pages actually used by current batch
- page_size: number of tokens per page

Input shapes:
- q: (batch_size, num_qo_heads, head_dim)
- k_cache: (total_pages, page_size, num_kv_heads, head_dim)
- v_cache: (total_pages, page_size, num_kv_heads, head_dim)  
- kv_indptr: (batch_size + 1,) - cumulative page counts
- kv_indices: (total_used_pages,) - page indices
- kv_last_page_len: (batch_size,) - tokens in last page per sequence

Computation:
- Paged attention with KV cache lookup
- Supports grouped query attention (GQA)
- Uses scaled dot-product attention with custom scaling
"""


if __name__ == "__main__":
    os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0" #B200
    
    # traced from Llama3.1 SGLang single gpu no tp/pp
    BATCH_SIZE = 2
    NUM_QO_HEADS = 32
    NUM_KV_HEADS = 8
    HEAD_DIM = 128
    PAGE_SIZE = 1
    SM_SCALE = 0.08838834764831845
    LOGITS_SOFT_CAP = 0.0
    DTYPE = torch.bfloat16
    DEVICE = "cuda"
    
    TOTAL_PAGES = 862424
    TOTAL_USED_PAGES = 20
    
    
    q = torch.randn(BATCH_SIZE, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    
    k_cache = torch.randn(TOTAL_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v_cache = torch.randn(TOTAL_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    
    # Traced from prompt, can experiment with more prompts, differnet # of batched sequences, and adjust indptr and indices
    # prompts = [
    #    "Hello, my name is",
    #    "The capital of France is",
    #]
    kv_indptr = torch.tensor([0, 10, 20], dtype=torch.int32, device=DEVICE)
    kv_indices = torch.tensor([1, 2, 3, 4, 5, 6, 13, 15, 17, 19, 1, 8, 9, 10, 11, 12, 14, 16, 18, 20], 
                             dtype=torch.int32, device=DEVICE)
    kv_last_page_len = torch.tensor([1, 1], dtype=torch.int32, device=DEVICE)
    
    print("batch decode (paged attention kv cache) PyTorch")
    out_pytorch = forward_pytorch(
        q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len,
        PAGE_SIZE, SM_SCALE, LOGITS_SOFT_CAP
    )
    
    print("batch decode (paged attention kv cache) FlashInfer")  
    out_flashinfer = forward_flashinfer(
        q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len,
        PAGE_SIZE, SM_SCALE, LOGITS_SOFT_CAP
    )
    
    print(f"PyTorch output shape: {out_pytorch.shape}")
    print(f"FlashInfer output shape: {out_flashinfer.shape}")
    
    try:
        torch.testing.assert_close(out_pytorch, out_flashinfer, rtol=1e-2, atol=1e-2)
        print(f"pytorch and flashinfer outputs match")
    except AssertionError as e:
        print(f"pytorch and flashinfer outputs don't match: {e}")