import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from typing import Tuple, Optional
import math


class ScaledDotProductAttention(nn.Module):
    """
    Args: 
        d_k (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: 
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: 
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, d_k, dropout) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = math.sqrt(d_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        q: (batch_size * num_heads, q_len, d_k)
        k: (batch_size * num_heads, k_len, d_k)
        v: (batch_size * num_heads, v_len, d_v)
        mask: (batch_size * num_heads, q_len, k_len)
        
        Returns:
        - context: (batch_size * num_heads, q_len, d_v)
        - attn: (batch_size * num_heads, q_len, k_len)
        """
        score: Tensor = torch.bmm(input=q, mat2=k.transpose(dim0=1, dim1=2)) / self.d_k       # score: (batch_size, q_len, k_len)
        if mask is not None:
            score = score.masked_fill(mask=(mask == 0), value=-1e9)                           # score: (batch_size, q_len, k_len)
        
        attn = F.softmax(input=score, dim=-1)                                                 # attn: (batch_size, q_len, k_len)
        attn = self.dropout(attn)
        context = torch.bmm(input=attn, mat2=v)                                               # context: (batch_size, q_len, d_model)
        return context, attn
