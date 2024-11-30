from attention import ScaledDotProductAttention
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model should be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        # simplifying by d_k == d_v

        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.Wq = nn.Linear(d_model, self.d_k * num_heads, bias=False)
        self.Wk = nn.Linear(d_model, self.d_k * num_heads, bias=False)
        self.Wv = nn.Linear(d_model, self.d_v * num_heads, bias=False)
        self.Wo = nn.Linear(self.d_v * num_heads, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        q: (batch_size, q_len, d_model)
        k: (batch_size, k_len, d_model)
        v: (batch_size, v_len, d_model)
        """
        batch_size = q.shape[0]
        residual = q

        Q: Tensor = self.Wq(q).view(batch_size, -1, self.num_heads, self.d_k)  # (batch_size, seq_len, num_heads, d_k)
        K: Tensor = self.Wk(k).view(batch_size, -1, self.num_heads, self.d_k)  # (batch_size, seq_len, num_heads, d_k)
        V: Tensor = self.Wv(v).view(batch_size, -1, self.num_heads, self.d_v)  # (batch_size, seq_len, num_heads, d_v)


        Q = Q.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_k)  # (batch_size * num_heads, seq_len, d_k)
        K = K.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_k)  # (batch_size * num_heads, seq_len, d_k)
        V = V.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_v)  # (batch_size * num_heads, seq_len, d_v)
        
        # print(f"Q.shape: {Q.shape}, \nK.shape: {K.shape}, \nV.shape: {V.shape}\n")
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)            # (batch_size, num_heads, seq_len, seq_len)
            seq_len = K.shape[1]
            mask = mask.view(batch_size * self.num_heads, -1, seq_len)          # mask: (batch_size * num_heads, seq_len, seq_len)

        context, attn = self.attention(Q, K, V, mask)                           # context: (batch_size * num_heads, seq_len, d_v)
        context = context.view(batch_size, self.num_heads, -1, self.d_v)        # (batch_size, num_heads, seq_len, d_v)
        context = context.permute(0, 2, 1, 3).contiguous()                      # (batch_size, seq_len, num_heads, d_v)
        context = context.view(batch_size, -1, self.d_v * self.num_heads)       # (batch_size, seq_len, d_v * num_heads)

        output = self.dropout(self.Wo(context))                                 # (batch_size, seq_len, d_model)
        output = output + residual
        output = self.layer_norm(output)

        return output, attn
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        # Add & Norm
        output = self.w_2(self.dropout(F.relu(self.w_1(x))))
        output = self.dropout(output)
        output = output + residual
        output = self.layer_norm(output)

        return output
