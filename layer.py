import torch.nn as nn
from sublayer import MultiHeadAttention, PositionwiseFeedForward
from torch import Tensor
from typing import Optional
from typing import Tuple


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, dropout: float) -> None:
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, enc_input: Tensor, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        enc_output, attn = self.attention(enc_input, enc_input, enc_input, attn_mask)
        enc_output = self.feed_forward(enc_output)
        return enc_output, attn
    


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, dropout: float) -> None:
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_dec_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)


    def forward(self, dec_input: Tensor, enc_output: Tensor, self_attn_mask: Optional[Tensor] = None, dec_enc_attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        dec_output, self_attn = self.self_attention(dec_input, dec_input, dec_input, self_attn_mask)
        dec_output, dec_enc_attn = self.enc_dec_attention(dec_output, enc_output, enc_output, dec_enc_attn_mask)
        dec_output = self.feed_forward(dec_output)
        return dec_output, self_attn, dec_enc_attn
