import torch
import torch.nn as nn
from layer import EncoderLayer, DecoderLayer
from torch import Tensor


def get_pad_mask(seq: Tensor, pad_idx: int) -> Tensor:
    """Mask the padding part of the sequence.
    Args:
        seq: (batch_size, seq_len)
        pad_idx: int
    ---------
    return: 
        (batch_size, 1, seq_len)
    """
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq: Tensor) -> Tensor:
    ''' Create a mask to hide future words in a sequence.
    Args:
        seq: (batch_size, seq_len)
    ---------
    return: 
        (batch_size, seq_len, seq_len)
    '''
    batch_size, seq_len = seq.size()
    subsequent_mask = (1 - torch.triu(torch.ones((1, seq_len, seq_len), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

# with numpy:
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_hidden, n_position=200):
#   v,.e      super(PositionalEncoding, self).__init__()
        
#         # Not a parameter
#         self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hidden))

#     def _get_sinusoid_encoding_table(self, n_position, d_hidden):
#         ''' Sinusoid position encoding table '''
#         def get_position_angle_vec(position):
#             return [position / np.power(10000, 2 * (hid_j // 2) / d_hidden) for hid_j in range(d_hidden)]
#         sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
#         sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
#         sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

#         return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    
#     def forward(self, x):
#         return x + self.pos_table[:, :x.size(1)].clone().detach()

class PositionalEncoding(nn.Module):
    def __init__(self, d_hidden: int, n_position: int = 200) -> None:
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)  # (n_position, 1)
        div_term = torch.exp(torch.arange(0, d_hidden, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_hidden))

        pe = torch.zeros(n_position, d_hidden)
        pe[:, 0::2] = torch.sin(position * div_term)  # even : 2i
        pe[:, 1::2] = torch.cos(position * div_term)  # odd : 2i+1
        
        pe = pe.unsqueeze(0)  # (1, n_position, d_hidden)
        self.register_buffer('pos_table', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """
                x:  (batch_size, seq_len, d_hidden)
        pos_table:  (1, n_position, d_hidden)

        Ensure that seq_len <= n_position
        """
        return x + self.pos_table[:, :x.size(1)].requires_grad_(False)  # (batch_size, seq_len, d_hidden)
    

class Encoder(nn.Module):
    def __init__(
            self, src_vocab_size: int, embed_size: int, n_layers: int, num_heads: int,
            d_model: int, d_ff: int, pad_idx: int, dropout: float=0.1, n_position: int=200, scale_emb: bool=False
        ) -> None:
        super(Encoder, self).__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, embed_size, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(embed_size, n_position=n_position)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(n_layers)
        ])
        self.scale_emb = scale_emb
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, src_seq: Tensor, src_mask: Tensor, return_attns: bool=False):
        """
        src_seq: (batch_size, src_len)
        src_mask: (batch_size, src_len)
        """
        enc_attn_list = []

        src_emb = self.src_embedding(src_seq)                          # (batch_size, src_len, embed_size)
        if self.scale_emb:
            src_emb *= self.d_model ** 0.5
        src_emb = self.dropout(self.position_enc(src_emb))             # (batch_size, src_len, embed_size)

        enc_output = self.layer_norm(src_emb)                          # (batch_size, src_len, d_model)
        for layer in self.layers:
            enc_output, enc_attn = layer(enc_output, src_mask)
            enc_attn_list += [enc_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_attn_list
        return enc_output,

class Decoder(nn.Module):
    def __init__(
            self, tgt_vocab_size: int, embed_size: int, n_layers: int, num_heads: int,
            d_model: int, d_ff: int, pad_idx: int, dropout: float=0.1, n_position: int=200, scale_emb: bool=False
        ) -> None:
        super(Decoder, self).__init__()

        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_size, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(embed_size, n_position=n_position)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(n_layers)
        ])
        self.scale_emb = scale_emb
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.d_model = d_model

    def forward(self, tgt_seq: Tensor, tgt_mask: Tensor, enc_output: Tensor, src_mask: Tensor, return_attns: bool=False):
        """
        tgt_seq: (batch_size, tgt_len)
        tgt_mask: (batch_size, tgt_len)
        enc_output: (batch_size, src_len, d_model)
        src_mask: (batch_size, src_len)
        """
        dec_attn_list = []
        dec_enc_attn_list = []

        tgt_emb = self.tgt_embedding(tgt_seq)                          # (batch_size, tgt_len, embed_size)
        if self.scale_emb:
            tgt_emb *= self.d_model ** 0.5
        tgt_emb = self.dropout(self.position_enc(tgt_emb))             # (batch_size, tgt_len, embed_size)

        dec_output = self.layer_norm(tgt_emb)                          # (batch_size, tgt_len, d_model)
        for layer in self.layers:
            dec_output, dec_attn, dec_enc_attn = layer(
                dec_output, enc_output, self_attn_mask=tgt_mask, dec_enc_attn_mask=src_mask
                )
            dec_attn_list += [dec_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_attn_list, dec_enc_attn_list
        return dec_output,

class Transformer(nn.Module):
    """
    In section 3.4 of paper "Attention Is All You Need", there is such detail:
    "In our model, we share the same weight matrix between the two
    embedding layers and the pre-softmax linear transformation...
    In the embedding layers, we multiply those weights by sqrt{d_model}".
    
    Options here:
      'emb': multiply sqrt{d_model} to embedding output
      'prj': multiply (sqrt{d_model} ^ -1) to linear projection output
      'none': no multiplication
    """
    def __init__(
         self, src_vocab_size: int, tgt_vocab_size: int, src_pad_idx: int, tgt_pad_idx: int,
         embed_size: int=512, d_model: int=512, d_ff: int=2048, n_layers: int=6, num_heads: int=8, dropout: float=0.1,
         n_position: int=200, tgt_emb_prj_weight_sharing: bool=True, emb_src_tgt_weight_sharing: bool=True,
         scale_emb_or_prj: str='prj'
        ):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        assert scale_emb_or_prj in ['emb', 'prj', 'none'], "Invalid value for scale_emb_or_prj, should be 'emb', 'prj', or 'none'"
        scale_emb = (scale_emb_or_prj == 'emb') if tgt_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if tgt_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            src_vocab_size, embed_size, n_layers,
            num_heads, d_model, d_ff, src_pad_idx,
            dropout, n_position, scale_emb
            )
        self.decoder = Decoder(
            tgt_vocab_size, embed_size, n_layers,
            num_heads, d_model, d_ff, tgt_pad_idx,
            dropout, n_position, scale_emb
            )
        
        self.tgt_word_prj = nn.Linear(d_model, tgt_vocab_size, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == embed_size, \
        "To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same."
        
        if emb_src_tgt_weight_sharing:
            self.encoder.src_embedding.weight = self.decoder.tgt_embedding.weight

        if tgt_emb_prj_weight_sharing:
            self.tgt_word_prj.weight = self.decoder.tgt_embedding.weight

    def forward(self, src_seq: Tensor, tgt_seq: Tensor):
        """
        src_seq: (batch_size, src_len)
        tgt_seq: (batch_size, tgt_len)
        """
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        tgt_mask = get_pad_mask(tgt_seq, self.tgt_pad_idx) & get_subsequent_mask(tgt_seq)
        print("src_mask: ", src_mask.shape)
        print("src_mask: ", src_mask)
        print("tgt_mask: ", tgt_mask.shape)
        print("tgt_mask: ", tgt_mask)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(tgt_seq, tgt_mask, enc_output, src_mask)

        seq_logit = self.tgt_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2))
    
if __name__ == '__main__':
    model = Transformer(
        src_vocab_size=100, tgt_vocab_size=100, src_pad_idx=0, tgt_pad_idx=0,
        embed_size=512, d_model=512, d_ff=2048, n_layers=2, num_heads=8, dropout=0.1,
        n_position=200, tgt_emb_prj_weight_sharing=True, emb_src_tgt_weight_sharing=True,
        scale_emb_or_prj='prj'
    )
    # src_seq = torch.randint(0, 100, (4, 10))  # (batch_size, src_len)
    src_seq = torch.tensor([
        [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
        [6, 7, 8, 9, 0, 0, 0, 0, 0, 0],
        [10, 11, 12, 13, 14, 15, 0, 0, 0, 0],
        [16, 17, 18, 19, 20, 21, 22, 0, 0, 0]
    ])
    tgt_seq = torch.randint(0, 100, (4, 15))
    output = model(src_seq, tgt_seq)
    print(model)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: {params}")
