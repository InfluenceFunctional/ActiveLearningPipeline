import math
import torch
import torch.nn as nn


class GFNTransformer(nn.Module):
    def __init__(self, num_tokens, num_outputs, num_hid,
                 num_layers, num_head, max_len=60, dropout=0.1,
                 partition_init=150.0,
                 causal=False):
        super().__init__()
        self.pos = PositionalEncoding(num_hid, dropout=dropout, max_len=max_len + 1)
        self.embedding = nn.Embedding(num_tokens, num_hid)
        encoder_layers = nn.TransformerEncoderLayer(num_hid, num_head, num_hid, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output = nn.Linear(num_hid, num_outputs)
        self.causal = causal
        # log(20**60 * 0.5**8) == 175
        #self.Z = nn.Parameter(torch.tensor([175.0], requires_grad=True))

        # splitting Z into more parameters means Adam will be able to
        # change Z faster, seems to help earlier but might be harder
        # at convergence?
        self._Z = nn.Parameter(torch.ones(64) * partition_init / 64)

    @property
    def Z(self):
        return self._Z.sum()

    def model_params(self):
        return list(self.pos.parameters()) + list(self.embedding.parameters()) + list(self.encoder.parameters()) + \
            list(self.output.parameters())

    def Z_param(self):
        return [self._Z]

    def forward(self, x, mask, return_all=False, lens=None):
        x = self.embedding(x)
        x = self.pos(x)
        if self.causal:
            x = self.encoder(x, src_key_padding_mask=mask,
                             mask=generate_square_subsequent_mask(x.shape[0]).to(x.device))
            pooled_x = x[lens, torch.arange(x.shape[1])]
        else:
            x = self.encoder(x, src_key_padding_mask=mask)
            pooled_x = x[0, :] # This is weird... but this is what BERT calls pooling?
        # indeed doing e.g. this:
        #_mask = (1-mask.float())
        #pooled_x = (x * _mask.T.unsqueeze(2)).sum(0) / _mask.sum(1).unsqueeze(1)
        # seems to no be as good? (Well, the max reward is lower but loss is similar..)
        if return_all:
            return self.output(x)
        y = self.output(pooled_x)
        return y


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# Taken from the PyTorch Transformer tutorial
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)