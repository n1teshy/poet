import torch
import torch.nn as nn

from core.config import device
from core.components.cnn import BasicBlock, BottleNeck, ResNet
from core.components.embedding import TokenEmbedding
from core.components.transformer import Encoder, Decoder


class Generator(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        max_len,
        enc_layers,
        enc_heads,
        src_pad_id,
    ):
        super().__init__()
        self.pad_id = src_pad_id
        self.emb = TokenEmbedding(vocab_size, embedding_size, max_len)
        self.encoder = Encoder(embedding_size, enc_layers, enc_heads)
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        x_mask = self.get_masks(x)
        x_emb = self.emb(x)
        out = self.encoder(x_emb, x_mask)
        return self.linear(out)

    def get_masks(self, x):
        seq_len = x.shape[1]
        pad_mask = (x != self.tgt_pad_id).unsqueeze(1)
        lh_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.long, device=device)
        )
        return pad_mask & lh_mask
