"""Decoder model class definition using PyTorch."""

import math

import torch
from torch import nn


class DecoderBlock(nn.Module):
    """
    Decoder-only Transformer block.

    Architecture:

        LayerNorm
            ↓
        Causal Self Attention
            ↓
        Residual
            ↓
        LayerNorm
            ↓
        Feed Forward Network
            ↓
        Residual
    """

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask, padding_mask=None):
        h = self.norm1(x)
        attention_output, _ = self.attn(
            h,
            h,
            h,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attention_output)

        h = self.norm2(x)
        ffn_out = self.ffn(h)
        x = x + self.dropout(ffn_out)
        return x


class DecoderTransformerClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_len=128,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1,
        num_classes=4,
    ):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_len:
            raise ValueError(
                f"sequence length {seq_len} "
                f"exceeds max_len {self.max_len}"
            )

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        ).unsqueeze(0)

        embedding_x = self.token_emb(input_ids)
        embedding_x = embedding_x * math.sqrt(self.d_model)
        pe_x = embedding_x + self.pos_emb(positions)
        x = self.dropout(pe_x)

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=input_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        key_padding_mask = ~attention_mask.bool()

        for block in self.blocks:
            x = block(
                x,
                causal_mask,
                key_padding_mask,
            )

        x = self.norm(x)

        lengths = (
            attention_mask.long().sum(dim=1).clamp_min(1) - 1
        )

        pooled = x[
            torch.arange(
                batch_size,
                device=input_ids.device,
            ),
            lengths,
        ]

        return self.classifier(pooled)
