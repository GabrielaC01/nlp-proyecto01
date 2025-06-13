"""
Este archivo contiene la implementación de un modelo Transformer autoregresivo
con máscara causal para predicción de texto token por token.
Basado en los modelos vistos en clase (como GPT) y adaptado para el proyecto 1.
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class ModeloAutoregresivo(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def generate_causal_mask(self, size):
        # Máscara para evitar que un token vea los futuros
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def forward(self, x):
        emb = self.embedding(x) * math.sqrt(self.d_model)
        emb = self.pos_encoder(emb)
        tgt_mask = self.generate_causal_mask(x.size(1)).to(x.device)
        output = self.transformer_decoder(emb, emb, tgt_mask=tgt_mask)
        logits = self.fc_out(output)
        return logits
