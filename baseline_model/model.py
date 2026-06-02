#model.py

import math
import torch
import torch.nn as nn


# Positional Encoding (absolute sinusoidal)
# transformers do not understand order and therefore a positional encoding is added.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)].to(x.dtype)


# Scale residual branch outputs to prevent variance explosion with depth.
# Based on GPT-2: multiply std of last linear in each sublayer by 1/sqrt(2*num_layers)

def scale_residual_init(module, num_layers):
    std = 0.02 / math.sqrt(2 * num_layers)
    for name, p in module.named_parameters():
        if "linear2.weight" in name or "out_proj.weight" in name:
            nn.init.normal_(p, mean=0.0, std=std)


# Encoder

class AudioEncoder(nn.Module):
    def __init__(self,
                 n_mels: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 dropout: float):
        super().__init__()

        # Projecting mel features
        self.input_projection = nn.Linear(n_mels, d_model)

        # Add positional encoding
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        return self.encoder(x)


# Decoder: generates midi tokens autoregressively

class EventDecoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 dropout: float):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        self.positional_encoding = PositionalEncoding(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.output_projection = nn.Linear(d_model, vocab_size)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self,
                tgt_tokens: torch.Tensor,
                memory: torch.Tensor) -> torch.Tensor:

        B, L = tgt_tokens.shape


        # Proper float causal mask: future tokens are blocked
        tgt_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=tgt_tokens.device),
            diagonal=1
        )

        # Padding mask: we ignore the padding tokens (PAD token = 0)
        tgt_key_padding_mask = torch.zeros(
            B, L, device=tgt_tokens.device, dtype=torch.float32
        ).masked_fill(tgt_tokens == 0, float("-inf"))

        tgt_emb = self.token_embedding(tgt_tokens)

        tgt_emb = self.positional_encoding(tgt_emb)

        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        return self.output_projection(out)


# Full model

class AudioTransformer(nn.Module):

    def __init__(self,
                 n_mels: int,
                 vocab_size: int,
                 d_model: int = 512,
                 num_encoder_layers: int = 8,
                 num_decoder_layers: int = 8,
                 num_heads: int = 8,
                 d_ff: int = 1024,
                 dropout: float = 0.1):

        super().__init__()

        self.encoder = AudioEncoder(
            n_mels=n_mels,
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )

        self.decoder = EventDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout
        )

        scale_residual_init(self.encoder, num_encoder_layers)
        scale_residual_init(self.decoder, num_decoder_layers)

    def forward(self,
                audio_features: torch.Tensor,
                tgt_tokens: torch.Tensor) -> torch.Tensor:

        memory = self.encoder(audio_features)
        logits = self.decoder(tgt_tokens, memory)

        return logits



# Teacher Forcing (used during training)

def shift_tokens_right(target_ids: torch.Tensor,
                       bos_id: int) -> torch.Tensor:

    shifted = target_ids.clone()
    shifted[:, 1:] = target_ids[:, :-1]
    shifted[:, 0] = bos_id

    return shifted


# Greedy Decoding (used during inference)

@torch.no_grad()
def greedy_decode(model: AudioTransformer,
                  audio_features: torch.Tensor,
                  bos_id: int,
                  eos_id: int,
                  max_len: int = 1024):

    model.eval()
    device = audio_features.device

    memory = model.encoder(audio_features)

    decoded = [bos_id]

    for _ in range(max_len):

        tgt = torch.tensor([decoded],
                           dtype=torch.long,
                           device=device)

        logits = model.decoder(tgt, memory)

        next_id = logits[0, -1].argmax().item()

        decoded.append(next_id)

        if next_id == eos_id:
            break

    return decoded[1:]  # remove BOS