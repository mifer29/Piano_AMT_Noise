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


# GroupNorm helper — DataParallel-safe alternative to BatchNorm2d.
# BatchNorm2d computes statistics per-GPU shard which becomes unstable
# with small per-GPU batch sizes. GroupNorm normalizes per-sample so
# it is unaffected by the number of GPUs or batch size.

def _gn(num_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=8, num_channels=num_channels)


# CNN Frontend
# Processes mel spectrogram [B, T, n_mels] into transformer-ready
# features [B, T, d_model].
#
# Two freq-stride-2 blocks reduce the frequency dimension by 4x before
# the final linear projection, keeping the projection input small:
#   n_mels=512 -> 128 * (512//4) = 128 * 128 = 16384  (manageable)
#
# 1x1 conv shortcuts handle channel/spatial mismatches — no zero-padding.

class CNNFrontend(nn.Module):
    def __init__(self, n_mels: int, d_model: int):
        super().__init__()

        # Stem: 1 -> 32, no spatial change
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            _gn(32),
            nn.ReLU(),
        )

        # Block 1: 32 -> 32, freq /2
        self.block1_main = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            _gn(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=(2, 1), padding=1, bias=False),
            _gn(32),
        )
        self.block1_skip = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, stride=(2, 1), bias=False),
            _gn(32),
        )

        # Block 2: 32 -> 64, freq /2
        self.block2_main = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            _gn(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=(2, 1), padding=1, bias=False),
            _gn(64),
        )
        self.block2_skip = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=(2, 1), bias=False),
            _gn(64),
        )

        # Block 3: 64 -> 128, no stride
        self.block3_main = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            _gn(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            _gn(128),
        )
        self.block3_skip = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            _gn(128),
        )

        self.relu = nn.ReLU()

        # After 2x freq-stride-2: F_out = n_mels // 4
        self.proj = nn.Linear(128 * (n_mels // 4), d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, n_mels]
        x = x.unsqueeze(1).permute(0, 1, 3, 2)                    # [B, 1, F, T]
        x = self.stem(x)                                            # [B, 32, F, T]
        x = self.relu(self.block1_main(x) + self.block1_skip(x))   # [B, 32, F/2, T]
        x = self.relu(self.block2_main(x) + self.block2_skip(x))   # [B, 64, F/4, T]
        x = self.relu(self.block3_main(x) + self.block3_skip(x))   # [B, 128, F/4, T]
        x = x.permute(0, 3, 1, 2)                                  # [B, T, 128, F/4]
        x = x.reshape(x.size(0), x.size(1), -1)                    # [B, T, 128 * F/4]
        return self.proj(x)                                         # [B, T, d_model]


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

        self.frontend = CNNFrontend(n_mels, d_model)
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
        x = self.frontend(x)
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