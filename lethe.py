"""One-file MVP for testing whether 4x VQ compression reduces total LM compute."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


RAW_BLOCK = 64
CODES_PER_BLOCK = 16
CODEBOOK_SIZE = 1024
COMPRESSION = RAW_BLOCK // CODES_PER_BLOCK
MASK_TOKEN_BASE = CODEBOOK_SIZE
REFINED_VOCAB_SIZE = CODEBOOK_SIZE + 256

SEMANTIC_STOPWORDS = set(
    "a an the and or but so because as at by for from in into of on onto to with is am are was were "
    "be been being have has had do does did can could would should will shall may might must it its "
    "he him his she her hers they them their theirs we us our ours i me my mine you your yours this "
    "that these those there here very then than once upon one day after before when while who which "
    "what where why how".split()
)
SEMANTIC_SENTENCE_STARTERS = set(
    "Once One The A An Then Suddenly After Before When While He She They It Can Her His Their Together Yes No".split()
)


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int


@dataclass
class AEConfig:
    vocab_size: int
    n_embd: int = 192
    n_layer: int = 3
    n_head: int = 6
    code_dim: int = 10
    codebook_size: int = CODEBOOK_SIZE
    commitment: float = 0.1
    quantizer: str = "fsq"
    diversity_weight: float = 1.0
    decoder_type: str = "autoregressive"
    decoder_n_embd: int = 256
    decoder_n_layer: int = 6
    decoder_n_head: int = 8
    decoder_context: int = 0
    token_dropout: float = 0.75
    bos_token_id: int = 1
    ema_decay: float = 0.99
    dead_code_threshold: float = 0.1
    dead_code_interval: int = 100
    refinement: bool = False
    refinement_rate: float = 0.25
    refinement_threshold: float = 0.0
    priority_head: bool = False


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def apply_rope(x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
    _, _, length, dim = x.shape
    half = dim // 2
    frequencies = 1.0 / (10000 ** (torch.arange(half, device=x.device).float() / half))
    if positions is None:
        positions = torch.arange(length, device=x.device)
    angles = torch.outer(positions.to(device=x.device, dtype=torch.float), frequencies).to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        assert n_embd % n_head == 0 and (n_embd // n_head) % 2 == 0
        self.n_head = n_head
        self.norm1 = RMSNorm(n_embd)
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.norm2 = RMSNorm(n_embd)
        hidden = int(8 * n_embd / 3)
        self.w1 = nn.Linear(n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, n_embd, bias=False)
        self.w3 = nn.Linear(n_embd, hidden, bias=False)

    def forward(
        self, x: torch.Tensor, causal: bool, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        shape = (batch, length, self.n_head, width // self.n_head)
        q, k, v = (part.view(shape).transpose(1, 2) for part in (q, k, v))
        attention = F.scaled_dot_product_attention(
            apply_rope(q, positions), apply_rope(k, positions), v, is_causal=causal
        )
        x = x + self.proj(attention.transpose(1, 2).contiguous().view(batch, length, width))
        h = self.norm2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(Block(config.n_embd, config.n_head) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.embedding.weight
        self.apply(init_weights)

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] > self.config.block_size:
            raise ValueError(f"sequence length {tokens.shape[1]} exceeds {self.config.block_size}")
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x, causal=True)
        return self.norm(x)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.head(self.hidden(tokens))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, count: int, temperature: float = 0.8, top_k: int = 50):
        for _ in range(count):
            logits, _ = self(tokens[:, -self.config.block_size :])
            logits = logits[:, -1] / temperature
            cutoff = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1, None]
            logits = logits.masked_fill(logits < cutoff, -torch.inf)
            tokens = torch.cat((tokens, torch.multinomial(F.softmax(logits, -1), 1)), dim=1)
        return tokens

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class CorrectionHead(nn.Module):
    def __init__(self, model_width: int, slots: int, bottleneck: int, vocab_size: int):
        super().__init__()
        self.slots = slots
        self.keys = nn.Linear(model_width, bottleneck, bias=False)
        self.queries = nn.Parameter(torch.randn(slots, bottleneck) * 0.02)
        self.apply_head = nn.Linear(bottleneck, 1)
        self.position = nn.Linear(bottleneck, RAW_BLOCK)
        self.token = nn.Linear(bottleneck, vocab_size)
        self.apply(init_weights)

    def forward(self, hidden: torch.Tensor):
        keys = self.keys(hidden[:, -CODES_PER_BLOCK:])
        attention = torch.softmax(
            torch.einsum("bsd,kd->bks", keys, self.queries) / math.sqrt(keys.shape[-1]), dim=-1
        )
        states = torch.tanh(torch.einsum("bks,bsd->bkd", attention, keys) + self.queries)
        return self.apply_head(states).squeeze(-1), self.position(states), self.token(states)


class VQAutoencoder(nn.Module):
    def __init__(self, config: AEConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.encoder = nn.ModuleList(Block(config.n_embd, config.n_head) for _ in range(config.n_layer))
        self.to_latent = nn.Linear(COMPRESSION * config.n_embd, config.code_dim)
        if config.refinement:
            self.to_refinement = nn.Linear(COMPRESSION * config.n_embd, config.code_dim)
        if config.quantizer == "fsq":
            if 2**config.code_dim != config.codebook_size:
                raise ValueError("binary FSQ requires codebook_size == 2 ** code_dim")
            self.register_buffer("bit_weights", 2 ** torch.arange(config.code_dim, dtype=torch.long))
        else:
            self.codebook = nn.Embedding(config.codebook_size, config.code_dim)
        if config.decoder_type == "parallel":
            self.from_latent = nn.Linear(config.code_dim, COMPRESSION * config.n_embd)
            self.decoder = nn.ModuleList(Block(config.n_embd, config.n_head) for _ in range(config.n_layer))
            self.norm = RMSNorm(config.n_embd)
            self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.head.weight = self.embedding.weight
        else:
            self.decoder_embedding = nn.Embedding(config.vocab_size, config.decoder_n_embd)
            self.decoder_code_projection = nn.Linear(config.code_dim, config.decoder_n_embd, bias=False)
            if config.refinement:
                self.decoder_refinement_projection = nn.Linear(
                    config.code_dim, config.decoder_n_embd, bias=False
                )
            self.decoder = nn.ModuleList(
                Block(config.decoder_n_embd, config.decoder_n_head) for _ in range(config.decoder_n_layer)
            )
            self.norm = RMSNorm(config.decoder_n_embd)
            self.head = nn.Linear(config.decoder_n_embd, config.vocab_size, bias=False)
            self.head.weight = self.decoder_embedding.weight
        if config.priority_head:
            self.priority_head = nn.Linear(config.code_dim, COMPRESSION * config.vocab_size)
        self.apply(init_weights)
        if config.quantizer == "vq":
            self.codebook.weight.requires_grad_(config.ema_decay == 0)
            initial_sum = self.codebook.weight.detach().clone()
        else:
            initial_sum = torch.zeros(config.codebook_size, config.code_dim)
        self.register_buffer("ema_count", torch.ones(config.codebook_size))
        self.register_buffer("ema_sum", initial_sum)
        self.register_buffer("ema_initialized", torch.tensor(config.quantizer == "fsq" or config.ema_decay == 0))
        self.register_buffer("ema_steps", torch.tensor(0, dtype=torch.long))

    def encode_groups(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] != RAW_BLOCK:
            raise ValueError(f"autoencoder expects exactly {RAW_BLOCK} tokens")
        x = self.embedding(tokens)
        for block in self.encoder:
            x = block(x, causal=False)
        return x.reshape(x.shape[0], CODES_PER_BLOCK, -1)

    def encode_vectors(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encode_groups(tokens))

    def encode_refinement_vectors(self, tokens: torch.Tensor):
        if not self.config.refinement:
            raise ValueError("refinement modules are not enabled")
        return self.quantize(self.to_refinement(self.encode_groups(tokens)))

    def quantize(self, encoded: torch.Tensor):
        if self.config.quantizer == "fsq":
            bounded = torch.tanh(encoded.float())
            hard = torch.where(bounded >= 0, 1.0, -1.0)
            quantized = bounded + (hard - bounded).detach()
            ids = (((hard > 0).long()) * self.bit_weights).sum(-1)
            flat = torch.tanh(encoded.float() / 0.1).flatten(0, 1)
            means = flat.mean(0)
            centered = flat - means
            covariance = centered.T @ centered / max(1, flat.shape[0] - 1)
            off_diagonal = covariance - torch.diag(covariance.diag())
            diversity = means.square().mean() + off_diagonal.square().mean()
            commitment = F.mse_loss(bounded, hard.detach())
            return quantized.to(encoded.dtype), ids, (
                self.config.commitment * commitment + self.config.diversity_weight * diversity
            )
        flat = encoded.float().flatten(0, 1)
        if self.training and self.config.ema_decay and not bool(self.ema_initialized):
            self._initialize_codebook(flat)
        codebook = self.codebook.weight.float()
        distances = flat.pow(2).sum(1, keepdim=True) + codebook.pow(2).sum(1) - 2 * flat @ codebook.T
        ids = distances.argmin(1).view(encoded.shape[:2])
        if self.training and self.config.ema_decay:
            self._update_codebook(flat.detach(), ids.flatten())
        quantized = self.codebook(ids)
        commitment_loss = F.mse_loss(encoded, quantized.detach())
        straight_through = encoded + (quantized - encoded).detach()
        if self.config.ema_decay:
            vq_loss = self.config.commitment * commitment_loss
        else:
            vq_loss = F.mse_loss(quantized, encoded.detach()) + self.config.commitment * commitment_loss
        return straight_through, ids, vq_loss

    def codes_to_vectors(self, codes: torch.Tensor) -> torch.Tensor:
        if self.config.quantizer == "fsq":
            bits = ((codes[..., None] & self.bit_weights) > 0).to(self.decoder_code_projection.weight.dtype)
            return bits.mul(2).sub(1)
        return self.codebook(codes)

    def priority_logits(self, quantized: torch.Tensor) -> torch.Tensor:
        if not self.config.priority_head:
            raise ValueError("priority head is not enabled")
        return self.priority_head(quantized).view(
            quantized.shape[0], RAW_BLOCK, self.config.vocab_size
        )

    @torch.no_grad()
    def _initialize_codebook(self, flat: torch.Tensor) -> None:
        choices = torch.randint(flat.shape[0], (self.config.codebook_size,), device=flat.device)
        centers = flat[choices] + 1e-3 * torch.randn_like(flat[choices])
        self.codebook.weight.copy_(centers.to(self.codebook.weight.dtype))
        self.ema_sum.copy_(centers)
        self.ema_count.fill_(1)
        self.ema_initialized.fill_(True)

    @torch.no_grad()
    def _update_codebook(self, flat: torch.Tensor, ids: torch.Tensor) -> None:
        counts = torch.bincount(ids, minlength=self.config.codebook_size).float()
        sums = torch.zeros_like(self.ema_sum)
        sums.index_add_(0, ids, flat)
        decay = self.config.ema_decay
        self.ema_count.mul_(decay).add_(counts, alpha=1 - decay)
        self.ema_sum.mul_(decay).add_(sums, alpha=1 - decay)
        self.codebook.weight.copy_((self.ema_sum / self.ema_count.clamp_min(1e-5)[:, None]).to(self.codebook.weight))
        self.ema_steps.add_(1)
        if int(self.ema_steps) % self.config.dead_code_interval:
            return
        dead = self.ema_count < self.config.dead_code_threshold
        if not bool(dead.any()):
            return
        replacements = flat[torch.randint(flat.shape[0], (int(dead.sum()),), device=flat.device)]
        self.codebook.weight[dead] = replacements.to(self.codebook.weight)
        self.ema_sum[dead] = replacements
        self.ema_count[dead] = 1

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if self.config.decoder_type != "parallel":
            raise ValueError("autoregressive decoders return token IDs through reconstruct()")
        if codes.shape[1] != CODES_PER_BLOCK:
            raise ValueError(f"decoder expects exactly {CODES_PER_BLOCK} codes")
        x = self.from_latent(self.codebook(codes)).reshape(codes.shape[0], RAW_BLOCK, self.config.n_embd)
        for block in self.decoder:
            x = block(x, causal=False)
        return self.head(self.norm(x))

    def _decode_teacher(
        self,
        quantized: torch.Tensor,
        tokens: torch.Tensor,
        history: torch.Tensor | None = None,
        token_dropout: float | None = None,
        refinement: torch.Tensor | None = None,
        refinement_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        previous = torch.cat(
            (torch.full_like(tokens[:, :1], self.config.bos_token_id), tokens[:, :-1]), dim=1
        )
        dropout = self.config.token_dropout if token_dropout is None else token_dropout
        if self.training and dropout:
            drop = torch.rand_like(previous.float()) < dropout
            drop[:, 0] = False
            previous = previous.masked_fill(drop, 0)
        condition = self.decoder_code_projection(quantized)
        if refinement is not None:
            condition = condition + self.decoder_refinement_projection(refinement) * refinement_mask[..., None]
        prefix = [condition]
        if history is not None:
            prefix.append(self.decoder_embedding(history))
        offset = sum(part.shape[1] for part in prefix)
        x = torch.cat((*prefix, self.decoder_embedding(previous)), dim=1)
        positions = self._decoder_positions(0 if history is None else history.shape[1], previous.shape[1], x.device)
        for block in self.decoder:
            x = block(x, causal=True, positions=positions)
        return self.head(self.norm(x[:, offset:]))

    @staticmethod
    def _decoder_positions(history: int, current: int, device: torch.device) -> torch.Tensor:
        if not history:
            return torch.arange(CODES_PER_BLOCK + current, device=device)
        return torch.cat(
            (
                torch.arange(CODES_PER_BLOCK, device=device),
                torch.arange(-history, 0, device=device),
                torch.arange(CODES_PER_BLOCK, CODES_PER_BLOCK + current, device=device),
            )
        )

    def decoder_logits(
        self,
        codes: torch.Tensor,
        history: torch.Tensor,
        tokens: torch.Tensor,
        token_dropout: float = 0.0,
        refinement_codes: torch.Tensor | None = None,
        refinement_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        refinement = None if refinement_codes is None else self.codes_to_vectors(refinement_codes)
        return self._decode_teacher(
            self.codes_to_vectors(codes),
            tokens,
            history,
            token_dropout,
            refinement,
            refinement_mask,
        )

    def forward(self, tokens: torch.Tensor):
        encoded = self.encode_vectors(tokens)
        quantized, codes, vq_loss = self.quantize(encoded)
        if self.config.decoder_type == "parallel":
            x = self.from_latent(quantized).reshape(tokens.shape[0], RAW_BLOCK, self.config.n_embd)
            for block in self.decoder:
                x = block(x, causal=False)
            logits = self.head(self.norm(x))
        else:
            logits = self._decode_teacher(quantized, tokens)
        reconstruction_loss = F.cross_entropy(logits.flatten(0, 1), tokens.flatten())
        return logits, codes, reconstruction_loss, vq_loss

    @torch.no_grad()
    def reconstruct(
        self,
        codes: torch.Tensor,
        history: torch.Tensor | None = None,
        refinement_codes: torch.Tensor | None = None,
        refinement_mask: torch.Tensor | None = None,
        forced_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if codes.shape[1] != CODES_PER_BLOCK:
            raise ValueError(f"decoder expects exactly {CODES_PER_BLOCK} codes")
        if self.config.decoder_type == "parallel":
            return self.decode_codes(codes).argmax(-1)
        condition = self.decoder_code_projection(self.codes_to_vectors(codes))
        if refinement_codes is not None:
            condition = condition + self.decoder_refinement_projection(
                self.codes_to_vectors(refinement_codes)
            ) * refinement_mask[..., None]
        if history is None and self.config.decoder_context:
            history = torch.full(
                (codes.shape[0], self.config.decoder_context),
                self.config.bos_token_id,
                dtype=torch.long,
                device=codes.device,
            )
        prefix = condition if history is None else torch.cat((condition, self.decoder_embedding(history)), dim=1)
        generated = torch.full(
            (codes.shape[0], 1), self.config.bos_token_id, dtype=torch.long, device=codes.device
        )
        for position in range(RAW_BLOCK):
            x = torch.cat((prefix, self.decoder_embedding(generated)), dim=1)
            positions = self._decoder_positions(
                0 if history is None else history.shape[1], generated.shape[1], x.device
            )
            for block in self.decoder:
                x = block(x, causal=True, positions=positions)
            logits = self.head(self.norm(x[:, -1]))
            next_token = logits.argmax(-1)
            if forced_tokens is not None:
                forced = forced_tokens[:, position]
                next_token = torch.where(forced >= 0, forced, next_token)
            generated = torch.cat((generated, next_token[:, None]), dim=1)
        return generated[:, 1:]

    @torch.no_grad()
    def reconstruct_beam(
        self, codes: torch.Tensor, history: torch.Tensor | None = None, width: int = 4
    ) -> torch.Tensor:
        if codes.shape[0] != 1 or width < 1:
            raise ValueError("beam reconstruction expects one block and width >= 1")
        if width == 1:
            return self.reconstruct(codes, history)
        condition = self.decoder_code_projection(self.codes_to_vectors(codes))
        if history is None and self.config.decoder_context:
            history = torch.full(
                (1, self.config.decoder_context), self.config.bos_token_id, dtype=torch.long, device=codes.device
            )
        prefix = condition if history is None else torch.cat((condition, self.decoder_embedding(history)), dim=1)
        sequences = torch.full((1, 1), self.config.bos_token_id, dtype=torch.long, device=codes.device)
        scores = torch.zeros(1, device=codes.device)
        for _ in range(RAW_BLOCK):
            repeated_prefix = prefix.expand(len(sequences), -1, -1)
            x = torch.cat((repeated_prefix, self.decoder_embedding(sequences)), dim=1)
            positions = self._decoder_positions(
                0 if history is None else history.shape[1], sequences.shape[1], x.device
            )
            for block in self.decoder:
                x = block(x, causal=True, positions=positions)
            candidates = scores[:, None] + F.log_softmax(self.head(self.norm(x[:, -1])).float(), -1)
            scores, indices = candidates.flatten().topk(width)
            parents, tokens = indices // self.config.vocab_size, indices % self.config.vocab_size
            sequences = torch.cat((sequences[parents], tokens[:, None]), dim=1)
        return sequences[:1, 1:]

    @torch.no_grad()
    def reconstruct_sequence(
        self, codes: torch.Tensor, history: torch.Tensor | None = None
    ) -> torch.Tensor:
        if codes.shape[1] % CODES_PER_BLOCK:
            raise ValueError(f"code sequence length must be divisible by {CODES_PER_BLOCK}")
        decoded = []
        for start in range(0, codes.shape[1], CODES_PER_BLOCK):
            block = self.reconstruct(codes[:, start : start + CODES_PER_BLOCK], history)
            decoded.append(block)
            if self.config.decoder_context:
                history = block[:, -self.config.decoder_context :]
        return torch.cat(decoded, dim=1)

    @torch.no_grad()
    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.quantize(self.encode_vectors(tokens))[1]

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, std=0.02)


def lm_config(kind: str, vocab_size: int) -> ModelConfig:
    if kind == "baseline":
        return ModelConfig(vocab_size, 256, 10, 8, 256)
    if kind == "refined":
        return ModelConfig(vocab_size, 96, 10, 6, 288)
    if kind == "latent40":
        return ModelConfig(CODEBOOK_SIZE, 64, 16, 8, 448)
    return ModelConfig(CODEBOOK_SIZE, 64, 10, 6, 288)


def device_from(value: str) -> str:
    return "cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value)


def amp_context(device: str):
    enabled = device == "cuda" and torch.cuda.is_bf16_supported()
    return torch.autocast("cuda", dtype=torch.bfloat16) if enabled else nullcontext()


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def load_tokens(path: Path) -> np.memmap:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.memmap(path, dtype=np.uint16, mode="r")


def token_batch(data: np.memmap, length: int, batch_size: int, device: str) -> torch.Tensor:
    if len(data) < length:
        raise ValueError(f"{len(data)} tokens is too short for a {length}-token batch")
    starts = np.random.randint(0, len(data) - length + 1, batch_size)
    offsets = starts[:, None] + np.arange(length)[None]
    return torch.from_numpy(data[offsets].astype(np.int64)).to(device)


def lm_batch(data: np.memmap, length: int, batch_size: int, device: str):
    sequence = token_batch(data, length + 1, batch_size, device)
    return sequence[:, :-1].contiguous(), sequence[:, 1:].contiguous()


def context_batch(
    raw: np.memmap, latent: np.memmap, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    blocks = min(len(raw) // RAW_BLOCK, len(latent) // CODES_PER_BLOCK)
    if blocks < 2:
        raise ValueError("context training needs at least two aligned blocks")
    indices = np.random.randint(1, blocks, batch_size)
    raw_offsets = np.arange(RAW_BLOCK)[None]
    code_offsets = np.arange(CODES_PER_BLOCK)[None]
    history = raw[(indices[:, None] - 1) * RAW_BLOCK + raw_offsets].astype(np.int64)
    targets = raw[indices[:, None] * RAW_BLOCK + raw_offsets].astype(np.int64)
    codes = latent[indices[:, None] * CODES_PER_BLOCK + code_offsets].astype(np.int64)
    return tuple(torch.from_numpy(array).to(device) for array in (history, targets, codes))


def grouped_token_losses(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
    return losses.view(losses.shape[0], CODES_PER_BLOCK, COMPRESSION).mean(-1)


@torch.no_grad()
def calibrate_refinement_threshold(
    model: VQAutoencoder,
    raw: np.memmap,
    latent: np.memmap,
    rate: float,
    batches: int,
    batch_size: int,
    device: str,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(batches):
        history, targets, codes = context_batch(raw, latent, batch_size, device)
        with amp_context(device):
            logits = model.decoder_logits(codes, history, targets)
        losses.append(grouped_token_losses(logits.float(), targets).flatten().cpu())
    model.train(was_training)
    return float(torch.quantile(torch.cat(losses), 1 - rate))


def load_tokenizer(data_dir: Path):
    from tokenizers import Tokenizer

    path = data_dir / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run prepare first")
    return Tokenizer.from_file(str(path))


def semantic_ir_ids(vocab_size: int) -> dict:
    return {
        "pad": vocab_size,
        "event": vocab_size + 1,
        "define": vocab_size + 2,
        "reference": vocab_size + 10,
        "separator": vocab_size + 18,
        "end": vocab_size + 19,
        "vocab_size": vocab_size + 20,
    }


def extract_semantic_ir(text: str, tokenizer, max_tokens: int) -> list[int]:
    ids = semantic_ir_ids(tokenizer.get_vocab_size())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentence_words = [re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", sentence) for sentence in sentences]
    flat = [word for words in sentence_words for word in words]
    counts = Counter(flat)
    named = {
        words[index + 1]
        for words in sentence_words
        for index, word in enumerate(words[:-1])
        if word.lower() == "named" and words[index + 1][:1].isupper()
    }
    entities = named | {
        word
        for word, count in counts.items()
        if count >= 2 and word[:1].isupper() and word not in SEMANTIC_SENTENCE_STARTERS
    }
    entity_ids, output = {}, []
    for words in sentence_words:
        if not words:
            continue
        output.append(ids["event"])
        for word in words:
            if word in entities and len(entity_ids) < 8:
                if word not in entity_ids:
                    entity_ids[word] = len(entity_ids)
                    output.append(ids["define"] + entity_ids[word])
                    output.extend(tokenizer.encode(" " + word).ids)
                else:
                    output.append(ids["reference"] + entity_ids[word])
            elif word.lower() not in SEMANTIC_STOPWORDS or word.lower() in {"no", "not", "never"} or word.isdigit():
                output.extend(tokenizer.encode(" " + word).ids)
            if len(output) >= max_tokens:
                return output[:max_tokens]
    return output[:max_tokens]


def prepare_semantic_ir(args) -> None:
    tokenizer = load_tokenizer(Path(args.data_dir))
    ids = semantic_ir_ids(tokenizer.get_vocab_size())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "semantic-ir-v1",
        "raw_block": RAW_BLOCK,
        "max_ir_tokens": args.max_ir_tokens,
        "ids": ids,
    }
    for split, token_limit in (("train", args.train_tokens), ("val", args.val_tokens)):
        raw = load_tokens(Path(args.data_dir) / f"{split}.bin")
        blocks = min(len(raw), token_limit) // RAW_BLOCK
        fixed = np.full((blocks, args.max_ir_tokens), ids["pad"], dtype=np.uint16)
        lengths = np.zeros(blocks, dtype=np.uint16)
        stream = []
        started = time.perf_counter()
        for block in range(blocks):
            tokens = np.asarray(raw[block * RAW_BLOCK : (block + 1) * RAW_BLOCK]).astype(np.int64)
            text = tokenizer.decode(tokens.tolist(), skip_special_tokens=True)
            ir = extract_semantic_ir(text, tokenizer, args.max_ir_tokens)
            fixed[block, : len(ir)] = ir
            lengths[block] = len(ir)
            stream.extend(ir)
            stream.append(ids["end"])
        fixed.tofile(out / f"{split}.bin")
        lengths.tofile(out / f"{split}.lengths.bin")
        np.asarray(stream, dtype=np.uint16).tofile(out / f"{split}.stream.bin")
        mean_length = float(lengths.mean())
        metadata[split] = {
            "blocks": blocks,
            "raw_tokens": blocks * RAW_BLOCK,
            "ir_tokens": int(lengths.sum()),
            "stream_tokens": len(stream),
            "mean_ir_tokens": mean_length,
            "sequence_compression": blocks * RAW_BLOCK / len(stream),
            "truncated_blocks": int((lengths == args.max_ir_tokens).sum()),
            "seconds": time.perf_counter() - started,
        }
    (out / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def semantic_realizer_batch(
    raw: np.memmap,
    ir: np.memmap,
    blocks: int,
    max_ir_tokens: int,
    separator: int,
    batch_size: int,
    device: str,
):
    indices = np.random.randint(0, blocks, size=batch_size)
    offsets = np.arange(RAW_BLOCK)
    ir_offsets = np.arange(max_ir_tokens)
    prompts = torch.from_numpy(
        ir[indices[:, None] * max_ir_tokens + ir_offsets].astype(np.int64)
    ).to(device)
    targets = torch.from_numpy(
        raw[indices[:, None] * RAW_BLOCK + offsets].astype(np.int64)
    ).to(device)
    separator_column = torch.full(
        (batch_size, 1), separator, dtype=torch.long, device=device
    )
    inputs = torch.cat((prompts, separator_column, targets[:, :-1]), dim=1)
    return inputs, targets


def semantic_realizer_config(vocab_size: int, max_ir_tokens: int) -> ModelConfig:
    return ModelConfig(vocab_size, max_ir_tokens + RAW_BLOCK, 10, 8, 256)


@torch.no_grad()
def evaluate_semantic_realizer(
    model: Transformer,
    raw: np.memmap,
    ir: np.memmap,
    blocks: int,
    max_ir_tokens: int,
    separator: int,
    batches: int,
    batch_size: int,
    device: str,
) -> dict:
    was_training = model.training
    model.eval()
    loss = accuracy = 0.0
    for _ in range(batches):
        inputs, targets = semantic_realizer_batch(
            raw, ir, blocks, max_ir_tokens, separator, batch_size, device
        )
        with amp_context(device):
            logits, _ = model(inputs)
            logits = logits[:, max_ir_tokens:]
            batch_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss += float(batch_loss)
        accuracy += float((logits.argmax(-1) == targets).float().mean())
    model.train(was_training)
    return {
        "teacher_loss": loss / batches,
        "teacher_perplexity": math.exp(min(20, loss / batches)),
        "teacher_token_accuracy": accuracy / batches,
    }


def semantic_realizer_payload(model, optimizer, step, best, elapsed, metrics, metadata) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "kind": "semantic-ir-realizer-v1",
        "step": step,
        "best_val_loss": best,
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "model_config": asdict(model.config),
        "semantic_meta": metadata,
    }


def train_semantic_realizer(args) -> None:
    device = device_from(args.device)
    data_dir, ir_dir, out = Path(args.data_dir), Path(args.ir_dir), Path(args.out)
    metadata = json.loads((ir_dir / "meta.json").read_text(encoding="utf-8"))
    max_ir_tokens = metadata["max_ir_tokens"]
    config = semantic_realizer_config(metadata["ids"]["vocab_size"], max_ir_tokens)
    model = Transformer(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    train_raw, val_raw = load_tokens(data_dir / "train.bin"), load_tokens(data_dir / "val.bin")
    train_ir, val_ir = load_tokens(ir_dir / "train.bin"), load_tokens(ir_dir / "val.bin")
    train_blocks = min(metadata["train"]["blocks"], len(train_raw) // RAW_BLOCK)
    val_blocks = min(metadata["val"]["blocks"], len(val_raw) // RAW_BLOCK)
    latest = out / "latest.pt"
    step, best, elapsed_before = 0, float("inf"), 0.0
    if latest.exists():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if checkpoint["model_config"] != asdict(config):
            raise ValueError(f"{latest} is incompatible with this run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, best = checkpoint["step"], checkpoint["best_val_loss"]
        elapsed_before = checkpoint.get("elapsed_seconds", 0.0)
        print(f"resuming semantic realizer at step {step:,}")
    print(
        f"semantic realizer: {model.num_parameters():,} parameters, {args.steps:,} steps on {device}",
        flush=True,
    )
    started = time.perf_counter()
    metrics = {}
    model.train()
    while step < args.steps:
        inputs, targets = semantic_realizer_batch(
            train_raw, train_ir, train_blocks, max_ir_tokens,
            metadata["ids"]["separator"], args.batch_size, device,
        )
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device):
            logits, _ = model(inputs)
            loss = F.cross_entropy(logits[:, max_ir_tokens:].flatten(0, 1), targets.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = learning_rate(step, args.steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        step += 1
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_semantic_realizer(
                model, val_raw, val_ir, val_blocks, max_ir_tokens,
                metadata["ids"]["separator"], args.eval_batches, args.batch_size, device,
            )
            row = {"step": step, "train_loss": float(loss.detach()), "lr": lr, **metrics}
            append_jsonl(out / "metrics.jsonl", row)
            print(json.dumps(row), flush=True)
            elapsed = elapsed_before + time.perf_counter() - started
            if metrics["teacher_loss"] < best:
                best = metrics["teacher_loss"]
                atomic_save(
                    semantic_realizer_payload(model, optimizer, step, best, elapsed, metrics, metadata),
                    out / "best.pt",
                )
        if step % args.save_every == 0 or step == args.steps:
            elapsed = elapsed_before + time.perf_counter() - started
            atomic_save(
                semantic_realizer_payload(model, optimizer, step, best, elapsed, metrics, metadata), latest
            )


@torch.no_grad()
def reconstruct_semantic_realizer(
    model: Transformer, prompts: torch.Tensor, separator: int, bpe_vocab_size: int
) -> torch.Tensor:
    tokens = torch.cat(
        (
            prompts,
            torch.full((len(prompts), 1), separator, dtype=torch.long, device=prompts.device),
        ),
        dim=1,
    )
    generated = []
    for _ in range(RAW_BLOCK):
        logits, _ = model(tokens)
        next_token = logits[:, -1, :bpe_vocab_size].argmax(-1, keepdim=True)
        generated.append(next_token)
        tokens = torch.cat((tokens, next_token), dim=1)
    return torch.cat(generated, dim=1)


def semantic_lexical_tokens(text: str, tokenizer, max_ir_tokens: int) -> list[int]:
    vocab_size = tokenizer.get_vocab_size()
    return [token for token in extract_semantic_ir(text, tokenizer, max_ir_tokens) if token < vocab_size]


@torch.no_grad()
def report_semantic_realizer(args) -> None:
    device = device_from(args.device)
    data_dir, ir_dir, out = Path(args.data_dir), Path(args.ir_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(data_dir)
    metadata = json.loads((ir_dir / "meta.json").read_text(encoding="utf-8"))
    model, checkpoint = load_lm(Path(args.checkpoint), device)
    raw, ir = load_tokens(data_dir / "val.bin"), load_tokens(ir_dir / "val.bin")
    max_ir_tokens = metadata["max_ir_tokens"]
    blocks = min(metadata["val"]["blocks"], len(raw) // RAW_BLOCK)
    metrics = evaluate_semantic_realizer(
        model, raw, ir, blocks, max_ir_tokens, metadata["ids"]["separator"],
        args.eval_batches, args.batch_size, device,
    )
    random.seed(args.seed)
    indices = random.sample(range(blocks), min(args.samples, blocks))
    offsets, ir_offsets = np.arange(RAW_BLOCK), np.arange(max_ir_tokens)
    targets = torch.from_numpy(raw[np.asarray(indices)[:, None] * RAW_BLOCK + offsets].astype(np.int64)).to(device)
    prompts = torch.from_numpy(ir[np.asarray(indices)[:, None] * max_ir_tokens + ir_offsets].astype(np.int64)).to(device)
    generated = reconstruct_semantic_realizer(
        model, prompts, metadata["ids"]["separator"], tokenizer.get_vocab_size()
    )
    correct = total = lexical_correct = lexical_total = 0
    samples = []
    for target, prediction in zip(targets.cpu().tolist(), generated.cpu().tolist()):
        source_text = tokenizer.decode(target, skip_special_tokens=True)
        predicted_text = tokenizer.decode(prediction, skip_special_tokens=True)
        correct += sum(a == b for a, b in zip(target, prediction))
        total += RAW_BLOCK
        expected = Counter(semantic_lexical_tokens(source_text, tokenizer, max_ir_tokens))
        actual = Counter(semantic_lexical_tokens(predicted_text, tokenizer, max_ir_tokens))
        lexical_correct += sum((expected & actual).values())
        lexical_total += sum(expected.values())
        samples.append({"source": source_text, "reconstruction": predicted_text})
    metrics.update(
        {
            "greedy_token_accuracy": correct / total,
            "semantic_lexical_recall": lexical_correct / max(1, lexical_total),
            "sequence_compression": metadata["val"]["sequence_compression"],
            "realizer_parameters": model.num_parameters(),
            "training_seconds": checkpoint.get("elapsed_seconds"),
            "samples_evaluated": len(samples),
        }
    )
    (out / "report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    sample_text = "\n\n".join(
        f"## Sample {index + 1}\n\nSource: {sample['source']}\n\nReconstruction: {sample['reconstruction']}"
        for index, sample in enumerate(samples)
    )
    (out / "samples.md").write_text(sample_text + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def stream_text(dataset: str, split: str, limit: int | None = None):
    from datasets import load_dataset

    for index, row in enumerate(load_dataset(dataset, split=split, streaming=True)):
        if limit is not None and index >= limit:
            break
        yield row["text"]


def write_token_split(
    dataset: str, split: str, path: Path, tokenizer, token_limit: int, label: str
) -> dict:
    eos = tokenizer.token_to_id("<|endoftext|>")
    written = documents = 0
    started = time.perf_counter()
    with path.open("wb") as handle:
        for text in stream_text(dataset, split):
            ids = tokenizer.encode(text).ids + [eos]
            ids = ids[: token_limit - written]
            np.asarray(ids, dtype=np.uint16).tofile(handle)
            written += len(ids)
            documents += 1
            if documents % 1000 == 0:
                print(f"{label}: {written:,}/{token_limit:,} tokens ({documents:,} stories)", flush=True)
            if written >= token_limit:
                break
    return {
        "tokens": written,
        "documents": documents,
        "seconds": time.perf_counter() - started,
        "path": str(path),
    }


def prepare(args) -> None:
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    out = Path(args.data_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer_path = out / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"reusing {tokenizer_path}")
    else:
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=args.vocab_size,
            special_tokens=["<unk>", "<|endoftext|>"],
            show_progress=True,
        )
        tokenizer.train_from_iterator(
            stream_text(args.dataset, "train", args.tokenizer_docs),
            trainer=trainer,
            length=args.tokenizer_docs,
        )
        tokenizer.save(str(tokenizer_path))

    if tokenizer.get_vocab_size() > np.iinfo(np.uint16).max:
        raise ValueError("vocabulary does not fit uint16")
    started = time.perf_counter()
    metadata = {
        "dataset": args.dataset,
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_docs": args.tokenizer_docs,
        "train": write_token_split(
            args.dataset, "train", out / "train.bin", tokenizer, args.train_tokens, "train"
        ),
        "val": write_token_split(
            args.dataset, "validation", out / "val.bin", tokenizer, args.val_tokens, "val"
        ),
    }
    metadata["total_seconds"] = time.perf_counter() - started
    (out / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def learning_rate(step: int, total: int, peak: float, warmup: int = 200) -> float:
    warmup = min(warmup, max(1, total // 20))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate_ae(
    model: VQAutoencoder,
    data: np.memmap,
    batches: int,
    batch_size: int,
    device: str,
    decode: bool = False,
):
    was_training = model.training
    model.eval()
    reconstruction = teacher_accuracy = decoded_accuracy = total = 0.0
    usage = torch.zeros(model.config.codebook_size, device=device)
    for _ in range(batches):
        tokens = token_batch(data, RAW_BLOCK, batch_size, device)
        with amp_context(device):
            logits, codes, reconstruction_loss, _ = model(tokens)
            if decode:
                decoded_accuracy += float((model.reconstruct(codes) == tokens).float().mean())
        reconstruction += float(reconstruction_loss)
        teacher_accuracy += float((logits.argmax(-1) == tokens).float().mean())
        usage += torch.bincount(codes.flatten(), minlength=model.config.codebook_size)
        total += 1
    probabilities = usage / usage.sum()
    active = probabilities > 0
    code_perplexity = torch.exp(-(probabilities[active] * probabilities[active].log()).sum())
    model.train(was_training)
    mean_loss = reconstruction / total
    metrics = {
        "reconstruction_loss": mean_loss,
        "reconstruction_perplexity": math.exp(min(20, mean_loss)),
        "token_accuracy": (decoded_accuracy if decode else teacher_accuracy) / total,
        "active_codes": int(active.sum()),
        "code_perplexity": float(code_perplexity),
    }
    if decode:
        metrics["teacher_forced_token_accuracy"] = teacher_accuracy / total
    return metrics


@torch.no_grad()
def evaluate_decoder(
    model: VQAutoencoder,
    raw: np.memmap,
    latent: np.memmap,
    batches: int,
    batch_size: int,
    device: str,
    decode: bool = False,
) -> dict:
    was_training = model.training
    model.eval()
    loss_total = teacher_accuracy = decoded_accuracy = 0.0
    usage = torch.zeros(model.config.codebook_size, device=device)
    for _ in range(batches):
        history, targets, codes = context_batch(raw, latent, batch_size, device)
        with amp_context(device):
            logits = model.decoder_logits(codes, history, targets)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            if decode:
                decoded_accuracy += float((model.reconstruct(codes, history) == targets).float().mean())
        loss_total += float(loss)
        teacher_accuracy += float((logits.argmax(-1) == targets).float().mean())
        usage += torch.bincount(codes.flatten(), minlength=model.config.codebook_size)
    probabilities = usage / usage.sum()
    active = probabilities > 0
    mean_loss = loss_total / batches
    metrics = {
        "reconstruction_loss": mean_loss,
        "reconstruction_perplexity": math.exp(min(20, mean_loss)),
        "token_accuracy": (decoded_accuracy if decode else teacher_accuracy) / batches,
        "active_codes": int(active.sum()),
        "code_perplexity": float(
            torch.exp(-(probabilities[active] * probabilities[active].log()).sum())
        ),
    }
    if decode:
        metrics["teacher_forced_token_accuracy"] = teacher_accuracy / batches
    model.train(was_training)
    return metrics


@torch.no_grad()
def evaluate_joint_codec(
    model: VQAutoencoder,
    raw: np.memmap,
    latent: np.memmap,
    batches: int,
    batch_size: int,
    device: str,
) -> dict:
    was_training = model.training
    model.eval()
    loss_total = accuracy = 0.0
    usage = torch.zeros(model.config.codebook_size, device=device)
    for _ in range(batches):
        history, targets, _ = context_batch(raw, latent, batch_size, device)
        with amp_context(device):
            quantized, codes, _ = model.quantize(model.encode_vectors(targets))
            logits = model._decode_teacher(quantized, targets, history, 0.0)
            loss_total += float(F.cross_entropy(logits.flatten(0, 1), targets.flatten()))
        accuracy += float((logits.argmax(-1) == targets).float().mean())
        usage += torch.bincount(codes.flatten(), minlength=model.config.codebook_size)
    probabilities = usage / usage.sum()
    active = probabilities > 0
    mean_loss = loss_total / batches
    model.train(was_training)
    return {
        "reconstruction_loss": mean_loss,
        "reconstruction_perplexity": math.exp(min(20, mean_loss)),
        "token_accuracy": accuracy / batches,
        "active_codes": int(active.sum()),
        "code_perplexity": float(torch.exp(-(probabilities[active] * probabilities[active].log()).sum())),
    }


@torch.no_grad()
def evaluate_refiner(
    model: VQAutoencoder,
    raw: np.memmap,
    latent: np.memmap,
    batches: int,
    batch_size: int,
    device: str,
    decode: bool = False,
) -> dict:
    was_training = model.training
    model.eval()
    loss_total = teacher_accuracy = decoded_accuracy = refinements = 0.0
    usage = torch.zeros(model.config.codebook_size, device=device)
    for _ in range(batches):
        history, targets, codes = context_batch(raw, latent, batch_size, device)
        with amp_context(device):
            base_logits = model.decoder_logits(codes, history, targets)
            mask = grouped_token_losses(base_logits.float(), targets) > model.config.refinement_threshold
            _, refinement_codes, _ = model.encode_refinement_vectors(targets)
            logits = model.decoder_logits(codes, history, targets, 0.0, refinement_codes, mask)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            if decode:
                decoded = model.reconstruct(codes, history, refinement_codes, mask)
                decoded_accuracy += float((decoded == targets).float().mean())
        loss_total += float(loss)
        teacher_accuracy += float((logits.argmax(-1) == targets).float().mean())
        refinements += float(mask.float().sum(1).mean())
        usage += torch.bincount(refinement_codes[mask], minlength=model.config.codebook_size)
    probabilities = usage / usage.sum().clamp_min(1)
    active = probabilities > 0
    mean_loss = loss_total / batches
    metrics = {
        "reconstruction_loss": mean_loss,
        "reconstruction_perplexity": math.exp(min(20, mean_loss)),
        "token_accuracy": (decoded_accuracy if decode else teacher_accuracy) / batches,
        "teacher_forced_token_accuracy": teacher_accuracy / batches,
        "active_codes": int(active.sum()),
        "code_perplexity": float(
            torch.exp(-(probabilities[active] * probabilities[active].log()).sum())
        ),
        "average_refinements": refinements / batches,
    }
    model.train(was_training)
    return metrics


def ae_payload(model, optimizer, step: int, best: float, elapsed: float, metrics: dict) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_reconstruction_loss": best,
        "elapsed_seconds": elapsed,
        "model_config": asdict(model.config),
        "metrics": metrics,
    }


def train_ae(args) -> None:
    device = device_from(args.device)
    data_dir, out = Path(args.data_dir), Path(args.out)
    metadata = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    train, val = load_tokens(data_dir / "train.bin"), load_tokens(data_dir / "val.bin")
    target_steps = args.steps or math.ceil(len(train) / (args.batch_size * RAW_BLOCK))
    model = VQAutoencoder(AEConfig(vocab_size=metadata["vocab_size"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    latest = out / "latest.pt"
    step, best, elapsed_before = 0, float("inf"), 0.0
    if latest.exists():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        best = checkpoint["best_reconstruction_loss"]
        elapsed_before = checkpoint.get("elapsed_seconds", 0.0)
        print(f"resuming autoencoder at step {step:,}")
    print(f"autoencoder: {model.num_parameters():,} parameters on {device}")
    if step >= target_steps:
        print(f"already reached {target_steps:,} steps")
        return

    started = time.perf_counter()
    metrics = {}
    model.train()
    while step < target_steps:
        optimizer.zero_grad(set_to_none=True)
        tokens = token_batch(train, RAW_BLOCK, args.batch_size, device)
        with amp_context(device):
            _, _, reconstruction_loss, vq_loss = model(tokens)
            loss = reconstruction_loss + vq_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = learning_rate(step, target_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        step += 1

        should_eval = step == 1 or step % args.eval_every == 0 or step == target_steps
        if should_eval:
            metrics = evaluate_ae(model, val, args.eval_batches, args.batch_size, device)
            row = {
                "step": step,
                "train_loss": float(loss.detach()),
                "vq_loss": float(vq_loss.detach()),
                "lr": lr,
                **metrics,
            }
            append_jsonl(out / "metrics.jsonl", row)
            print(json.dumps(row), flush=True)
            elapsed = elapsed_before + time.perf_counter() - started
            if metrics["reconstruction_loss"] < best:
                best = metrics["reconstruction_loss"]
                atomic_save(ae_payload(model, optimizer, step, best, elapsed, metrics), out / "best.pt")

        if step % args.save_every == 0 or step == target_steps:
            elapsed = elapsed_before + time.perf_counter() - started
            atomic_save(ae_payload(model, optimizer, step, best, elapsed, metrics), latest)


def train_decoder(args) -> None:
    device = device_from(args.device)
    raw_dir, latent_dir, out = Path(args.data_dir), Path(args.latent_dir), Path(args.out)
    latest = out / "latest.pt"
    source = latest if latest.exists() else Path(args.autoencoder)
    model, checkpoint = load_ae(source, device)
    model.config.decoder_context = args.context
    model.config.token_dropout = args.end_dropout
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.decoder_embedding, model.decoder_code_projection, model.decoder, model.norm, model.head):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    train_raw, val_raw = load_tokens(raw_dir / "train.bin"), load_tokens(raw_dir / "val.bin")
    train_latent, val_latent = load_tokens(latent_dir / "train.bin"), load_tokens(latent_dir / "val.bin")
    blocks = min(len(train_raw) // RAW_BLOCK, len(train_latent) // CODES_PER_BLOCK) - 1
    target_steps = args.steps or math.ceil(blocks / args.batch_size)
    if latest.exists():
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        best = checkpoint["best_reconstruction_loss"]
        elapsed_before = checkpoint.get("elapsed_seconds", 0.0)
        print(f"resuming contextual decoder at step {step:,}")
    else:
        step, best = 0, float("inf")
        elapsed_before = accounting_checkpoint(Path(args.autoencoder), device).get("elapsed_seconds", 0.0)
    print(
        f"contextual decoder: {sum(p.numel() for p in parameters):,} trainable parameters, "
        f"{target_steps:,} steps on {device}",
        flush=True,
    )
    if step >= target_steps:
        print(f"already reached {target_steps:,} steps")
        return

    started = time.perf_counter()
    metrics = checkpoint.get("metrics", {}) if latest.exists() else {}
    model.train()
    while step < target_steps:
        progress = step / max(1, target_steps - 1)
        dropout = args.start_dropout + (args.end_dropout - args.start_dropout) * progress
        history, targets, codes = context_batch(train_raw, train_latent, args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device):
            logits = model.decoder_logits(codes, history, targets, dropout)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        lr = learning_rate(step, target_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        step += 1

        should_eval = step == 1 or step % args.eval_every == 0 or step == target_steps
        if should_eval:
            metrics = evaluate_decoder(
                model, val_raw, val_latent, args.eval_batches, args.batch_size, device
            )
            row = {"step": step, "train_loss": float(loss.detach()), "dropout": dropout, "lr": lr, **metrics}
            append_jsonl(out / "metrics.jsonl", row)
            print(json.dumps(row), flush=True)
            elapsed = elapsed_before + time.perf_counter() - started
            payload = ae_payload(model, optimizer, step, min(best, metrics["reconstruction_loss"]), elapsed, metrics)
            payload["base_checkpoint"] = str(args.autoencoder)
            if metrics["reconstruction_loss"] < best:
                best = metrics["reconstruction_loss"]
                atomic_save(payload, out / "best.pt")

        if step % args.save_every == 0 or step == target_steps:
            elapsed = elapsed_before + time.perf_counter() - started
            payload = ae_payload(model, optimizer, step, best, elapsed, metrics)
            payload["base_checkpoint"] = str(args.autoencoder)
            atomic_save(payload, latest)


def train_joint_codec(args) -> None:
    device = device_from(args.device)
    raw_dir, latent_dir, out = Path(args.data_dir), Path(args.latent_dir), Path(args.out)
    latest = out / "latest.pt"
    source = latest if latest.exists() else Path(args.autoencoder)
    priority = getattr(args, "priority", False)
    model, checkpoint = load_ae(source, device, enable_priority=priority)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    train_raw, val_raw = load_tokens(raw_dir / "train.bin"), load_tokens(raw_dir / "val.bin")
    train_latent, val_latent = load_tokens(latent_dir / "train.bin"), load_tokens(latent_dir / "val.bin")
    frequencies = np.bincount(train_raw, minlength=model.config.vocab_size).astype(np.float32)
    if priority:
        rare_weights = np.sqrt(args.rare_cutoff / np.maximum(frequencies, 1)).clip(1, args.priority_weight)
    else:
        rare_weights = np.ones_like(frequencies)
        rare = frequencies < args.rare_cutoff
        rare_weights[rare] += (args.rare_weight - 1) * (1 - frequencies[rare] / args.rare_cutoff)
    rare_weights = torch.from_numpy(rare_weights).to(device)
    blocks = min(len(train_raw) // RAW_BLOCK, len(train_latent) // CODES_PER_BLOCK) - 1
    one_pass = math.ceil(blocks / args.batch_size)
    target_steps = args.steps or args.epochs * one_pass
    if latest.exists():
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, best = checkpoint["step"], checkpoint["best_reconstruction_loss"]
        elapsed_before = checkpoint.get("elapsed_seconds", 0.0)
        print(f"resuming joint codec at step {step:,}")
    else:
        step, best = 0, float("inf")
        elapsed_before = accounting_checkpoint(Path(args.autoencoder), device).get("elapsed_seconds", 0.0)
    print(
        f"joint codec: {sum(p.numel() for p in parameters):,} trainable parameters, "
        f"{target_steps:,} steps ({target_steps / one_pass:.1f} passes) on {device}", flush=True
    )
    if step >= target_steps:
        print(f"already reached {target_steps:,} steps")
        return
    started = time.perf_counter()
    metrics = checkpoint.get("metrics", {}) if latest.exists() else {}
    model.train()
    while step < target_steps:
        history, targets, _ = context_batch(train_raw, train_latent, args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device):
            quantized, _, quantizer_loss = model.quantize(model.encode_vectors(targets))
            with torch.no_grad():
                if args.self_condition:
                    proposed = model._decode_teacher(quantized.detach(), targets, history, 0.0).argmax(-1)
                    replace = torch.rand_like(targets.float()) < args.self_condition
                    decoder_tokens = torch.where(replace, proposed, targets)
                else:
                    decoder_tokens = targets
            logits = model._decode_teacher(quantized, decoder_tokens, history, args.dropout)
            token_loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
            weights = rare_weights[targets]
            auxiliary_loss = token_loss.new_zeros(())
            hard = torch.zeros_like(targets, dtype=torch.bool)
            if priority:
                hard = token_loss.detach() >= torch.quantile(token_loss.detach(), 1 - args.hard_rate)
                weights = (weights * torch.where(hard, args.hard_weight, 1.0)).clamp_max(
                    args.priority_weight
                )
                auxiliary_token_loss = F.cross_entropy(
                    model.priority_logits(quantized).transpose(1, 2), targets, reduction="none"
                )
                priority_mask = (rare_weights[targets] > 1) | hard
                auxiliary_weights = weights * priority_mask
                auxiliary_loss = (auxiliary_token_loss * auxiliary_weights).sum() / auxiliary_weights.sum()
            reconstruction_loss = (token_loss * weights).sum() / weights.sum()
            loss = reconstruction_loss + quantizer_loss + args.aux_weight * auxiliary_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        lr = learning_rate(step, target_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        step += 1
        if step == 1 or step % args.eval_every == 0 or step == target_steps:
            metrics = evaluate_joint_codec(model, val_raw, val_latent, args.eval_batches, args.batch_size, device)
            row = {
                "step": step,
                "train_loss": float(loss.detach()),
                "reconstruction_loss_weighted": float(reconstruction_loss.detach()),
                "quantizer_loss": float(quantizer_loss.detach()),
                "auxiliary_loss": float(auxiliary_loss.detach()),
                "hard_fraction": float(hard.float().mean()),
                "lr": lr,
                **metrics,
            }
            append_jsonl(out / "metrics.jsonl", row)
            print(json.dumps(row), flush=True)
            elapsed = elapsed_before + time.perf_counter() - started
            payload = ae_payload(model, optimizer, step, min(best, metrics["reconstruction_loss"]), elapsed, metrics)
            payload["base_checkpoint"] = str(args.autoencoder)
            if metrics["reconstruction_loss"] < best:
                best = metrics["reconstruction_loss"]
                atomic_save(payload, out / "best.pt")
        if step % args.save_every == 0 or step == target_steps:
            elapsed = elapsed_before + time.perf_counter() - started
            payload = ae_payload(model, optimizer, step, best, elapsed, metrics)
            payload["base_checkpoint"] = str(args.autoencoder)
            atomic_save(payload, latest)


@torch.no_grad()
def rare_diagnostic(args) -> None:
    device = device_from(args.device)
    model, _ = load_ae(Path(args.autoencoder), device)
    raw_dir, latent_dir = Path(args.data_dir), Path(args.latent_dir)
    train, raw, latent = (
        load_tokens(raw_dir / "train.bin"),
        load_tokens(raw_dir / "val.bin"),
        load_tokens(latent_dir / "val.bin"),
    )
    frequencies = torch.from_numpy(
        np.bincount(train, minlength=model.config.vocab_size).astype(np.int64)
    ).to(device)
    correct = {cutoff: 0 for cutoff in args.cutoffs}
    counts = {cutoff: 0 for cutoff in args.cutoffs}
    overall_correct = total = 0
    for first in range(1, args.blocks + 1, args.batch_size):
        count = min(args.batch_size, args.blocks - first + 1)
        indices = np.arange(first, first + count)
        offsets = np.arange(RAW_BLOCK)
        code_offsets = np.arange(CODES_PER_BLOCK)
        history = torch.from_numpy(raw[(indices[:, None] - 1) * RAW_BLOCK + offsets].astype(np.int64)).to(device)
        targets = torch.from_numpy(raw[indices[:, None] * RAW_BLOCK + offsets].astype(np.int64)).to(device)
        codes = torch.from_numpy(latent[indices[:, None] * CODES_PER_BLOCK + code_offsets].astype(np.int64)).to(device)
        matches = model.reconstruct(codes, history) == targets
        token_frequencies = frequencies[targets]
        overall_correct += int(matches.sum())
        total += targets.numel()
        for cutoff in args.cutoffs:
            mask = token_frequencies <= cutoff
            correct[cutoff] += int((matches & mask).sum())
            counts[cutoff] += int(mask.sum())
    result = {
        "blocks": args.blocks,
        "overall_accuracy": overall_correct / total,
        "rare_accuracy": {
            str(cutoff): correct[cutoff] / max(1, counts[cutoff]) for cutoff in args.cutoffs
        },
        "rare_tokens": counts,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


@torch.no_grad()
def near_lossless_diagnostic(args) -> None:
    if not args.targets or any(not 0 < target <= 1 for target in args.targets):
        raise ValueError("--targets must contain values in (0, 1]")
    device = device_from(args.device)
    model, _ = load_ae(Path(args.autoencoder), device)
    raw = load_tokens(Path(args.data_dir) / "val.bin")
    latent = load_tokens(Path(args.latent_dir) / "val.bin")
    wrong_counts = []
    for first in range(1, args.blocks + 1, args.batch_size):
        count = min(args.batch_size, args.blocks - first + 1)
        indices = np.arange(first, first + count)
        offsets, code_offsets = np.arange(RAW_BLOCK), np.arange(CODES_PER_BLOCK)
        history = torch.from_numpy(
            raw[(indices[:, None] - 1) * RAW_BLOCK + offsets].astype(np.int64)
        ).to(device)
        targets = torch.from_numpy(
            raw[indices[:, None] * RAW_BLOCK + offsets].astype(np.int64)
        ).to(device)
        codes = torch.from_numpy(
            latent[indices[:, None] * CODES_PER_BLOCK + code_offsets].astype(np.int64)
        ).to(device)
        wrong_counts.extend((model.reconstruct(codes, history) != targets).sum(1).cpu().tolist())

    wrong = np.asarray(wrong_counts)
    rows = []
    for target in args.targets:
        target_correct = math.ceil(target * RAW_BLOCK)
        corrections = np.maximum(0, target_correct - (RAW_BLOCK - wrong))
        achieved = np.minimum(RAW_BLOCK, RAW_BLOCK - wrong + corrections).mean() / RAW_BLOCK
        mean_corrections = float(corrections.mean())
        packed_bits = CODES_PER_BLOCK * 10 + 6 + 19 * mean_corrections
        transformer_tokens = CODES_PER_BLOCK + 2 * mean_corrections
        rows.append(
            {
                "target_accuracy": target,
                "achieved_accuracy": achieved,
                "corrections_per_block": mean_corrections,
                "packed_bit_compression_vs_13bit_bpe": RAW_BLOCK * 13 / packed_bits,
                "transformer_sequence_compression_vs_bpe": RAW_BLOCK / transformer_tokens,
            }
        )
    result = {
        "blocks": args.blocks,
        "base_accuracy": 1 - float(wrong.mean()) / RAW_BLOCK,
        "format": "16 ten-bit v3 codes + 6-bit count + N exact 19-bit (position, BPE token) residuals",
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


@torch.no_grad()
def reconstruct_with_confidence(
    model: VQAutoencoder, codes: torch.Tensor, history: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    generated, confidences = [], []
    for _ in range(RAW_BLOCK):
        prefix = (
            torch.cat(generated, dim=1)
            if generated
            else torch.empty((len(codes), 0), dtype=torch.long, device=codes.device)
        )
        probe = torch.cat(
            (prefix, torch.zeros((len(codes), 1), dtype=torch.long, device=codes.device)), dim=1
        )
        logits = model.decoder_logits(codes, history, probe, 0.0)[:, -1].float()
        confidence, token = F.log_softmax(logits, -1).max(-1, keepdim=True)
        generated.append(token)
        confidences.append(confidence)
    return torch.cat(generated, dim=1), torch.cat(confidences, dim=1)


@torch.no_grad()
def dual_codec_diagnostic(args) -> None:
    device = device_from(args.device)
    base, _ = load_ae(Path(args.base_autoencoder), device)
    priority, _ = load_ae(Path(args.priority_autoencoder), device)
    raw_dir = Path(args.data_dir)
    raw, train = load_tokens(raw_dir / "val.bin"), load_tokens(raw_dir / "train.bin")
    base_latent = load_tokens(Path(args.base_latent_dir) / "val.bin")
    priority_latent = load_tokens(Path(args.priority_latent_dir) / "val.bin")
    frequencies = torch.from_numpy(
        np.bincount(train, minlength=base.config.vocab_size).astype(np.int64)
    ).to(device)
    totals = Counter()
    for first in range(1, args.blocks + 1, args.batch_size):
        count = min(args.batch_size, args.blocks - first + 1)
        indices = np.arange(first, first + count)
        offsets, code_offsets = np.arange(RAW_BLOCK), np.arange(CODES_PER_BLOCK)
        history = torch.from_numpy(
            raw[(indices[:, None] - 1) * RAW_BLOCK + offsets].astype(np.int64)
        ).to(device)
        targets = torch.from_numpy(
            raw[indices[:, None] * RAW_BLOCK + offsets].astype(np.int64)
        ).to(device)
        base_codes = torch.from_numpy(
            base_latent[indices[:, None] * CODES_PER_BLOCK + code_offsets].astype(np.int64)
        ).to(device)
        priority_codes = torch.from_numpy(
            priority_latent[indices[:, None] * CODES_PER_BLOCK + code_offsets].astype(np.int64)
        ).to(device)
        with amp_context(device):
            base_tokens, base_confidence = reconstruct_with_confidence(base, base_codes, history)
            priority_tokens, priority_confidence = reconstruct_with_confidence(
                priority, priority_codes, history
            )
        base_correct, priority_correct = base_tokens == targets, priority_tokens == targets
        rare = frequencies[targets] <= args.rare_cutoff
        confidence_tokens = torch.where(
            priority_confidence > base_confidence, priority_tokens, base_tokens
        )
        rare_gate = (frequencies[priority_tokens] <= args.rare_cutoff) & (
            priority_confidence > base_confidence
        )
        rare_confidence_tokens = torch.where(rare_gate, priority_tokens, base_tokens)
        for name, matches in (
            ("base", base_correct),
            ("priority", priority_correct),
            ("oracle_union", base_correct | priority_correct),
            ("confidence_merge", confidence_tokens == targets),
            ("rare_confidence_merge", rare_confidence_tokens == targets),
        ):
            totals[f"{name}_correct"] += int(matches.sum())
            totals[f"{name}_rare_correct"] += int((matches & rare).sum())
        totals["agree"] += int((base_tokens == priority_tokens).sum())
        totals["both_correct"] += int((base_correct & priority_correct).sum())
        totals["tokens"] += targets.numel()
        totals["rare_tokens"] += int(rare.sum())

    def metrics(name: str) -> dict:
        return {
            "accuracy": totals[f"{name}_correct"] / totals["tokens"],
            "rare_accuracy": totals[f"{name}_rare_correct"] / max(1, totals["rare_tokens"]),
        }

    result = {
        "blocks": args.blocks,
        "compression_if_both_streams": 2.0,
        "agreement": totals["agree"] / totals["tokens"],
        "both_correct": totals["both_correct"] / totals["tokens"],
        **{
            name: metrics(name)
            for name in (
                "base", "priority", "oracle_union", "confidence_merge", "rare_confidence_merge"
            )
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


@torch.no_grad()
def correction_diagnostic(args) -> None:
    device = device_from(args.device)
    model, _ = load_ae(Path(args.autoencoder), device)
    raw_dir, latent_dir = Path(args.data_dir), Path(args.latent_dir)
    train, raw, latent = (
        load_tokens(raw_dir / "train.bin"),
        load_tokens(raw_dir / "val.bin"),
        load_tokens(latent_dir / "val.bin"),
    )
    frequencies = torch.from_numpy(
        np.bincount(train, minlength=model.config.vocab_size).astype(np.float32)
    ).to(device)
    rows = {slots: {"correct": 0, "rare_correct": 0, "selected_frequency": 0.0} for slots in args.slots}
    rare_total = total = 0
    for first in range(1, args.blocks + 1, args.batch_size):
        count = min(args.batch_size, args.blocks - first + 1)
        indices = np.arange(first, first + count)
        offsets, code_offsets = np.arange(RAW_BLOCK), np.arange(CODES_PER_BLOCK)
        history = torch.from_numpy(raw[(indices[:, None] - 1) * RAW_BLOCK + offsets].astype(np.int64)).to(device)
        targets = torch.from_numpy(raw[indices[:, None] * RAW_BLOCK + offsets].astype(np.int64)).to(device)
        codes = torch.from_numpy(latent[indices[:, None] * CODES_PER_BLOCK + code_offsets].astype(np.int64)).to(device)
        with amp_context(device):
            logits = model.decoder_logits(codes, history, targets, 0.0)
        losses = F.cross_entropy(logits.transpose(1, 2).float(), targets, reduction="none")
        token_frequencies = frequencies[targets]
        rarity = torch.log((len(train) + model.config.vocab_size) / (token_frequencies + 1))
        ranking = rarity + args.surprisal_weight * losses
        rare = token_frequencies <= args.rare_cutoff
        rare_total += int(rare.sum())
        total += targets.numel()
        for slots, row in rows.items():
            positions = ranking.topk(slots, dim=1).indices
            selected_tokens = targets.gather(1, positions)
            forced = torch.full_like(targets, -1)
            forced.scatter_(1, positions, selected_tokens)
            decoded = model.reconstruct(codes, history, forced_tokens=forced)
            matches = decoded == targets
            row["correct"] += int(matches.sum())
            row["rare_correct"] += int((matches & rare).sum())
            row["selected_frequency"] += float(token_frequencies.gather(1, positions).sum())
    result = []
    for slots, row in rows.items():
        result.append(
            {
                "correction_slots": slots,
                "compression": RAW_BLOCK / (CODES_PER_BLOCK + 2 * slots),
                "overall_accuracy": row["correct"] / total,
                "rare_accuracy": row["rare_correct"] / max(1, rare_total),
                "mean_selected_frequency": row["selected_frequency"] / (args.blocks * slots),
            }
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


@torch.no_grad()
def beam_diagnostic(args) -> None:
    device = device_from(args.device)
    model, _ = load_ae(Path(args.autoencoder), device)
    raw = load_tokens(Path(args.data_dir) / "val.bin")
    latent = load_tokens(Path(args.latent_dir) / "val.bin")
    greedy_correct = beam_correct = total = 0
    for block in range(1, args.blocks + 1):
        history = torch.from_numpy(np.asarray(raw[(block - 1) * RAW_BLOCK : block * RAW_BLOCK]).astype(np.int64))[None].to(device)
        targets = torch.from_numpy(np.asarray(raw[block * RAW_BLOCK : (block + 1) * RAW_BLOCK]).astype(np.int64))[None].to(device)
        codes = torch.from_numpy(np.asarray(latent[block * CODES_PER_BLOCK : (block + 1) * CODES_PER_BLOCK]).astype(np.int64))[None].to(device)
        greedy_correct += int((model.reconstruct(codes, history) == targets).sum())
        beam_correct += int((model.reconstruct_beam(codes, history, args.width) == targets).sum())
        total += RAW_BLOCK
    result = {"blocks": args.blocks, "width": args.width, "greedy_accuracy": greedy_correct / total, "beam_accuracy": beam_correct / total}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def train_refiner(args) -> None:
    device = device_from(args.device)
    raw_dir, latent_dir, out = Path(args.data_dir), Path(args.latent_dir), Path(args.out)
    latest = out / "latest.pt"
    source = latest if latest.exists() else Path(args.autoencoder)
    model, checkpoint = load_ae(source, device, enable_refinement=True)
    train_raw, val_raw = load_tokens(raw_dir / "train.bin"), load_tokens(raw_dir / "val.bin")
    train_latent, val_latent = load_tokens(latent_dir / "train.bin"), load_tokens(latent_dir / "val.bin")

    if not latest.exists():
        model.config.refinement_rate = args.rate
        model.config.refinement_threshold = args.threshold or calibrate_refinement_threshold(
            model,
            val_raw,
            val_latent,
            args.rate,
            args.calibration_batches,
            args.batch_size,
            device,
        )
        print(f"refinement threshold: {model.config.refinement_threshold:.4f}", flush=True)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (
        model.to_refinement,
        model.decoder_refinement_projection,
        model.decoder_embedding,
        model.decoder_code_projection,
        model.decoder,
        model.norm,
        model.head,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    blocks = min(len(train_raw) // RAW_BLOCK, len(train_latent) // CODES_PER_BLOCK) - 1
    target_steps = args.steps or math.ceil(blocks / args.batch_size)
    if latest.exists():
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        best = checkpoint["best_reconstruction_loss"]
        elapsed_before = checkpoint.get("elapsed_seconds", 0.0)
        print(f"resuming refiner at step {step:,}")
    else:
        step, best = 0, float("inf")
        elapsed_before = accounting_checkpoint(Path(args.autoencoder), device).get("elapsed_seconds", 0.0)
    print(
        f"refiner: {sum(p.numel() for p in parameters):,} trainable parameters, "
        f"{target_steps:,} steps on {device}",
        flush=True,
    )
    if step >= target_steps:
        print(f"already reached {target_steps:,} steps")
        return

    started = time.perf_counter()
    metrics = checkpoint.get("metrics", {}) if latest.exists() else {}
    model.train()
    while step < target_steps:
        history, targets, codes = context_batch(train_raw, train_latent, args.batch_size, device)
        with torch.no_grad(), amp_context(device):
            base_logits = model.decoder_logits(codes, history, targets)
            mask = grouped_token_losses(base_logits.float(), targets) > model.config.refinement_threshold
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device):
            refinement, refinement_codes, refinement_loss = model.encode_refinement_vectors(targets)
            logits = model._decode_teacher(
                model.codes_to_vectors(codes),
                targets,
                history,
                args.dropout,
                refinement,
                mask,
            )
            token_loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
            weights = 1 + args.mask_weight * mask.repeat_interleave(COMPRESSION, dim=1)
            reconstruction_loss = (token_loss * weights).sum() / weights.sum()
            loss = reconstruction_loss + refinement_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        lr = learning_rate(step, target_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        step += 1

        should_eval = step == 1 or step % args.eval_every == 0 or step == target_steps
        if should_eval:
            metrics = evaluate_refiner(
                model, val_raw, val_latent, args.eval_batches, args.batch_size, device
            )
            row = {
                "step": step,
                "train_loss": float(loss.detach()),
                "refinement_loss": float(refinement_loss.detach()),
                "masked_groups": float(mask.float().sum(1).mean()),
                "lr": lr,
                **metrics,
            }
            append_jsonl(out / "metrics.jsonl", row)
            print(json.dumps(row), flush=True)
            elapsed = elapsed_before + time.perf_counter() - started
            payload = ae_payload(model, optimizer, step, min(best, metrics["reconstruction_loss"]), elapsed, metrics)
            payload["base_checkpoint"] = str(args.autoencoder)
            if metrics["reconstruction_loss"] < best:
                best = metrics["reconstruction_loss"]
                atomic_save(payload, out / "best.pt")

        if step % args.save_every == 0 or step == target_steps:
            elapsed = elapsed_before + time.perf_counter() - started
            payload = ae_payload(model, optimizer, step, best, elapsed, metrics)
            payload["base_checkpoint"] = str(args.autoencoder)
            atomic_save(payload, latest)


def load_ae(
    path: Path, device: str, enable_refinement: bool = False, enable_priority: bool = False
):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = dict(checkpoint["model_config"])
    if "quantizer" not in config:
        config["quantizer"] = "vq"
    if enable_refinement:
        config["refinement"] = True
    if enable_priority:
        config["priority_head"] = True
    if "decoder_type" not in config:
        config.update(
            decoder_type="parallel",
            decoder_n_embd=config["n_embd"],
            decoder_n_layer=config["n_layer"],
            decoder_n_head=config["n_head"],
            token_dropout=0.0,
            ema_decay=0.0,
        )
    model = VQAutoencoder(AEConfig(**config)).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


@torch.no_grad()
def encode_split(model: VQAutoencoder, source: Path, destination: Path, batch_size: int, device: str):
    data = load_tokens(source)
    windows = len(data) // RAW_BLOCK
    started = time.perf_counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for first in range(0, windows, batch_size):
            count = min(batch_size, windows - first)
            raw = np.asarray(data[first * RAW_BLOCK : (first + count) * RAW_BLOCK]).reshape(count, RAW_BLOCK)
            tokens = torch.from_numpy(raw.astype(np.int64)).to(device)
            with amp_context(device):
                codes = model.encode(tokens)
            codes.cpu().numpy().astype(np.uint16).tofile(handle)
            if first == 0 or (first // batch_size) % 100 == 0:
                print(f"{source.stem}: {first + count:,}/{windows:,} blocks", flush=True)
    return {
        "raw_tokens": windows * RAW_BLOCK,
        "codes": windows * CODES_PER_BLOCK,
        "dropped_raw_tokens": len(data) - windows * RAW_BLOCK,
        "compression_ratio": COMPRESSION,
        "seconds": time.perf_counter() - started,
        "path": str(destination),
    }


def encode_dataset(args) -> None:
    device = device_from(args.device)
    model, checkpoint = load_ae(Path(args.checkpoint), device)
    data_dir, out = Path(args.data_dir), Path(args.out_dir)
    started = time.perf_counter()
    metadata = {
        "codebook_size": model.config.codebook_size,
        "autoencoder_checkpoint": str(Path(args.checkpoint)),
        "autoencoder_step": checkpoint["step"],
        "train": encode_split(model, data_dir / "train.bin", out / "train.bin", args.batch_size, device),
        "val": encode_split(model, data_dir / "val.bin", out / "val.bin", args.batch_size, device),
    }
    metadata["total_seconds"] = time.perf_counter() - started
    (out / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


@torch.no_grad()
def encode_corrected_split(
    model: VQAutoencoder,
    raw_path: Path,
    base_path: Path,
    destination: Path,
    frequencies: torch.Tensor,
    slots: int,
    surprisal_weight: float,
    batch_size: int,
    device: str,
) -> dict:
    raw, base = load_tokens(raw_path), load_tokens(base_path)
    blocks = min(len(raw) // RAW_BLOCK, len(base) // CODES_PER_BLOCK)
    started = time.perf_counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for first in range(0, blocks, batch_size):
            count = min(batch_size, blocks - first)
            indices = np.arange(first, first + count)
            targets_np = np.asarray(
                raw[first * RAW_BLOCK : (first + count) * RAW_BLOCK]
            ).reshape(count, RAW_BLOCK)
            history_np = np.empty_like(targets_np)
            for row, index in enumerate(indices):
                if index:
                    history_np[row] = raw[(index - 1) * RAW_BLOCK : index * RAW_BLOCK]
                else:
                    history_np[row].fill(model.config.bos_token_id)
            base_np = np.asarray(
                base[first * CODES_PER_BLOCK : (first + count) * CODES_PER_BLOCK]
            ).reshape(count, CODES_PER_BLOCK)
            history = torch.from_numpy(history_np.astype(np.int64)).to(device)
            targets = torch.from_numpy(targets_np.astype(np.int64)).to(device)
            codes = torch.from_numpy(base_np.astype(np.int64)).to(device)
            with amp_context(device):
                logits = model.decoder_logits(codes, history, targets, 0.0)
            losses = F.cross_entropy(logits.transpose(1, 2).float(), targets, reduction="none")
            token_frequencies = frequencies[targets]
            rarity = torch.log(
                (frequencies.sum() + model.config.vocab_size) / (token_frequencies + 1)
            )
            positions = (rarity + surprisal_weight * losses).topk(slots, dim=1).indices
            packed = (positions << 13) | targets.gather(1, positions)
            corrections = torch.stack((packed & 1023, packed >> 10), dim=-1).flatten(1)
            torch.cat((codes, corrections), dim=1).cpu().numpy().astype(np.uint16).tofile(handle)
            if first == 0 or (first // batch_size) % 100 == 0:
                print(f"{raw_path.stem}: {first + count:,}/{blocks:,} blocks", flush=True)
    serialized = blocks * (CODES_PER_BLOCK + 2 * slots)
    return {
        "raw_tokens": blocks * RAW_BLOCK,
        "serialized_tokens": serialized,
        "blocks": blocks,
        "sequence_compression": blocks * RAW_BLOCK / serialized,
        "seconds": time.perf_counter() - started,
        "path": str(destination),
    }


def encode_corrected(args) -> None:
    if not 1 <= args.slots <= 8:
        raise ValueError("--slots must be between 1 and 8")
    device = device_from(args.device)
    model, checkpoint = load_ae(Path(args.autoencoder), device)
    raw_dir, base_dir, out = Path(args.data_dir), Path(args.base_latent_dir), Path(args.out_dir)
    frequencies = torch.from_numpy(
        np.bincount(
            load_tokens(raw_dir / "train.bin"), minlength=model.config.vocab_size
        ).astype(np.float32)
    ).to(device)
    started = time.perf_counter()
    metadata = {
        "format": "corrected-v1",
        "codebook_size": model.config.codebook_size,
        "correction_slots": args.slots,
        "surprisal_weight": args.surprisal_weight,
        "autoencoder_checkpoint": str(Path(args.autoencoder)),
        "autoencoder_step": checkpoint["step"],
        "train": encode_corrected_split(
            model, raw_dir / "train.bin", base_dir / "train.bin", out / "train.bin",
            frequencies, args.slots, args.surprisal_weight, args.batch_size, device,
        ),
        "val": encode_corrected_split(
            model, raw_dir / "val.bin", base_dir / "val.bin", out / "val.bin",
            frequencies, args.slots, args.surprisal_weight, args.batch_size, device,
        ),
    }
    metadata["total_seconds"] = time.perf_counter() - started
    (out / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def corrected_forced_tokens(records: torch.Tensor, slots: int) -> torch.Tensor:
    pairs = records[:, CODES_PER_BLOCK:].view(records.shape[0], slots, 2)
    packed = pairs[..., 0] | (pairs[..., 1] << 10)
    positions, token_ids = packed >> 13, packed & 8191
    forced = torch.full(
        (records.shape[0], RAW_BLOCK), -1, dtype=torch.long, device=records.device
    )
    forced.scatter_(1, positions, token_ids)
    return forced


@torch.no_grad()
def decode_corrected_records(
    model: VQAutoencoder, records: torch.Tensor, history: torch.Tensor, slots: int
) -> torch.Tensor:
    decoded = []
    for record in records:
        row = record[None]
        block = model.reconstruct(
            row[:, :CODES_PER_BLOCK], history, forced_tokens=corrected_forced_tokens(row, slots)
        )
        decoded.append(block)
        history = block[:, -model.config.decoder_context :]
    return torch.cat(decoded, dim=1)


@torch.no_grad()
def generate_corrected_blocks(
    model: Transformer,
    seed: torch.Tensor,
    blocks: int,
    slots: int,
    raw_vocab_size: int,
    temperature: float = 0.7,
    top_k: int = 50,
) -> torch.Tensor:
    tokens = seed

    def append_allowed(high: int, valid: torch.Tensor | None = None) -> int:
        nonlocal tokens
        logits, _ = model(tokens[:, -model.config.block_size :])
        allowed = logits[:, -1, :high] / temperature
        if valid is not None:
            allowed = allowed.masked_fill(~valid[None], -torch.inf)
        cutoff = torch.topk(allowed, min(top_k, high)).values[:, -1, None]
        allowed = allowed.masked_fill(allowed < cutoff, -torch.inf)
        token = torch.multinomial(F.softmax(allowed, -1), 1)
        tokens = torch.cat((tokens, token), dim=1)
        return int(token[0, 0])

    for _ in range(blocks):
        for _ in range(CODES_PER_BLOCK):
            append_allowed(CODEBOOK_SIZE)
        for _ in range(slots):
            low = append_allowed(min(CODEBOOK_SIZE, raw_vocab_size))
            high_ids = torch.arange(512, device=tokens.device)
            valid = low + ((high_ids & 7) << 10) < raw_vocab_size
            append_allowed(512, valid)
    return tokens


@torch.no_grad()
def evaluate_corrected(
    model: VQAutoencoder,
    raw: np.memmap,
    serialized: np.memmap,
    frequencies: torch.Tensor,
    slots: int,
    batches: int,
    batch_size: int,
    rare_cutoff: int,
    device: str,
) -> dict:
    record_length = CODES_PER_BLOCK + 2 * slots
    blocks = min(len(raw) // RAW_BLOCK, len(serialized) // record_length)
    loss_total = accuracy = rare_correct = rare_total = 0.0
    usage = torch.zeros(model.config.codebook_size, device=device)
    for _ in range(batches):
        indices = np.random.randint(1, blocks, size=batch_size)
        offsets, record_offsets = np.arange(RAW_BLOCK), np.arange(record_length)
        history = torch.from_numpy(raw[(indices[:, None] - 1) * RAW_BLOCK + offsets].astype(np.int64)).to(device)
        targets = torch.from_numpy(raw[indices[:, None] * RAW_BLOCK + offsets].astype(np.int64)).to(device)
        records = torch.from_numpy(serialized[indices[:, None] * record_length + record_offsets].astype(np.int64)).to(device)
        codes = records[:, :CODES_PER_BLOCK]
        with amp_context(device):
            logits = model.decoder_logits(codes, history, targets, 0.0)
            decoded = model.reconstruct(
                codes, history, forced_tokens=corrected_forced_tokens(records, slots)
            )
        loss_total += float(F.cross_entropy(logits.flatten(0, 1), targets.flatten()))
        matches = decoded == targets
        accuracy += float(matches.float().mean())
        rare = frequencies[targets] <= rare_cutoff
        rare_correct += int((matches & rare).sum())
        rare_total += int(rare.sum())
        usage += torch.bincount(codes.flatten(), minlength=model.config.codebook_size)
    probabilities = usage / usage.sum()
    active = probabilities > 0
    mean_loss = loss_total / batches
    return {
        "reconstruction_loss": mean_loss,
        "reconstruction_perplexity": math.exp(min(20, mean_loss)),
        "token_accuracy": accuracy / batches,
        "rare_token_accuracy": rare_correct / max(1, rare_total),
        "rare_tokens": rare_total,
        "active_codes": int(active.sum()),
        "code_perplexity": float(torch.exp(-(probabilities[active] * probabilities[active].log()).sum())),
    }


@torch.no_grad()
def encode_refined_split(
    model: VQAutoencoder,
    raw_path: Path,
    base_path: Path,
    destination: Path,
    batch_size: int,
    device: str,
) -> dict:
    raw, base = load_tokens(raw_path), load_tokens(base_path)
    blocks = min(len(raw) // RAW_BLOCK, len(base) // CODES_PER_BLOCK)
    refinements = serialized = 0
    started = time.perf_counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    bit_values = 2 ** np.arange(CODES_PER_BLOCK, dtype=np.uint16)
    with destination.open("wb") as handle:
        for first in range(0, blocks, batch_size):
            count = min(batch_size, blocks - first)
            indices = np.arange(first, first + count)
            targets_np = np.asarray(raw[first * RAW_BLOCK : (first + count) * RAW_BLOCK]).reshape(
                count, RAW_BLOCK
            )
            history_np = np.empty_like(targets_np)
            for row, index in enumerate(indices):
                if index:
                    history_np[row] = raw[(index - 1) * RAW_BLOCK : index * RAW_BLOCK]
                else:
                    history_np[row].fill(model.config.bos_token_id)
            base_np = np.asarray(
                base[first * CODES_PER_BLOCK : (first + count) * CODES_PER_BLOCK]
            ).reshape(count, CODES_PER_BLOCK)
            history = torch.from_numpy(history_np.astype(np.int64)).to(device)
            targets = torch.from_numpy(targets_np.astype(np.int64)).to(device)
            base_codes = torch.from_numpy(base_np.astype(np.int64)).to(device)
            with amp_context(device):
                base_logits = model.decoder_logits(base_codes, history, targets)
                mask = grouped_token_losses(base_logits.float(), targets) > model.config.refinement_threshold
                _, refinement_codes, _ = model.encode_refinement_vectors(targets)
            mask_np = mask.cpu().numpy()
            refinement_np = refinement_codes.cpu().numpy().astype(np.uint16)
            mask_values = (mask_np.astype(np.uint16) * bit_values).sum(1)
            records = []
            for row in range(count):
                selected = refinement_np[row, mask_np[row]]
                record = np.concatenate(
                    (
                        base_np[row].astype(np.uint16),
                        np.asarray(
                            [
                                MASK_TOKEN_BASE + (int(mask_values[row]) & 255),
                                MASK_TOKEN_BASE + (int(mask_values[row]) >> 8),
                            ],
                            dtype=np.uint16,
                        ),
                        selected,
                    )
                )
                records.append(record)
                refinements += len(selected)
                serialized += len(record)
            np.concatenate(records).tofile(handle)
            if first == 0 or (first // batch_size) % 100 == 0:
                print(f"{raw_path.stem}: {first + count:,}/{blocks:,} blocks", flush=True)
    raw_tokens = blocks * RAW_BLOCK
    return {
        "raw_tokens": raw_tokens,
        "serialized_tokens": serialized,
        "blocks": blocks,
        "refinements": refinements,
        "average_refinements": refinements / blocks,
        "sequence_compression": raw_tokens / serialized,
        "seconds": time.perf_counter() - started,
        "path": str(destination),
    }


def encode_refined(args) -> None:
    device = device_from(args.device)
    model, checkpoint = load_ae(Path(args.checkpoint), device)
    if not model.config.refinement:
        raise ValueError("checkpoint has no refinement modules")
    if not 0 < args.rate <= 1:
        raise ValueError("--rate must be in (0, 1]")
    model.config.refinement_rate = args.rate
    raw_dir, base_dir, out = Path(args.data_dir), Path(args.base_latent_dir), Path(args.out_dir)
    model.config.refinement_threshold = calibrate_refinement_threshold(
        model,
        load_tokens(raw_dir / "val.bin"),
        load_tokens(base_dir / "val.bin"),
        model.config.refinement_rate,
        args.calibration_batches,
        args.batch_size,
        device,
    )
    print(f"recalibrated refinement threshold: {model.config.refinement_threshold:.4f}", flush=True)
    started = time.perf_counter()
    metadata = {
        "format": "refined-v1",
        "vocab_size": REFINED_VOCAB_SIZE,
        "codebook_size": model.config.codebook_size,
        "refinement_threshold": model.config.refinement_threshold,
        "autoencoder_checkpoint": str(Path(args.checkpoint)),
        "autoencoder_step": checkpoint["step"],
        "train": encode_refined_split(
            model,
            raw_dir / "train.bin",
            base_dir / "train.bin",
            out / "train.bin",
            args.batch_size,
            device,
        ),
        "val": encode_refined_split(
            model,
            raw_dir / "val.bin",
            base_dir / "val.bin",
            out / "val.bin",
            args.batch_size,
            device,
        ),
    }
    metadata["total_seconds"] = time.perf_counter() - started
    (out / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def sweep_refinement(args) -> None:
    device = device_from(args.device)
    model, _ = load_ae(Path(args.checkpoint), device)
    raw = load_tokens(Path(args.data_dir) / "val.bin")
    latent = load_tokens(Path(args.base_latent_dir) / "val.bin")
    rows = []
    for extras in args.extras:
        if not 1 <= extras <= CODES_PER_BLOCK:
            raise ValueError("--extras values must be between 1 and 16")
        rate = extras / CODES_PER_BLOCK
        np.random.seed(1337)
        model.config.refinement_threshold = calibrate_refinement_threshold(
            model, raw, latent, rate, args.calibration_batches, args.batch_size, device
        )
        np.random.seed(7331)
        metrics = evaluate_refiner(
            model, raw, latent, args.eval_batches, args.batch_size, device, decode=True
        )
        serialized = CODES_PER_BLOCK + 2 + metrics["average_refinements"]
        rows.append(
            {
                "target_extras": extras,
                "threshold": model.config.refinement_threshold,
                "compression": RAW_BLOCK / serialized,
                **metrics,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    table = [
        "# Refinement-rate sweep",
        "",
        "| Target extras | Actual extras | Compression | Greedy accuracy | Perplexity |",
        "|---:|---:|---:|---:|---:|",
    ]
    table += [
        f"| {row['target_extras']} | {row['average_refinements']:.2f} | "
        f"{row['compression']:.2f}x | {row['token_accuracy']:.2%} | "
        f"{row['reconstruction_perplexity']:.3f} |"
        for row in rows
    ]
    (out / "sweep.md").write_text("\n".join(table) + "\n", encoding="utf-8")


def pack_literal_tail(positions: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
    if len(positions) != len(token_ids) or len(positions) > RAW_BLOCK:
        raise ValueError("invalid literal escape arrays")
    values = [len(positions)]
    for position, token_id in zip(positions, token_ids):
        if not 0 <= int(position) < RAW_BLOCK or not 0 <= int(token_id) < 65536:
            raise ValueError("literal position or token is out of range")
        values += [int(position), int(token_id) & 255, int(token_id) >> 8]
    return np.asarray(values, dtype=np.uint16) + MASK_TOKEN_BASE


def pack_correction(position: int, token_id: int) -> tuple[int, int]:
    if not 0 <= position < RAW_BLOCK or not 0 <= token_id < 8192:
        raise ValueError("correction position or token is out of range")
    packed = (position << 13) | token_id
    return packed & 1023, packed >> 10


def unpack_correction(low: int, high: int) -> tuple[int, int]:
    if not 0 <= low < 1024 or not 0 <= high < 512:
        raise ValueError("invalid packed correction")
    packed = low | (high << 10)
    return packed >> 13, packed & 8191


def unpack_literal_tail(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(data, dtype=np.int64) - MASK_TOKEN_BASE
    if not len(values) or len(values) != 1 + 3 * int(values[0]):
        raise ValueError("invalid literal escape tail")
    triples = values[1:].reshape(-1, 3)
    return triples[:, 0], triples[:, 1] | (triples[:, 2] << 8)


@torch.no_grad()
def sweep_literals(args) -> None:
    device = device_from(args.device)
    model, _ = load_ae(Path(args.checkpoint), device)
    raw_dir, latent_dir = Path(args.data_dir), Path(args.base_latent_dir)
    raw, latent = load_tokens(raw_dir / "val.bin"), load_tokens(latent_dir / "val.bin")
    frequencies = np.bincount(load_tokens(raw_dir / "train.bin"), minlength=model.config.vocab_size)
    frequency_tensor = torch.from_numpy(frequencies.astype(np.int64)).to(device)
    model.config.refinement_threshold = calibrate_refinement_threshold(
        model, raw, latent, args.rate, args.calibration_batches, args.batch_size, device
    )
    totals = {cutoff: {"selected": 0, "rare": 0, "rare_before": 0} for cutoff in args.cutoffs}
    correct = refinements = tokens_seen = 0
    np.random.seed(7331)
    for _ in range(args.eval_batches):
        history, targets, codes = context_batch(raw, latent, args.batch_size, device)
        with amp_context(device):
            base_logits = model.decoder_logits(codes, history, targets)
            refinement_mask = grouped_token_losses(base_logits.float(), targets) > model.config.refinement_threshold
            _, refinement_codes, _ = model.encode_refinement_vectors(targets)
            decoded = model.reconstruct(codes, history, refinement_codes, refinement_mask)
        matches = decoded == targets
        token_frequencies = frequency_tensor[targets]
        correct += int(matches.sum())
        refinements += int(refinement_mask.sum())
        tokens_seen += targets.numel()
        for cutoff, total in totals.items():
            rare = token_frequencies <= cutoff
            candidates = rare & ~matches
            ranked = token_frequencies.masked_fill(~candidates, frequencies.max() + 1).argsort(1)
            chosen = ranked[:, : args.max_literals]
            selected = torch.zeros_like(candidates)
            selected.scatter_(1, chosen, candidates.gather(1, chosen))
            total["selected"] += int(selected.sum())
            total["rare"] += int(rare.sum())
            total["rare_before"] += int((matches & rare).sum())
    blocks = tokens_seen / RAW_BLOCK
    average_refinements = refinements / blocks
    rows = []
    for cutoff, total in totals.items():
        literals = total["selected"] / blocks
        rows.append(
            {
                "frequency_cutoff": cutoff,
                "average_literals": literals,
                "compression": RAW_BLOCK / (CODES_PER_BLOCK + 2 + average_refinements + 1 + 3 * literals),
                "token_accuracy": (correct + total["selected"]) / tokens_seen,
                "rare_token_accuracy_before": total["rare_before"] / max(1, total["rare"]),
                "rare_token_accuracy_after": (total["rare_before"] + total["selected"]) / max(1, total["rare"]),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "literal-sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    table = ["# Literal fallback sweep", "", "| Max frequency | Literals/block | Compression | Greedy accuracy | Rare accuracy |", "|---:|---:|---:|---:|---:|"]
    table += [f"| {r['frequency_cutoff']} | {r['average_literals']:.2f} | {r['compression']:.2f}x | {r['token_accuracy']:.2%} | {r['rare_token_accuracy_before']:.2%} -> {r['rare_token_accuracy_after']:.2%} |" for r in rows]
    (out / "literal-sweep.md").write_text("\n".join(table) + "\n", encoding="utf-8")


def parse_refined_records(data: np.ndarray | np.memmap, limit: int | None = None):
    position = blocks = 0
    while position < len(data) and (limit is None or blocks < limit):
        if position + CODES_PER_BLOCK + 2 > len(data):
            raise ValueError("truncated refined latent record")
        base = np.asarray(data[position : position + CODES_PER_BLOCK]).astype(np.int64)
        low, high = (int(value) - MASK_TOKEN_BASE for value in data[position + 16 : position + 18])
        if not (0 <= low < 256 and 0 <= high < 256):
            raise ValueError("invalid refinement mask token")
        mask_value = low | (high << 8)
        mask = np.asarray([(mask_value >> bit) & 1 for bit in range(CODES_PER_BLOCK)], dtype=bool)
        count = int(mask.sum())
        start = position + CODES_PER_BLOCK + 2
        extras = np.asarray(data[start : start + count]).astype(np.int64)
        if len(extras) != count:
            raise ValueError("truncated refinement codes")
        yield base, mask, extras, np.asarray(data[position : start + count]).astype(np.int64)
        position = start + count
        blocks += 1


@torch.no_grad()
def generate_refined_blocks(
    model: Transformer,
    seed: torch.Tensor,
    blocks: int,
    temperature: float = 0.7,
    top_k: int = 50,
) -> torch.Tensor:
    tokens = seed

    def append_allowed(low: int, high: int) -> int:
        nonlocal tokens
        logits, _ = model(tokens[:, -model.config.block_size :])
        allowed = logits[:, -1, low:high] / temperature
        cutoff = torch.topk(allowed, min(top_k, allowed.shape[-1])).values[:, -1, None]
        allowed = allowed.masked_fill(allowed < cutoff, -torch.inf)
        relative = torch.multinomial(F.softmax(allowed, -1), 1)
        token = relative + low
        tokens = torch.cat((tokens, token), dim=1)
        return int(token[0, 0])

    for _ in range(blocks):
        for _ in range(CODES_PER_BLOCK):
            append_allowed(0, CODEBOOK_SIZE)
        low = append_allowed(MASK_TOKEN_BASE, REFINED_VOCAB_SIZE) - MASK_TOKEN_BASE
        high = append_allowed(MASK_TOKEN_BASE, REFINED_VOCAB_SIZE) - MASK_TOKEN_BASE
        for _ in range((low | (high << 8)).bit_count()):
            append_allowed(0, CODEBOOK_SIZE)
    return tokens


@torch.no_grad()
def decode_refined_records(
    model: VQAutoencoder,
    records,
    history: torch.Tensor,
    device: str,
) -> torch.Tensor:
    decoded = []
    for base, mask, extras, _ in records:
        base_codes = torch.from_numpy(base)[None].to(device)
        mask_tensor = torch.from_numpy(mask)[None].to(device)
        refinement = torch.zeros_like(base_codes)
        refinement[mask_tensor] = torch.from_numpy(extras).to(device)
        block = model.reconstruct(base_codes, history, refinement, mask_tensor)
        decoded.append(block)
        history = block[:, -model.config.decoder_context :]
    return torch.cat(decoded, dim=1)


@torch.no_grad()
def evaluate_lm(model: Transformer, data: np.memmap, batches: int, batch_size: int, device: str) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = lm_batch(data, model.config.block_size, batch_size, device)
        with amp_context(device):
            _, loss = model(x, y)
        total += float(loss)
    model.train(was_training)
    return total / batches


def lm_payload(
    model, optimizer, kind: str, step: int, target_steps: int, best: float, elapsed: float, val_loss: float, args
):
    units = step * args.batch_size * model.config.block_size * args.grad_accum
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "kind": kind,
        "step": step,
        "target_steps": target_steps,
        "best_val_loss": best,
        "val_loss": val_loss,
        "elapsed_seconds": elapsed,
        "training_units": units,
        "equivalent_bpe_tokens": units * args.unit_ratio,
        "model_config": asdict(model.config),
    }


def train_lm(args) -> None:
    device = device_from(args.device)
    if not args.batch_size:
        args.batch_size = 64 if args.kind in ("latent", "latent40", "refined", "corrected") else 16
    if not args.lr:
        args.lr = 6e-4 if args.kind in ("latent", "latent40", "refined", "corrected") else 3e-4
    default_data = {
        "baseline": "data/tinystories",
        "latent": "data/latent-v2",
        "latent40": "data/latent-v2",
        "refined": "data/latent-v4-refined",
        "corrected": "data/latent-v3.3-corrected",
    }
    data_dir = Path(args.data_dir or default_data[args.kind])
    out = Path(
        args.out
        or {
            "baseline": "checkpoints/baseline",
            "latent": "checkpoints/latent-v2-b64",
            "latent40": "checkpoints/latent40-v3",
            "refined": "checkpoints/latent-v4-refined",
            "corrected": "checkpoints/latent-v3.3-corrected",
        }[args.kind]
    )
    metadata = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    vocab_size = metadata["vocab_size"] if args.kind in ("baseline", "refined") else metadata["codebook_size"]
    args.unit_ratio = (
        1.0
        if args.kind == "baseline"
        else metadata["train"].get("sequence_compression", COMPRESSION)
    )
    train, val = load_tokens(data_dir / "train.bin"), load_tokens(data_dir / "val.bin")
    config = lm_config(args.kind, vocab_size)
    model = Transformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    target_steps = args.steps or math.ceil(
        (len(train) - 1) / (args.batch_size * config.block_size * args.grad_accum)
    )
    latest = out / "latest.pt"
    step, best, elapsed_before = 0, float("inf"), 0.0
    if latest.exists():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if checkpoint["kind"] != args.kind or checkpoint["model_config"] != asdict(config):
            raise ValueError(f"{latest} is incompatible with this run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        best = checkpoint["best_val_loss"]
        elapsed_before = checkpoint.get("elapsed_seconds", 0.0)
        print(f"resuming {args.kind} LM at step {step:,}")
    print(
        f"{args.kind} LM: {model.num_parameters():,} parameters, {target_steps:,} steps on {device}",
        flush=True,
    )
    if step >= target_steps:
        print(f"already reached {target_steps:,} steps")
        return

    started = time.perf_counter()
    val_loss = float("inf")
    model.train()
    while step < target_steps:
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = lm_batch(train, config.block_size, args.batch_size, device)
            with amp_context(device):
                _, loss = model(x, y)
                scaled_loss = loss / args.grad_accum
            scaled_loss.backward()
            train_loss += float(scaled_loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = learning_rate(step, target_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        step += 1

        should_eval = step == 1 or step % args.eval_every == 0 or step == target_steps
        if should_eval:
            val_loss = evaluate_lm(model, val, args.eval_batches, args.batch_size, device)
            row = {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "equivalent_bpe_tokens": step
                * args.batch_size
                * config.block_size
                * args.grad_accum
                * args.unit_ratio,
            }
            append_jsonl(out / "metrics.jsonl", row)
            print(json.dumps(row), flush=True)
            elapsed = elapsed_before + time.perf_counter() - started
            if val_loss < best:
                best = val_loss
                atomic_save(
                    lm_payload(model, optimizer, args.kind, step, target_steps, best, elapsed, val_loss, args),
                    out / "best.pt",
                )

        if step % args.save_every == 0 or step == target_steps:
            elapsed = elapsed_before + time.perf_counter() - started
            atomic_save(
                lm_payload(model, optimizer, args.kind, step, target_steps, best, elapsed, val_loss, args),
                latest,
            )


def load_lm(path: Path, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = Transformer(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def correction_head_batch(
    data: np.memmap,
    record_length: int,
    slots: int,
    batch_size: int,
    device: str,
):
    blocks = len(data) // record_length
    indices = np.random.randint(3, blocks, size=batch_size)
    context_blocks = indices[:, None] + np.arange(-3, 1)
    base_offsets = np.arange(CODES_PER_BLOCK)
    contexts = data[
        context_blocks[:, :, None] * record_length + base_offsets[None, None, :]
    ].reshape(batch_size, -1).astype(np.int64)
    correction_offsets = CODES_PER_BLOCK + np.arange(2 * slots)
    corrections = data[indices[:, None] * record_length + correction_offsets].reshape(
        batch_size, slots, 2
    ).astype(np.int64)
    packed = corrections[..., 0] | (corrections[..., 1] << 10)
    return (
        torch.from_numpy(contexts).to(device),
        torch.from_numpy(packed >> 13).to(device),
        torch.from_numpy(packed & 8191).to(device),
    )


@torch.no_grad()
def evaluate_correction_head(
    base_model: Transformer,
    head: CorrectionHead,
    data: np.memmap,
    frequencies: torch.Tensor,
    record_length: int,
    slots: int,
    batches: int,
    batch_size: int,
    rare_cutoff: int,
    device: str,
) -> dict:
    head.eval()
    totals = {"loss": 0.0, "apply_tp": 0, "apply_fp": 0, "apply_fn": 0, "position": 0, "token": 0, "token_top5": 0, "pairs": 0, "positive": 0}
    for _ in range(batches):
        contexts, positions, token_ids = correction_head_batch(
            data, record_length, slots, batch_size, device
        )
        apply_targets = frequencies[token_ids] <= rare_cutoff
        hidden = base_model.hidden(contexts)
        apply_logits, position_logits, token_logits = head(hidden)
        positive = apply_targets.sum().clamp_min(1)
        loss = F.binary_cross_entropy_with_logits(apply_logits, apply_targets.float())
        loss += (F.cross_entropy(position_logits.flatten(0, 1), positions.flatten(), reduction="none").view_as(positions) * apply_targets).sum() / positive
        loss += (F.cross_entropy(token_logits.flatten(0, 1), token_ids.flatten(), reduction="none").view_as(token_ids) * apply_targets).sum() / positive
        predicted_apply = apply_logits.sigmoid() >= 0.5
        totals["loss"] += float(loss)
        totals["apply_tp"] += int((predicted_apply & apply_targets).sum())
        totals["apply_fp"] += int((predicted_apply & ~apply_targets).sum())
        totals["apply_fn"] += int((~predicted_apply & apply_targets).sum())
        position_match = position_logits.argmax(-1) == positions
        token_match = token_logits.argmax(-1) == token_ids
        top5_match = (token_logits.topk(5, -1).indices == token_ids[..., None]).any(-1)
        totals["position"] += int((position_match & apply_targets).sum())
        totals["token"] += int((token_match & apply_targets).sum())
        totals["token_top5"] += int((top5_match & apply_targets).sum())
        totals["pairs"] += int((position_match & token_match & apply_targets).sum())
        totals["positive"] += int(apply_targets.sum())
    positive = max(1, totals["positive"])
    precision = totals["apply_tp"] / max(1, totals["apply_tp"] + totals["apply_fp"])
    recall = totals["apply_tp"] / max(1, totals["apply_tp"] + totals["apply_fn"])
    return {
        "loss": totals["loss"] / batches,
        "apply_precision": precision,
        "apply_recall": recall,
        "apply_f1": 2 * precision * recall / max(1e-9, precision + recall),
        "position_accuracy": totals["position"] / positive,
        "token_accuracy": totals["token"] / positive,
        "token_top5_accuracy": totals["token_top5"] / positive,
        "exact_pair_accuracy": totals["pairs"] / positive,
        "positive_rate": totals["positive"] / (batches * batch_size * slots),
    }


def correction_head_payload(head, optimizer, step, best, elapsed, metrics, args, base_model):
    payload = {
        "model": head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_loss": best,
        "elapsed_seconds": elapsed,
        "metrics": metrics,
        "slots": args.slots,
        "bottleneck": args.bottleneck,
        "vocab_size": args.vocab_size,
        "base_model": str(args.base_model),
        "base_width": base_model.config.n_embd,
        "joint": args.joint,
    }
    if args.joint:
        payload["base_model_state"] = base_model.state_dict()
    return payload


def train_correction_head(args) -> None:
    device = device_from(args.device)
    base_model, _ = load_lm(Path(args.base_model), device)
    for parameter in base_model.parameters():
        parameter.requires_grad_(args.joint)
    meta = json.loads((Path(args.corrected_dir) / "meta.json").read_text(encoding="utf-8"))
    args.slots = meta["correction_slots"]
    args.vocab_size = load_tokenizer(Path(args.data_dir)).get_vocab_size()
    record_length = CODES_PER_BLOCK + 2 * args.slots
    train = load_tokens(Path(args.corrected_dir) / "train.bin")
    val = load_tokens(Path(args.corrected_dir) / "val.bin")
    frequencies = torch.from_numpy(
        np.bincount(load_tokens(Path(args.data_dir) / "train.bin"), minlength=args.vocab_size).astype(np.int64)
    ).to(device)
    out, latest = Path(args.out), Path(args.out) / "latest.pt"
    head = CorrectionHead(base_model.config.n_embd, args.slots, args.bottleneck, args.vocab_size).to(device)
    parameters = list(head.parameters()) + (
        list(base_model.parameters()) if args.joint else []
    )
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    steps = args.steps or math.ceil((len(train) // record_length) / args.batch_size)
    step, best, elapsed_before = 0, float("inf"), 0.0
    if latest.exists():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        head.load_state_dict(checkpoint["model"])
        if args.joint:
            base_model.load_state_dict(checkpoint["base_model_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, best = checkpoint["step"], checkpoint["best_loss"]
        elapsed_before = checkpoint.get("elapsed_seconds", 0.0)
        print(f"resuming correction head at step {step:,}")
    print(f"correction head: {sum(p.numel() for p in head.parameters()):,} parameters, {steps:,} steps on {device}", flush=True)
    if step >= steps:
        print(f"already reached {steps:,} steps")
        return
    started = time.perf_counter()
    metrics = {}
    head.train()
    while step < steps:
        contexts, positions, token_ids = correction_head_batch(
            train, record_length, args.slots, args.batch_size, device
        )
        apply_targets = frequencies[token_ids] <= args.rare_cutoff
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device):
            if args.joint:
                hidden = base_model.hidden(contexts)
            else:
                with torch.no_grad():
                    hidden = base_model.hidden(contexts)
            apply_logits, position_logits, token_logits = head(hidden)
            positive = apply_targets.sum().clamp_min(1)
            apply_loss = F.binary_cross_entropy_with_logits(apply_logits, apply_targets.float())
            position_loss = (F.cross_entropy(position_logits.flatten(0, 1), positions.flatten(), reduction="none").view_as(positions) * apply_targets).sum() / positive
            token_loss = (F.cross_entropy(token_logits.flatten(0, 1), token_ids.flatten(), reduction="none").view_as(token_ids) * apply_targets).sum() / positive
            code_loss = (
                F.cross_entropy(
                    base_model.head(hidden[:, :-1]).flatten(0, 1), contexts[:, 1:].flatten()
                )
                if args.joint
                else token_loss.new_zeros(())
            )
            loss = (
                args.code_weight * code_loss
                + args.apply_weight * apply_loss
                + args.position_weight * position_loss
                + args.token_weight * token_loss
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        step += 1
        if step == 1 or step % args.eval_every == 0 or step == steps:
            metrics = evaluate_correction_head(
                base_model, head, val, frequencies, record_length, args.slots,
                args.eval_batches, args.batch_size, args.rare_cutoff, device,
            )
            row = {
                "step": step,
                "train_loss": float(loss.detach()),
                "code_loss": float(code_loss.detach()),
                **metrics,
            }
            append_jsonl(out / "metrics.jsonl", row)
            print(json.dumps(row), flush=True)
            elapsed = elapsed_before + time.perf_counter() - started
            payload = correction_head_payload(head, optimizer, step, min(best, metrics["loss"]), elapsed, metrics, args, base_model)
            if metrics["loss"] < best:
                best = metrics["loss"]
                atomic_save(payload, out / "best.pt")
        if step % args.save_every == 0 or step == steps:
            elapsed = elapsed_before + time.perf_counter() - started
            atomic_save(correction_head_payload(head, optimizer, step, best, elapsed, metrics, args, base_model), latest)


def load_correction_head(path: Path, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    head = CorrectionHead(
        checkpoint["base_width"], checkpoint["slots"], checkpoint["bottleneck"], checkpoint["vocab_size"]
    ).to(device)
    head.load_state_dict(checkpoint["model"])
    head.eval()
    return head, checkpoint


@torch.no_grad()
def structured_correction_diagnostic(args) -> None:
    device = device_from(args.device)
    codec, _ = load_ae(Path(args.autoencoder), device)
    base_model, _ = load_lm(Path(args.base_model), device)
    head, head_checkpoint = load_correction_head(Path(args.correction_head), device)
    if "base_model_state" in head_checkpoint:
        base_model.load_state_dict(head_checkpoint["base_model_state"])
    slots = head_checkpoint["slots"]
    record_length = CODES_PER_BLOCK + 2 * slots
    raw_dir, corrected_dir = Path(args.data_dir), Path(args.corrected_dir)
    raw, records = load_tokens(raw_dir / "val.bin"), load_tokens(corrected_dir / "val.bin")
    frequencies = torch.from_numpy(
        np.bincount(load_tokens(raw_dir / "train.bin"), minlength=codec.config.vocab_size).astype(np.int64)
    ).to(device)
    overall_correct = rare_correct = rare_total = total = applied = 0
    for first in range(3, args.blocks + 3, args.batch_size):
        count = min(args.batch_size, args.blocks + 3 - first)
        indices = np.arange(first, first + count)
        context_blocks = indices[:, None] + np.arange(-3, 1)
        base_offsets = np.arange(CODES_PER_BLOCK)
        contexts = torch.from_numpy(
            records[context_blocks[:, :, None] * record_length + base_offsets[None, None, :]]
            .reshape(count, -1).astype(np.int64)
        ).to(device)
        current = contexts[:, -CODES_PER_BLOCK:]
        offsets = np.arange(RAW_BLOCK)
        history = torch.from_numpy(raw[(indices[:, None] - 1) * RAW_BLOCK + offsets].astype(np.int64)).to(device)
        targets = torch.from_numpy(raw[indices[:, None] * RAW_BLOCK + offsets].astype(np.int64)).to(device)
        with amp_context(device):
            hidden = base_model.hidden(contexts)
            apply_logits, position_logits, token_logits = head(hidden)
        apply_mask = apply_logits.sigmoid() >= args.threshold
        positions, token_ids = position_logits.argmax(-1), token_logits.argmax(-1)
        forced = torch.full_like(targets, -1)
        for slot in range(slots):
            rows = torch.where(apply_mask[:, slot])[0]
            forced[rows, positions[rows, slot]] = token_ids[rows, slot]
        decoded = codec.reconstruct(current, history, forced_tokens=forced)
        matches = decoded == targets
        rare = frequencies[targets] <= args.rare_cutoff
        overall_correct += int(matches.sum())
        rare_correct += int((matches & rare).sum())
        rare_total += int(rare.sum())
        total += targets.numel()
        applied += int(apply_mask.sum())
    result = {
        "blocks": args.blocks,
        "threshold": args.threshold,
        "overall_accuracy": overall_correct / total,
        "rare_accuracy": rare_correct / max(1, rare_total),
        "corrections_applied_per_block": applied / args.blocks,
        "head_validation": head_checkpoint["metrics"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def accounting_checkpoint(path: Path, device: str) -> dict:
    latest = path.with_name("latest.pt")
    return torch.load(latest if latest.exists() else path, map_location=device, weights_only=False)


def repair_with_lmstudio(text: str, url: str, model: str, timeout: float) -> str:
    repaired = ""
    for attempt in range(2):
        instruction = (
            "You conservatively repair malformed generated children's story text. Rewrite every "
            "sentence and paragraph; do not summarize, truncate, or omit content. Preserve names, "
            "events, objects, order, and length. Fix grammar, broken fragments, accidental repetition, "
            "and direct contradictions. Do not invent a different story, add commentary, or explain "
            "changes. Return only the complete repaired text."
        )
        if attempt:
            instruction += " Your previous answer was too short; the answer must be at least 70% as long as the input."
        payload = json.dumps(
            {
                "model": model,
                "temperature": 0.2,
                "max_tokens": 700,
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": text},
                ],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url.rstrip("/") + "/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            repaired = result["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as error:
            raise RuntimeError(f"LM Studio repair failed: {error}") from error
        if len(repaired) >= 0.7 * len(text):
            return repaired
    return text


@torch.no_grad()
def report(args) -> None:
    device = device_from(args.device)
    raw_dir, latent_dir, out = Path(args.data_dir), Path(args.latent_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(raw_dir)
    ae, ae_best = load_ae(Path(args.autoencoder), device)
    baseline, baseline_best = load_lm(Path(args.baseline), device)
    latent, latent_best = load_lm(Path(args.latent), device)
    ae_accounting = accounting_checkpoint(Path(args.autoencoder), device)
    baseline_accounting = accounting_checkpoint(Path(args.baseline), device)
    latent_accounting = accounting_checkpoint(Path(args.latent), device)
    raw_val, latent_val = load_tokens(raw_dir / "val.bin"), load_tokens(latent_dir / "val.bin")
    latent_meta = json.loads((latent_dir / "meta.json").read_text(encoding="utf-8"))

    reconstruction = (
        evaluate_decoder(
            ae, raw_val, latent_val, args.eval_batches, args.eval_batch_size, device, decode=True
        )
        if ae.config.decoder_context
        else evaluate_ae(ae, raw_val, args.eval_batches, args.eval_batch_size, device, decode=True)
    )
    examples = []
    first_block = 1 if ae.config.decoder_context else 0
    for block_index in range(first_block, first_block + 3):
        raw = np.asarray(raw_val[block_index * RAW_BLOCK : (block_index + 1) * RAW_BLOCK]).astype(np.int64)
        tokens = torch.from_numpy(raw)[None].to(device)
        history = None
        if ae.config.decoder_context:
            previous = np.asarray(
                raw_val[(block_index - 1) * RAW_BLOCK : block_index * RAW_BLOCK]
            ).astype(np.int64)
            history = torch.from_numpy(previous)[None].to(device)
        with amp_context(device):
            codes = ae.encode(tokens)
            decoded = ae.reconstruct(codes, history)
        examples.append(
            {
                "original": tokenizer.decode(raw.tolist(), skip_special_tokens=True),
                "reconstructed": tokenizer.decode(decoded[0].tolist(), skip_special_tokens=True),
            }
        )
    reconstruction_text = ["# Reconstruction samples", ""]
    for index, example in enumerate(examples, 1):
        reconstruction_text += [
            f"## Sample {index}",
            "",
            f"Original: {example['original']}",
            "",
            f"Reconstructed: {example['reconstructed']}",
            "",
        ]
    (out / "reconstructions.md").write_text("\n".join(reconstruction_text), encoding="utf-8")

    if args.new_tokens % RAW_BLOCK:
        raise ValueError(f"--new-tokens must be divisible by {RAW_BLOCK}")
    # Warm kernels without counting them.
    warm_raw = torch.from_numpy(np.asarray(raw_val[:RAW_BLOCK]).astype(np.int64))[None].to(device)
    warm_codes = torch.from_numpy(np.asarray(latent_val[:CODES_PER_BLOCK]).astype(np.int64))[None].to(device)
    with amp_context(device):
        baseline(warm_raw)
        latent(warm_codes)
        ae.reconstruct(warm_codes, warm_raw if ae.config.decoder_context else None)
    synchronize(device)

    pairs, timings = [], {"baseline": [], "latent": [], "repair": []}
    for sample_index in range(args.samples):
        block_index = sample_index * 17
        raw_start, code_start = block_index * RAW_BLOCK, block_index * CODES_PER_BLOCK
        raw_seed_np = np.asarray(raw_val[raw_start : raw_start + RAW_BLOCK]).astype(np.int64)
        code_seed_np = np.asarray(latent_val[code_start : code_start + CODES_PER_BLOCK]).astype(np.int64)
        raw_seed = torch.from_numpy(raw_seed_np)[None].to(device)
        code_seed = torch.from_numpy(code_seed_np)[None].to(device)
        prompt = tokenizer.decode(raw_seed_np.tolist(), skip_special_tokens=True)

        torch.manual_seed(1337 + sample_index)
        synchronize(device)
        started = time.perf_counter()
        with amp_context(device):
            baseline_ids = baseline.generate(raw_seed, args.new_tokens)
        baseline_text = tokenizer.decode(
            baseline_ids[0, RAW_BLOCK:].cpu().tolist(), skip_special_tokens=True
        )
        synchronize(device)
        timings["baseline"].append(time.perf_counter() - started)

        torch.manual_seed(1337 + sample_index)
        synchronize(device)
        started = time.perf_counter()
        new_codes = args.new_tokens // COMPRESSION
        with amp_context(device):
            latent_ids = latent.generate(code_seed, new_codes, temperature=0.7, top_k=50)[
                0, CODES_PER_BLOCK:
            ]
            decoded_ids = ae.reconstruct_sequence(
                latent_ids[None], raw_seed if ae.config.decoder_context else None
            ).flatten()
        latent_text = tokenizer.decode(decoded_ids.cpu().tolist(), skip_special_tokens=True)
        synchronize(device)
        timings["latent"].append(time.perf_counter() - started)
        pair = {"prompt": prompt, "baseline": baseline_text, "latent": latent_text}
        if args.repair_model:
            started = time.perf_counter()
            pair["repaired"] = repair_with_lmstudio(
                latent_text, args.repair_url, args.repair_model, args.repair_timeout
            )
            timings["repair"].append(time.perf_counter() - started)
        pairs.append(pair)

    rng = random.Random(1337)
    blind_text = [
        "# Blinded sample comparison",
        "",
        "Rate each continuation 1-5 for coherence and story quality before opening `sample_key.json`.",
        "",
    ]
    key = {}
    for index, pair in enumerate(pairs, 1):
        names = ["baseline", "latent"] + (["repaired"] if args.repair_model else [])
        rng.shuffle(names)
        labels = [chr(ord("A") + position) for position in range(len(names))]
        key[str(index)] = dict(zip(labels, names))
        blind_text += [
            f"## Pair {index}",
            "",
            f"Prompt: {pair['prompt']}",
            "",
        ]
        for label, name in zip(labels, names):
            blind_text += [f"{label}: {pair[name]}", ""]
    (out / "samples_blind.md").write_text("\n".join(blind_text), encoding="utf-8")
    (out / "sample_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")

    baseline_seconds = baseline_accounting["elapsed_seconds"]
    latent_lm_seconds = latent_accounting["elapsed_seconds"]
    ae_seconds = ae_accounting["elapsed_seconds"]
    encoding_seconds = latent_meta["total_seconds"]
    latent_total = ae_seconds + encoding_seconds + latent_lm_seconds
    one_time_codec_seconds = ae_seconds + encoding_seconds
    per_generator_savings = baseline_seconds - latent_lm_seconds
    break_even_runs = (
        math.floor(one_time_codec_seconds / per_generator_savings) + 1 if per_generator_savings > 0 else None
    )
    baseline_latency = sum(timings["baseline"]) / len(timings["baseline"])
    latent_latency = sum(timings["latent"]) / len(timings["latent"])
    repair_latency = (
        sum(timings["repair"]) / len(timings["repair"]) if timings["repair"] else None
    )
    raw_count = latent_meta["train"]["raw_tokens"]
    code_count = latent_meta["train"]["codes"]
    metrics = {
        "reconstruction": reconstruction,
        "compression": {
            "raw_bpe_tokens": raw_count,
            "latent_codes": code_count,
            "ratio": raw_count / code_count,
        },
        "training_seconds": {
            "autoencoder": ae_seconds,
            "encoding": encoding_seconds,
            "latent_model": latent_lm_seconds,
            "latent_total": latent_total,
            "baseline_model": baseline_seconds,
            "latent_model_speedup": baseline_seconds / latent_lm_seconds,
            "total_speedup": baseline_seconds / latent_total,
            "one_time_codec": one_time_codec_seconds,
            "per_generator_savings": per_generator_savings,
            "break_even_generator_runs": break_even_runs,
        },
        "generation_seconds": {
            "bpe_tokens_per_sample": args.new_tokens,
            "baseline_mean": baseline_latency,
            "latent_mean_including_decode": latent_latency,
            "speedup": baseline_latency / latent_latency,
            "repair_mean": repair_latency,
            "latent_decode_plus_repair_mean": (
                latent_latency + repair_latency if repair_latency is not None else None
            ),
        },
        "sample_quality": {
            "method": "blinded human comparison, 1-5 coherence and story quality",
            "status": "pending rating",
            "samples": str(out / "samples_blind.md"),
            "key": str(out / "sample_key.json"),
        },
        "checkpoints": {
            "autoencoder_best_step": ae_best["step"],
            "baseline_best_step": baseline_best["step"],
            "latent_best_step": latent_best["step"],
        },
    }
    (out / "report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    repair_markdown = (
        f"| LFM repair | {repair_latency:.3f}s; decode + repair {latent_latency + repair_latency:.3f}s |\n"
        if repair_latency is not None
        else ""
    )
    markdown = f"""# Lethe MVP results

| Measure | Result |
|---|---:|
| Reconstruction token accuracy | {reconstruction['token_accuracy']:.2%} |
| Reconstruction perplexity | {reconstruction['reconstruction_perplexity']:.3f} |
| Active latent codes | {reconstruction['active_codes']}/{CODEBOOK_SIZE} |
| Sequence compression | {raw_count / code_count:.2f}x |
| Latent LM training | {latent_lm_seconds:.1f}s ({baseline_seconds / latent_lm_seconds:.2f}x vs baseline) |
| Latent total (AE + encode + LM) | {latent_total:.1f}s ({baseline_seconds / latent_total:.2f}x vs baseline) |
| Baseline LM training | {baseline_seconds:.1f}s |
| Amortized break-even | {break_even_runs if break_even_runs is not None else 'never'} generator runs |
| Latent generation + decode | {latent_latency:.3f}s ({baseline_latency / latent_latency:.2f}x vs baseline) |
{repair_markdown}| Baseline generation | {baseline_latency:.3f}s |
| Final sample quality | Pending blinded rating in `samples_blind.md` |

Success is pending until the blinded stories are comparable and reusable generator training is materially faster.
"""
    (out / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)


@torch.no_grad()
def report_refined(args) -> None:
    device = device_from(args.device)
    raw_dir, base_dir, refined_dir, out = (
        Path(args.data_dir),
        Path(args.base_latent_dir),
        Path(args.refined_dir),
        Path(args.out_dir),
    )
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(raw_dir)
    codec, codec_best = load_ae(Path(args.autoencoder), device)
    refined_lm, refined_best = load_lm(Path(args.refined_model), device)
    baseline, baseline_best = load_lm(Path(args.baseline), device)
    raw_val = load_tokens(raw_dir / "val.bin")
    base_val = load_tokens(base_dir / "val.bin")
    serialized_val = load_tokens(refined_dir / "val.bin")
    refined_meta = json.loads((refined_dir / "meta.json").read_text(encoding="utf-8"))
    codec.config.refinement_threshold = refined_meta["refinement_threshold"]
    reconstruction = evaluate_refiner(
        codec, raw_val, base_val, args.eval_batches, args.eval_batch_size, device, decode=True
    )

    records = list(parse_refined_records(serialized_val, limit=max(35, args.samples * 17 + 1)))
    reconstruction_text = ["# Variable-rate reconstruction samples", ""]
    for block_index in range(1, 4):
        raw = np.asarray(raw_val[block_index * RAW_BLOCK : (block_index + 1) * RAW_BLOCK]).astype(np.int64)
        previous = np.asarray(raw_val[(block_index - 1) * RAW_BLOCK : block_index * RAW_BLOCK]).astype(np.int64)
        decoded = decode_refined_records(
            codec,
            [records[block_index]],
            torch.from_numpy(previous)[None].to(device),
            device,
        )[0].cpu().tolist()
        reconstruction_text += [
            f"## Sample {block_index}",
            "",
            f"Refinement codes: {int(records[block_index][1].sum())}",
            "",
            f"Original: {tokenizer.decode(raw.tolist(), skip_special_tokens=True)}",
            "",
            f"Reconstructed: {tokenizer.decode(decoded, skip_special_tokens=True)}",
            "",
        ]
    (out / "reconstructions.md").write_text("\n".join(reconstruction_text), encoding="utf-8")

    pairs, timings = [], {"baseline": [], "refined": []}
    for sample_index in range(args.samples):
        block_index = sample_index * 17
        raw_start = block_index * RAW_BLOCK
        raw_seed_np = np.asarray(raw_val[raw_start : raw_start + RAW_BLOCK]).astype(np.int64)
        raw_seed = torch.from_numpy(raw_seed_np)[None].to(device)
        seed_record = records[block_index][3]
        refined_seed = torch.from_numpy(seed_record)[None].to(device)
        prompt = tokenizer.decode(raw_seed_np.tolist(), skip_special_tokens=True)

        torch.manual_seed(1337 + sample_index)
        synchronize(device)
        started = time.perf_counter()
        with amp_context(device):
            baseline_ids = baseline.generate(raw_seed, args.new_tokens)
        baseline_text = tokenizer.decode(
            baseline_ids[0, RAW_BLOCK:].cpu().tolist(), skip_special_tokens=True
        )
        synchronize(device)
        timings["baseline"].append(time.perf_counter() - started)

        torch.manual_seed(1337 + sample_index)
        synchronize(device)
        started = time.perf_counter()
        with amp_context(device):
            generated = generate_refined_blocks(
                refined_lm, refined_seed, args.new_tokens // RAW_BLOCK
            )[0, len(seed_record) :]
            generated_records = list(parse_refined_records(generated.cpu().numpy()))
            decoded = decode_refined_records(codec, generated_records, raw_seed, device)
        refined_text = tokenizer.decode(decoded[0].cpu().tolist(), skip_special_tokens=True)
        synchronize(device)
        timings["refined"].append(time.perf_counter() - started)
        pairs.append({"prompt": prompt, "baseline": baseline_text, "refined": refined_text})

    rng = random.Random(1337)
    blind_text = [
        "# Blinded variable-rate sample comparison",
        "",
        "Rate each continuation 1-5 before opening `sample_key.json`.",
        "",
    ]
    key = {}
    for index, pair in enumerate(pairs, 1):
        swapped = bool(rng.getrandbits(1))
        a_name, b_name = (("refined", "baseline") if swapped else ("baseline", "refined"))
        key[str(index)] = {"A": a_name, "B": b_name}
        blind_text += [
            f"## Pair {index}",
            "",
            f"Prompt: {pair['prompt']}",
            "",
            f"A: {pair[a_name]}",
            "",
            f"B: {pair[b_name]}",
            "",
        ]
    (out / "samples_blind.md").write_text("\n".join(blind_text), encoding="utf-8")
    (out / "sample_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")

    codec_seconds = accounting_checkpoint(Path(args.autoencoder), device)["elapsed_seconds"]
    encoding_seconds = refined_meta["total_seconds"]
    refined_seconds = accounting_checkpoint(Path(args.refined_model), device)["elapsed_seconds"]
    baseline_seconds = accounting_checkpoint(Path(args.baseline), device)["elapsed_seconds"]
    one_time = codec_seconds + encoding_seconds
    total = one_time + refined_seconds
    savings = baseline_seconds - refined_seconds
    break_even = math.floor(one_time / savings) + 1 if savings > 0 else None
    baseline_latency = sum(timings["baseline"]) / len(timings["baseline"])
    refined_latency = sum(timings["refined"]) / len(timings["refined"])
    train_meta = refined_meta["train"]
    metrics = {
        "reconstruction": reconstruction,
        "compression": {
            "raw_bpe_tokens": train_meta["raw_tokens"],
            "serialized_tokens": train_meta["serialized_tokens"],
            "sequence_ratio": train_meta["sequence_compression"],
            "average_refinements": train_meta["average_refinements"],
        },
        "training_seconds": {
            "codec_total": codec_seconds,
            "encoding": encoding_seconds,
            "refined_model": refined_seconds,
            "baseline_model": baseline_seconds,
            "generator_speedup": baseline_seconds / refined_seconds,
            "first_run_total": total,
            "break_even_generator_runs": break_even,
        },
        "generation_seconds": {
            "baseline_mean": baseline_latency,
            "refined_mean_including_decode": refined_latency,
            "speedup": baseline_latency / refined_latency,
        },
        "checkpoints": {
            "codec_best_step": codec_best["step"],
            "refined_best_step": refined_best["step"],
            "baseline_best_step": baseline_best["step"],
        },
    }
    (out / "report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    markdown = f"""# Lethe variable-rate results

| Measure | Result |
|---|---:|
| Greedy reconstruction accuracy | {reconstruction['token_accuracy']:.2%} |
| Reconstruction perplexity | {reconstruction['reconstruction_perplexity']:.3f} |
| Average refinement codes/block | {train_meta['average_refinements']:.2f}/16 |
| Effective sequence compression | {train_meta['sequence_compression']:.2f}x |
| Refined generator training | {refined_seconds:.1f}s ({baseline_seconds / refined_seconds:.2f}x vs baseline) |
| Baseline generator training | {baseline_seconds:.1f}s |
| Amortized break-even | {break_even if break_even is not None else 'never'} generator runs |
| Refined generation + decode | {refined_latency:.3f}s ({baseline_latency / refined_latency:.2f}x vs baseline) |
| Baseline generation | {baseline_latency:.3f}s |

Sample quality remains a blinded human judgment in `samples_blind.md`.
"""
    (out / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)


@torch.no_grad()
def report_corrected(args) -> None:
    device = device_from(args.device)
    raw_dir, corrected_dir, out = Path(args.data_dir), Path(args.corrected_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(raw_dir)
    codec, codec_best = load_ae(Path(args.autoencoder), device)
    corrected_lm, corrected_best = load_lm(Path(args.corrected_model), device)
    baseline, baseline_best = load_lm(Path(args.baseline), device)
    raw_val, serialized_val = load_tokens(raw_dir / "val.bin"), load_tokens(corrected_dir / "val.bin")
    meta = json.loads((corrected_dir / "meta.json").read_text(encoding="utf-8"))
    slots = meta["correction_slots"]
    record_length = CODES_PER_BLOCK + 2 * slots
    frequencies = torch.from_numpy(
        np.bincount(load_tokens(raw_dir / "train.bin"), minlength=codec.config.vocab_size).astype(np.int64)
    ).to(device)
    reconstruction = evaluate_corrected(
        codec, raw_val, serialized_val, frequencies, slots,
        args.eval_batches, args.eval_batch_size, args.rare_cutoff, device,
    )
    records_np = np.asarray(serialized_val).reshape(-1, record_length).astype(np.int64)
    reconstruction_text = ["# Corrected reconstruction samples", ""]
    for block_index in range(1, 4):
        history = torch.from_numpy(
            np.asarray(raw_val[(block_index - 1) * RAW_BLOCK : block_index * RAW_BLOCK]).astype(np.int64)
        )[None].to(device)
        record = torch.from_numpy(records_np[block_index : block_index + 1]).to(device)
        decoded = decode_corrected_records(codec, record, history, slots)[0].cpu().tolist()
        original = np.asarray(raw_val[block_index * RAW_BLOCK : (block_index + 1) * RAW_BLOCK]).astype(np.int64)
        reconstruction_text += [
            f"## Sample {block_index}", "",
            f"Original: {tokenizer.decode(original.tolist(), skip_special_tokens=True)}", "",
            f"Reconstructed: {tokenizer.decode(decoded, skip_special_tokens=True)}", "",
        ]
    (out / "reconstructions.md").write_text("\n".join(reconstruction_text), encoding="utf-8")

    if args.new_tokens % RAW_BLOCK:
        raise ValueError(f"--new-tokens must be divisible by {RAW_BLOCK}")
    pairs, timings = [], {"baseline": [], "corrected": []}
    for sample_index in range(args.samples):
        block_index = sample_index * 17
        raw_seed_np = np.asarray(raw_val[block_index * RAW_BLOCK : (block_index + 1) * RAW_BLOCK]).astype(np.int64)
        raw_seed = torch.from_numpy(raw_seed_np)[None].to(device)
        record_seed = torch.from_numpy(records_np[block_index : block_index + 1]).to(device)
        prompt = tokenizer.decode(raw_seed_np.tolist(), skip_special_tokens=True)

        torch.manual_seed(1337 + sample_index)
        synchronize(device)
        started = time.perf_counter()
        with amp_context(device):
            baseline_ids = baseline.generate(raw_seed, args.new_tokens)
        synchronize(device)
        timings["baseline"].append(time.perf_counter() - started)
        baseline_text = tokenizer.decode(
            baseline_ids[0, RAW_BLOCK:].cpu().tolist(), skip_special_tokens=True
        )

        torch.manual_seed(1337 + sample_index)
        synchronize(device)
        started = time.perf_counter()
        with amp_context(device):
            generated = generate_corrected_blocks(
                corrected_lm, record_seed, args.new_tokens // RAW_BLOCK, slots, codec.config.vocab_size
            )[:, record_length:]
            generated_records = generated.view(-1, record_length)
            decoded = decode_corrected_records(codec, generated_records, raw_seed, slots)
        synchronize(device)
        timings["corrected"].append(time.perf_counter() - started)
        corrected_text = tokenizer.decode(decoded[0].cpu().tolist(), skip_special_tokens=True)
        pairs.append({"prompt": prompt, "baseline": baseline_text, "corrected": corrected_text})

    rng = random.Random(1337)
    blind = ["# Blinded corrected-code sample comparison", "", "Rate each continuation 1-5 before opening `sample_key.json`.", ""]
    key = {}
    for index, pair in enumerate(pairs, 1):
        swapped = bool(rng.getrandbits(1))
        a, b = (("corrected", "baseline") if swapped else ("baseline", "corrected"))
        key[str(index)] = {"A": a, "B": b}
        blind += [f"## Pair {index}", "", f"Prompt: {pair['prompt']}", "", f"A: {pair[a]}", "", f"B: {pair[b]}", ""]
    (out / "samples_blind.md").write_text("\n".join(blind), encoding="utf-8")
    (out / "sample_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")

    codec_seconds = accounting_checkpoint(Path(args.autoencoder), device)["elapsed_seconds"]
    encoding_seconds = meta["total_seconds"]
    corrected_seconds = accounting_checkpoint(Path(args.corrected_model), device)["elapsed_seconds"]
    baseline_seconds = accounting_checkpoint(Path(args.baseline), device)["elapsed_seconds"]
    savings = baseline_seconds - corrected_seconds
    full_break_even = math.floor((codec_seconds + encoding_seconds) / savings) + 1 if savings > 0 else None
    incremental_break_even = math.floor(encoding_seconds / savings) + 1 if savings > 0 else None
    baseline_latency = sum(timings["baseline"]) / len(timings["baseline"])
    corrected_latency = sum(timings["corrected"]) / len(timings["corrected"])
    metrics = {
        "reconstruction": reconstruction,
        "compression": {
            "sequence_ratio": meta["train"]["sequence_compression"],
            "correction_slots": slots,
            "serialized_tokens": meta["train"]["serialized_tokens"],
        },
        "training_seconds": {
            "codec": codec_seconds,
            "encoding": encoding_seconds,
            "corrected_model": corrected_seconds,
            "baseline_model": baseline_seconds,
            "generator_speedup": baseline_seconds / corrected_seconds,
            "full_break_even_runs": full_break_even,
            "incremental_break_even_runs_reusing_v3": incremental_break_even,
        },
        "generation_seconds": {
            "baseline_mean": baseline_latency,
            "corrected_mean_including_decode": corrected_latency,
            "speedup": baseline_latency / corrected_latency,
        },
        "checkpoints": {
            "codec_step": codec_best["step"],
            "corrected_step": corrected_best["step"],
            "baseline_step": baseline_best["step"],
        },
    }
    (out / "report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    markdown = f"""# Fixed correction-code results

| Measure | Result |
|---|---:|
| Greedy reconstruction accuracy | {reconstruction['token_accuracy']:.2%} |
| Rare-token accuracy (frequency <= {args.rare_cutoff:,}) | {reconstruction['rare_token_accuracy']:.2%} |
| Sequence compression | {meta['train']['sequence_compression']:.2f}x |
| Correction slots/block | {slots} |
| Corrected generator training | {corrected_seconds:.1f}s ({baseline_seconds / corrected_seconds:.2f}x vs baseline) |
| Baseline generator training | {baseline_seconds:.1f}s |
| Break-even reusing existing v3 codec | {incremental_break_even if incremental_break_even is not None else 'never'} generator runs |
| Corrected generation + decode | {corrected_latency:.3f}s ({baseline_latency / corrected_latency:.2f}x vs baseline) |
| Baseline generation | {baseline_latency:.3f}s |

Sample quality remains a blinded human judgment in `samples_blind.md`.
"""
    (out / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)


def smoke(_args) -> None:
    torch.manual_seed(1)
    literal_tail = pack_literal_tail(np.asarray([3, 63]), np.asarray([8191, 42]))
    literal_positions, literal_tokens = unpack_literal_tail(literal_tail)
    assert literal_positions.tolist() == [3, 63] and literal_tokens.tolist() == [8191, 42]
    correction = pack_correction(63, 8191)
    assert unpack_correction(*correction) == (63, 8191)
    ae = VQAutoencoder(
        AEConfig(
            128,
            n_embd=48,
            n_layer=1,
            n_head=3,
            code_dim=5,
            codebook_size=32,
            decoder_n_embd=48,
            decoder_n_layer=1,
            decoder_n_head=3,
            refinement=True,
        )
    )
    tokens = torch.randint(0, 128, (2, RAW_BLOCK))
    logits, codes, reconstruction, vq = ae(tokens)
    assert logits.shape == (2, RAW_BLOCK, 128)
    assert codes.shape == (2, CODES_PER_BLOCK)
    refinement, refinement_codes, refinement_loss = ae.encode_refinement_vectors(tokens)
    mask = torch.zeros((2, CODES_PER_BLOCK), dtype=torch.bool)
    mask[:, ::4] = True
    refined_logits = ae._decode_teacher(
        ae.codes_to_vectors(codes), tokens, None, 0.0, refinement, mask
    )
    (reconstruction + vq + refinement_loss + F.cross_entropy(refined_logits.flatten(0, 1), tokens.flatten())).backward()
    ae.eval()
    ae.config.decoder_context = RAW_BLOCK
    assert ae.decoder_logits(codes, tokens, tokens).shape == logits.shape
    assert ae.reconstruct(codes, tokens).shape == tokens.shape
    assert ae.reconstruct(codes, tokens, refinement_codes, mask).shape == tokens.shape
    assert ae.reconstruct_sequence(torch.cat((codes, codes), dim=1), tokens).shape == (2, 2 * RAW_BLOCK)
    assert int(ae.ema_initialized) == 1

    lm = Transformer(ModelConfig(128, 16, 1, 3, 48))
    x = torch.randint(0, 128, (2, 16))
    lm_logits, loss = lm(x, x)
    assert lm_logits.shape == (2, 16, 128) and torch.isfinite(loss)
    assert lm.generate(x[:1, :4], 2).shape == (1, 6)

    baseline = Transformer(lm_config("baseline", 8192)).num_parameters()
    latent = Transformer(lm_config("latent", CODEBOOK_SIZE)).num_parameters()
    latent40 = Transformer(lm_config("latent40", CODEBOOK_SIZE)).num_parameters()
    assert 9_000_000 <= baseline <= 11_000_000
    assert 9_000_000 <= latent <= 11_000_000
    assert 38_000_000 <= latent40 <= 42_000_000
    print(
        json.dumps(
            {
                "status": "ok",
                "compression": f"{RAW_BLOCK} BPE -> {CODES_PER_BLOCK} codes",
                "baseline_parameters": baseline,
                "latent_parameters": latent,
                "latent40_parameters": latent40,
            },
            indent=2,
        )
    )


def add_device(parser) -> None:
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))


def main() -> None:
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("prepare", help="stream TinyStories, train BPE, and write 50M tokens")
    command.add_argument("--dataset", default="roneneldan/TinyStories")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--vocab-size", type=int, default=8192)
    command.add_argument("--tokenizer-docs", type=int, default=100_000)
    command.add_argument("--train-tokens", type=int, default=50_000_000)
    command.add_argument("--val-tokens", type=int, default=1_000_000)
    command.set_defaults(function=prepare)

    command = commands.add_parser(
        "prepare-semantic-ir", help="extract a compact deterministic story-event representation"
    )
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--out-dir", default="data/semantic-ir")
    command.add_argument("--train-tokens", type=int, default=1_000_000)
    command.add_argument("--val-tokens", type=int, default=200_000)
    command.add_argument("--max-ir-tokens", type=int, default=56)
    command.set_defaults(function=prepare_semantic_ir)

    command = commands.add_parser(
        "train-semantic-realizer", help="train a 10M model to reconstruct text from semantic IR"
    )
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--ir-dir", default="data/semantic-ir")
    command.add_argument("--out", default="checkpoints/semantic-ir-realizer")
    command.add_argument("--steps", type=int, default=2000)
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument("--lr", type=float, default=3e-4)
    command.add_argument("--eval-every", type=int, default=250)
    command.add_argument("--eval-batches", type=int, default=20)
    command.add_argument("--save-every", type=int, default=250)
    add_device(command)
    command.set_defaults(function=train_semantic_realizer)

    command = commands.add_parser(
        "report-semantic-realizer", help="measure semantic-IR reconstruction and write samples"
    )
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--ir-dir", default="data/semantic-ir")
    command.add_argument("--checkpoint", default="checkpoints/semantic-ir-realizer/best.pt")
    command.add_argument("--out-dir", default="results/semantic-ir-pilot")
    command.add_argument("--eval-batches", type=int, default=20)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--samples", type=int, default=32)
    command.add_argument("--seed", type=int, default=1337)
    add_device(command)
    command.set_defaults(function=report_semantic_realizer)

    command = commands.add_parser("train-ae", help="train the 64-token to 16-code VQ autoencoder")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--out", default="checkpoints/autoencoder-v2")
    command.add_argument("--steps", type=int, default=0, help="0 means one pass over the training tokens")
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--lr", type=float, default=3e-4)
    command.add_argument("--eval-every", type=int, default=500)
    command.add_argument("--eval-batches", type=int, default=20)
    command.add_argument("--save-every", type=int, default=500)
    add_device(command)
    command.set_defaults(function=train_ae)

    command = commands.add_parser(
        "train-decoder", help="freeze existing codes and train a previous-block-aware decoder"
    )
    command.add_argument("--autoencoder", default="checkpoints/autoencoder-v2/best.pt")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--latent-dir", default="data/latent-v2")
    command.add_argument("--out", default="checkpoints/autoencoder-v3-context-rope")
    command.add_argument("--steps", type=int, default=0, help="0 means one pass over aligned blocks")
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument("--context", type=int, default=64)
    command.add_argument("--lr", type=float, default=2e-4)
    command.add_argument("--start-dropout", type=float, default=0.75)
    command.add_argument("--end-dropout", type=float, default=0.25)
    command.add_argument("--eval-every", type=int, default=500)
    command.add_argument("--eval-batches", type=int, default=20)
    command.add_argument("--save-every", type=int, default=500)
    add_device(command)
    command.set_defaults(function=train_decoder)

    command = commands.add_parser("train-joint-codec", help="jointly fine-tune the fixed-rate contextual codec")
    command.add_argument("--autoencoder", default="checkpoints/autoencoder-v3-context-rope/best.pt")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--latent-dir", default="data/latent-v2")
    command.add_argument("--out", default="checkpoints/autoencoder-v3.1-joint")
    command.add_argument("--steps", type=int, default=0)
    command.add_argument("--epochs", type=int, default=2)
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument("--lr", type=float, default=5e-5)
    command.add_argument("--rare-cutoff", type=int, default=1000)
    command.add_argument("--rare-weight", type=float, default=3.0)
    command.add_argument("--self-condition", type=float, default=0.25)
    command.add_argument("--dropout", type=float, default=0.10)
    command.add_argument("--eval-every", type=int, default=500)
    command.add_argument("--eval-batches", type=int, default=20)
    command.add_argument("--save-every", type=int, default=500)
    add_device(command)
    command.set_defaults(
        function=train_joint_codec,
        priority=False,
        priority_weight=3.0,
        hard_rate=0.0,
        hard_weight=1.0,
        aux_weight=0.0,
    )

    command = commands.add_parser(
        "train-priority-codec", help="jointly prioritize generally rare and surprising occurrences"
    )
    command.add_argument("--autoencoder", default="checkpoints/autoencoder-v3-context-rope/best.pt")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--latent-dir", default="data/latent-v2")
    command.add_argument("--out", default="checkpoints/autoencoder-v3.2-priority")
    command.add_argument("--steps", type=int, default=0)
    command.add_argument("--epochs", type=int, default=1)
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument("--lr", type=float, default=5e-5)
    command.add_argument("--rare-cutoff", type=int, default=1000)
    command.add_argument("--rare-weight", type=float, default=3.0)
    command.add_argument("--priority-weight", type=float, default=6.0)
    command.add_argument("--hard-rate", type=float, default=0.20)
    command.add_argument("--hard-weight", type=float, default=2.0)
    command.add_argument("--aux-weight", type=float, default=0.25)
    command.add_argument("--self-condition", type=float, default=0.0)
    command.add_argument("--dropout", type=float, default=0.25)
    command.add_argument("--eval-every", type=int, default=500)
    command.add_argument("--eval-batches", type=int, default=20)
    command.add_argument("--save-every", type=int, default=500)
    add_device(command)
    command.set_defaults(function=train_joint_codec, priority=True)

    command = commands.add_parser("rare-diagnostic", help="measure greedy accuracy by token frequency")
    command.add_argument("--autoencoder", default="checkpoints/autoencoder-v3-context-rope/best.pt")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--latent-dir", default="data/latent-v2")
    command.add_argument("--out", default="results/v3.2-priority/rare-v3.json")
    command.add_argument("--blocks", type=int, default=200)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--cutoffs", type=int, nargs="+", default=(100, 1000, 10000))
    add_device(command)
    command.set_defaults(function=rare_diagnostic)

    command = commands.add_parser(
        "near-lossless-diagnostic", help="sweep exact residuals needed for near-lossless v3"
    )
    command.add_argument("--autoencoder", default="checkpoints/autoencoder-v3-context-rope-2pass/best.pt")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--latent-dir", default="data/latent-v2")
    command.add_argument("--out", default="results/v3-near-lossless/diagnostic.json")
    command.add_argument("--blocks", type=int, default=1600)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--targets", type=float, nargs="+", default=(0.80, 0.90, 0.95, 0.99, 1.0))
    add_device(command)
    command.set_defaults(function=near_lossless_diagnostic)

    command = commands.add_parser(
        "dual-codec-diagnostic", help="merge plain and rare-priority v3 decoder outputs"
    )
    command.add_argument("--base-autoencoder", default="checkpoints/autoencoder-v3-context-rope/best.pt")
    command.add_argument("--priority-autoencoder", default="checkpoints/autoencoder-v3.2-priority/best.pt")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--base-latent-dir", default="data/latent-v2")
    command.add_argument("--priority-latent-dir", default="data/latent-v3.2-priority")
    command.add_argument("--out", default="results/v3-dual-codec/diagnostic.json")
    command.add_argument("--blocks", type=int, default=1600)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--rare-cutoff", type=int, default=1000)
    add_device(command)
    command.set_defaults(function=dual_codec_diagnostic)

    command = commands.add_parser(
        "correction-diagnostic", help="test fixed packed corrections for rare and surprising tokens"
    )
    command.add_argument("--autoencoder", default="checkpoints/autoencoder-v3-context-rope/best.pt")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--latent-dir", default="data/latent-v2")
    command.add_argument("--out", default="results/v3.3-corrections/diagnostic.json")
    command.add_argument("--blocks", type=int, default=1600)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--slots", type=int, nargs="+", default=(1, 2))
    command.add_argument("--rare-cutoff", type=int, default=1000)
    command.add_argument("--surprisal-weight", type=float, default=0.5)
    add_device(command)
    command.set_defaults(function=correction_diagnostic)

    command = commands.add_parser("beam-diagnostic", help="compare greedy and beam codec reconstruction")
    command.add_argument("--autoencoder", default="checkpoints/autoencoder-v3.1-joint/best.pt")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--latent-dir", default="data/latent-v3.1")
    command.add_argument("--out", default="results/v3.1-joint/beam.json")
    command.add_argument("--blocks", type=int, default=10)
    command.add_argument("--width", type=int, default=4)
    add_device(command)
    command.set_defaults(function=beam_diagnostic)

    command = commands.add_parser(
        "train-refiner", help="add optional codes for high-surprisal four-token groups"
    )
    command.add_argument("--autoencoder", default="checkpoints/autoencoder-v3-context-rope/best.pt")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--latent-dir", default="data/latent-v2")
    command.add_argument("--out", default="checkpoints/autoencoder-v4-refined")
    command.add_argument("--steps", type=int, default=0, help="0 means one pass over aligned blocks")
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument("--rate", type=float, default=0.25)
    command.add_argument("--threshold", type=float, default=0.0)
    command.add_argument("--calibration-batches", type=int, default=50)
    command.add_argument("--dropout", type=float, default=0.25)
    command.add_argument("--mask-weight", type=float, default=2.0)
    command.add_argument("--lr", type=float, default=2e-4)
    command.add_argument("--eval-every", type=int, default=500)
    command.add_argument("--eval-batches", type=int, default=20)
    command.add_argument("--save-every", type=int, default=500)
    add_device(command)
    command.set_defaults(function=train_refiner)

    command = commands.add_parser("encode", help="freeze the autoencoder and convert BPE data to latent codes")
    command.add_argument("--checkpoint", default="checkpoints/autoencoder-v2/best.pt")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--out-dir", default="data/latent-v2")
    command.add_argument("--batch-size", type=int, default=256)
    add_device(command)
    command.set_defaults(function=encode_dataset)

    command = commands.add_parser("encode-corrected", help="append fixed packed rare-token corrections")
    command.add_argument("--autoencoder", default="checkpoints/autoencoder-v3-context-rope/best.pt")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--base-latent-dir", default="data/latent-v2")
    command.add_argument("--out-dir", default="data/latent-v3.3-corrected")
    command.add_argument("--slots", type=int, default=2)
    command.add_argument("--surprisal-weight", type=float, default=0.5)
    command.add_argument("--batch-size", type=int, default=256)
    add_device(command)
    command.set_defaults(function=encode_corrected)

    command = commands.add_parser("encode-refined", help="write variable-rate base, mask, and refinement records")
    command.add_argument("--checkpoint", default="checkpoints/autoencoder-v4-refined/best.pt")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--base-latent-dir", default="data/latent-v2")
    command.add_argument("--out-dir", default="data/latent-v4-refined")
    command.add_argument("--batch-size", type=int, default=256)
    command.add_argument("--calibration-batches", type=int, default=50)
    command.add_argument("--rate", type=float, default=0.25)
    add_device(command)
    command.set_defaults(function=encode_refined)

    command = commands.add_parser("sweep-refinement", help="measure reconstruction across extra-code rates")
    command.add_argument("--checkpoint", default="checkpoints/autoencoder-v4-refined/best.pt")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--base-latent-dir", default="data/latent-v2")
    command.add_argument("--out-dir", default="results/v4-rate-sweep")
    command.add_argument("--extras", type=int, nargs="+", default=(1, 2, 3, 4))
    command.add_argument("--calibration-batches", type=int, default=50)
    command.add_argument("--eval-batches", type=int, default=50)
    command.add_argument("--batch-size", type=int, default=32)
    add_device(command)
    command.set_defaults(function=sweep_refinement)

    command = commands.add_parser("sweep-literals", help="test exact rare-token escape tails")
    command.add_argument("--checkpoint", default="checkpoints/autoencoder-v4-refined/best.pt")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--base-latent-dir", default="data/latent-v2")
    command.add_argument("--out-dir", default="results/v4-literal-sweep")
    command.add_argument("--rate", type=float, default=0.125)
    command.add_argument("--cutoffs", type=int, nargs="+", default=(100, 300, 1000, 3000))
    command.add_argument("--max-literals", type=int, default=4)
    command.add_argument("--calibration-batches", type=int, default=50)
    command.add_argument("--eval-batches", type=int, default=50)
    command.add_argument("--batch-size", type=int, default=32)
    add_device(command)
    command.set_defaults(function=sweep_literals)

    command = commands.add_parser("train-lm", help="train one equal-text pass of a 10M LM")
    command.add_argument("kind", choices=("latent", "latent40", "refined", "corrected", "baseline"))
    command.add_argument("--data-dir", default="")
    command.add_argument("--out", default="")
    command.add_argument("--steps", type=int, default=0, help="0 means one pass over the dataset")
    command.add_argument("--batch-size", type=int, default=0, help="0 means 64 latent or 16 baseline")
    command.add_argument("--grad-accum", type=int, default=1)
    command.add_argument("--lr", type=float, default=0, help="0 means 6e-4 latent or 3e-4 baseline")
    command.add_argument("--eval-every", type=int, default=500)
    command.add_argument("--eval-batches", type=int, default=20)
    command.add_argument("--save-every", type=int, default=500)
    add_device(command)
    command.set_defaults(function=train_lm)

    command = commands.add_parser("train-correction-head", help="train structured correction outputs over frozen v3 states")
    command.add_argument("--base-model", default="checkpoints/latent-v2-b64/best.pt")
    command.add_argument("--corrected-dir", default="data/latent-v3.3-corrected")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--out", default="checkpoints/correction-head-v3.4")
    command.add_argument("--steps", type=int, default=0)
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument("--bottleneck", type=int, default=64)
    command.add_argument("--rare-cutoff", type=int, default=1000)
    command.add_argument("--lr", type=float, default=3e-4)
    command.add_argument("--apply-weight", type=float, default=1.0)
    command.add_argument("--position-weight", type=float, default=1.0)
    command.add_argument("--token-weight", type=float, default=1.0)
    command.add_argument("--code-weight", type=float, default=1.0)
    command.add_argument("--joint", action="store_true")
    command.add_argument("--eval-every", type=int, default=500)
    command.add_argument("--eval-batches", type=int, default=20)
    command.add_argument("--save-every", type=int, default=500)
    add_device(command)
    command.set_defaults(function=train_correction_head)

    command = commands.add_parser("structured-correction-diagnostic", help="apply predicted correction heads during reconstruction")
    command.add_argument("--autoencoder", default="checkpoints/autoencoder-v3-context-rope/best.pt")
    command.add_argument("--base-model", default="checkpoints/latent-v2-b64/best.pt")
    command.add_argument("--correction-head", default="checkpoints/correction-head-v3.4/best.pt")
    command.add_argument("--corrected-dir", default="data/latent-v3.3-corrected")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--out", default="results/v3.4-structured-head/diagnostic.json")
    command.add_argument("--blocks", type=int, default=1600)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--rare-cutoff", type=int, default=1000)
    command.add_argument("--threshold", type=float, default=0.5)
    add_device(command)
    command.set_defaults(function=structured_correction_diagnostic)

    command = commands.add_parser("report", help="write the six requested metrics and blinded samples")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--latent-dir", default="data/latent-v2")
    command.add_argument("--autoencoder", default="checkpoints/autoencoder-v3-context-rope/best.pt")
    command.add_argument("--baseline", default="checkpoints/baseline/best.pt")
    command.add_argument("--latent", default="checkpoints/latent-v2-b64/best.pt")
    command.add_argument("--out-dir", default="results/v3-context")
    command.add_argument("--samples", type=int, default=3)
    command.add_argument("--new-tokens", type=int, default=256)
    command.add_argument("--eval-batches", type=int, default=50)
    command.add_argument("--eval-batch-size", type=int, default=32)
    command.add_argument("--repair-url", default="http://127.0.0.1:1234/v1")
    command.add_argument("--repair-model", default="", help="LM Studio model ID; empty disables repair")
    command.add_argument("--repair-timeout", type=float, default=120.0)
    add_device(command)
    command.set_defaults(function=report)

    command = commands.add_parser("report-refined", help="compare variable-rate generation with the baseline")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--base-latent-dir", default="data/latent-v2")
    command.add_argument("--refined-dir", default="data/latent-v4-refined")
    command.add_argument("--autoencoder", default="checkpoints/autoencoder-v4-refined/best.pt")
    command.add_argument("--refined-model", default="checkpoints/latent-v4-refined/best.pt")
    command.add_argument("--baseline", default="checkpoints/baseline/best.pt")
    command.add_argument("--out-dir", default="results/v4-refined")
    command.add_argument("--samples", type=int, default=3)
    command.add_argument("--new-tokens", type=int, default=256)
    command.add_argument("--eval-batches", type=int, default=50)
    command.add_argument("--eval-batch-size", type=int, default=32)
    add_device(command)
    command.set_defaults(function=report_refined)

    command = commands.add_parser("report-corrected", help="compare fixed correction codes with the baseline")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--corrected-dir", default="data/latent-v3.3-corrected")
    command.add_argument("--autoencoder", default="checkpoints/autoencoder-v3-context-rope/best.pt")
    command.add_argument("--corrected-model", default="checkpoints/latent-v3.3-corrected/best.pt")
    command.add_argument("--baseline", default="checkpoints/baseline/best.pt")
    command.add_argument("--out-dir", default="results/v3.3-corrections")
    command.add_argument("--samples", type=int, default=3)
    command.add_argument("--new-tokens", type=int, default=256)
    command.add_argument("--eval-batches", type=int, default=50)
    command.add_argument("--eval-batch-size", type=int, default=32)
    command.add_argument("--rare-cutoff", type=int, default=1000)
    add_device(command)
    command.set_defaults(function=report_corrected)

    command = commands.add_parser("smoke", help="run the smallest shape, gradient, and parameter-count checks")
    command.set_defaults(function=smoke)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
