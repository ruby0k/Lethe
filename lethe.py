"""One-file MVP for testing whether 4x VQ compression reduces total LM compute."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
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
    code_dim: int = 96
    codebook_size: int = CODEBOOK_SIZE
    commitment: float = 0.25


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    _, _, length, dim = x.shape
    half = dim // 2
    frequencies = 1.0 / (10000 ** (torch.arange(half, device=x.device).float() / half))
    angles = torch.outer(torch.arange(length, device=x.device).float(), frequencies).to(x.dtype)
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

    def forward(self, x: torch.Tensor, causal: bool) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        shape = (batch, length, self.n_head, width // self.n_head)
        q, k, v = (part.view(shape).transpose(1, 2) for part in (q, k, v))
        attention = F.scaled_dot_product_attention(apply_rope(q), apply_rope(k), v, is_causal=causal)
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

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        if tokens.shape[1] > self.config.block_size:
            raise ValueError(f"sequence length {tokens.shape[1]} exceeds {self.config.block_size}")
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x, causal=True)
        logits = self.head(self.norm(x))
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


class VQAutoencoder(nn.Module):
    def __init__(self, config: AEConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.encoder = nn.ModuleList(Block(config.n_embd, config.n_head) for _ in range(config.n_layer))
        self.to_latent = nn.Linear(COMPRESSION * config.n_embd, config.code_dim)
        self.codebook = nn.Embedding(config.codebook_size, config.code_dim)
        self.from_latent = nn.Linear(config.code_dim, COMPRESSION * config.n_embd)
        self.decoder = nn.ModuleList(Block(config.n_embd, config.n_head) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = self.embedding.weight
        self.apply(init_weights)

    def encode_vectors(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] != RAW_BLOCK:
            raise ValueError(f"autoencoder expects exactly {RAW_BLOCK} tokens")
        x = self.embedding(tokens)
        for block in self.encoder:
            x = block(x, causal=False)
        return self.to_latent(x.reshape(x.shape[0], CODES_PER_BLOCK, -1))

    def quantize(self, encoded: torch.Tensor):
        flat = encoded.float().flatten(0, 1)
        codebook = self.codebook.weight.float()
        distances = flat.pow(2).sum(1, keepdim=True) + codebook.pow(2).sum(1) - 2 * flat @ codebook.T
        ids = distances.argmin(1).view(encoded.shape[:2])
        quantized = self.codebook(ids)
        codebook_loss = F.mse_loss(quantized, encoded.detach())
        commitment_loss = F.mse_loss(encoded, quantized.detach())
        straight_through = encoded + (quantized - encoded).detach()
        return straight_through, ids, codebook_loss + self.config.commitment * commitment_loss

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.shape[1] != CODES_PER_BLOCK:
            raise ValueError(f"decoder expects exactly {CODES_PER_BLOCK} codes")
        x = self.from_latent(self.codebook(codes)).reshape(codes.shape[0], RAW_BLOCK, self.config.n_embd)
        for block in self.decoder:
            x = block(x, causal=False)
        return self.head(self.norm(x))

    def forward(self, tokens: torch.Tensor):
        encoded = self.encode_vectors(tokens)
        quantized, codes, vq_loss = self.quantize(encoded)
        x = self.from_latent(quantized).reshape(tokens.shape[0], RAW_BLOCK, self.config.n_embd)
        for block in self.decoder:
            x = block(x, causal=False)
        logits = self.head(self.norm(x))
        reconstruction_loss = F.cross_entropy(logits.flatten(0, 1), tokens.flatten())
        return logits, codes, reconstruction_loss, vq_loss

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


def load_tokenizer(data_dir: Path):
    from tokenizers import Tokenizer

    path = data_dir / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run prepare first")
    return Tokenizer.from_file(str(path))


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
def evaluate_ae(model: VQAutoencoder, data: np.memmap, batches: int, batch_size: int, device: str):
    model.eval()
    reconstruction = accuracy = total = 0.0
    usage = torch.zeros(model.config.codebook_size, device=device)
    for _ in range(batches):
        tokens = token_batch(data, RAW_BLOCK, batch_size, device)
        with amp_context(device):
            logits, codes, reconstruction_loss, _ = model(tokens)
        reconstruction += float(reconstruction_loss)
        accuracy += float((logits.argmax(-1) == tokens).float().mean())
        usage += torch.bincount(codes.flatten(), minlength=model.config.codebook_size)
        total += 1
    probabilities = usage / usage.sum()
    active = probabilities > 0
    code_perplexity = torch.exp(-(probabilities[active] * probabilities[active].log()).sum())
    model.train()
    mean_loss = reconstruction / total
    return {
        "reconstruction_loss": mean_loss,
        "reconstruction_perplexity": math.exp(min(20, mean_loss)),
        "token_accuracy": accuracy / total,
        "active_codes": int(active.sum()),
        "code_perplexity": float(code_perplexity),
    }


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


def load_ae(path: Path, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = VQAutoencoder(AEConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
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
def evaluate_lm(model: Transformer, data: np.memmap, batches: int, batch_size: int, device: str) -> float:
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = lm_batch(data, model.config.block_size, batch_size, device)
        with amp_context(device):
            _, loss = model(x, y)
        total += float(loss)
    model.train()
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
        "equivalent_bpe_tokens": units * (COMPRESSION if kind == "latent" else 1),
        "model_config": asdict(model.config),
    }


def train_lm(args) -> None:
    device = device_from(args.device)
    data_dir = Path(args.data_dir or ("data/tinystories" if args.kind == "baseline" else "data/latent"))
    out = Path(args.out or f"checkpoints/{args.kind}")
    metadata = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    vocab_size = metadata["vocab_size"] if args.kind == "baseline" else metadata["codebook_size"]
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
                * (COMPRESSION if args.kind == "latent" else 1),
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


def accounting_checkpoint(path: Path, device: str) -> dict:
    latest = path.with_name("latest.pt")
    return torch.load(latest if latest.exists() else path, map_location=device, weights_only=False)


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

    reconstruction = evaluate_ae(ae, raw_val, args.eval_batches, args.eval_batch_size, device)
    examples = []
    for block_index in range(3):
        raw = np.asarray(raw_val[block_index * RAW_BLOCK : (block_index + 1) * RAW_BLOCK]).astype(np.int64)
        tokens = torch.from_numpy(raw)[None].to(device)
        with amp_context(device):
            logits, _, _, _ = ae(tokens)
        examples.append(
            {
                "original": tokenizer.decode(raw.tolist(), skip_special_tokens=True),
                "reconstructed": tokenizer.decode(logits.argmax(-1)[0].tolist(), skip_special_tokens=True),
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
        ae.decode_codes(warm_codes)
    synchronize(device)

    pairs, timings = [], {"baseline": [], "latent": []}
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
            latent_ids = latent.generate(code_seed, new_codes)[0, CODES_PER_BLOCK:]
            code_blocks = latent_ids.view(-1, CODES_PER_BLOCK)
            decoded_ids = ae.decode_codes(code_blocks).argmax(-1).flatten()
        latent_text = tokenizer.decode(decoded_ids.cpu().tolist(), skip_special_tokens=True)
        synchronize(device)
        timings["latent"].append(time.perf_counter() - started)
        pairs.append({"prompt": prompt, "baseline": baseline_text, "latent": latent_text})

    rng = random.Random(1337)
    blind_text = [
        "# Blinded sample comparison",
        "",
        "Rate each continuation 1-5 for coherence and story quality before opening `sample_key.json`.",
        "",
    ]
    key = {}
    for index, pair in enumerate(pairs, 1):
        swapped = bool(rng.getrandbits(1))
        a_name, b_name = (("latent", "baseline") if swapped else ("baseline", "latent"))
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

    baseline_seconds = baseline_accounting["elapsed_seconds"]
    latent_lm_seconds = latent_accounting["elapsed_seconds"]
    ae_seconds = ae_accounting["elapsed_seconds"]
    encoding_seconds = latent_meta["total_seconds"]
    latent_total = ae_seconds + encoding_seconds + latent_lm_seconds
    baseline_latency = sum(timings["baseline"]) / len(timings["baseline"])
    latent_latency = sum(timings["latent"]) / len(timings["latent"])
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
        },
        "generation_seconds": {
            "bpe_tokens_per_sample": args.new_tokens,
            "baseline_mean": baseline_latency,
            "latent_mean_including_decode": latent_latency,
            "speedup": baseline_latency / latent_latency,
        },
        "sample_quality": {
            "method": "blinded human A/B, 1-5 coherence and story quality",
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
    markdown = f"""# Lethe MVP results

| Measure | Result |
|---|---:|
| Reconstruction token accuracy | {reconstruction['token_accuracy']:.2%} |
| Reconstruction perplexity | {reconstruction['reconstruction_perplexity']:.3f} |
| Active VQ codes | {reconstruction['active_codes']}/{CODEBOOK_SIZE} |
| Sequence compression | {raw_count / code_count:.2f}x |
| Latent LM training | {latent_lm_seconds:.1f}s ({baseline_seconds / latent_lm_seconds:.2f}x vs baseline) |
| Latent total (AE + encode + LM) | {latent_total:.1f}s ({baseline_seconds / latent_total:.2f}x vs baseline) |
| Baseline LM training | {baseline_seconds:.1f}s |
| Latent generation + decode | {latent_latency:.3f}s ({baseline_latency / latent_latency:.2f}x vs baseline) |
| Baseline generation | {baseline_latency:.3f}s |
| Final sample quality | Pending blinded rating in `samples_blind.md` |

Success is pending until the blinded stories are comparable and total speedup is materially above 1x.
"""
    (out / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)


def smoke(_args) -> None:
    torch.manual_seed(1)
    ae = VQAutoencoder(AEConfig(128, n_embd=48, n_layer=1, n_head=3, code_dim=24, codebook_size=32))
    tokens = torch.randint(0, 128, (2, RAW_BLOCK))
    logits, codes, reconstruction, vq = ae(tokens)
    assert logits.shape == (2, RAW_BLOCK, 128)
    assert codes.shape == (2, CODES_PER_BLOCK)
    assert ae.decode_codes(codes).shape == logits.shape
    (reconstruction + vq).backward()

    lm = Transformer(ModelConfig(128, 16, 1, 3, 48))
    x = torch.randint(0, 128, (2, 16))
    lm_logits, loss = lm(x, x)
    assert lm_logits.shape == (2, 16, 128) and torch.isfinite(loss)
    assert lm.generate(x[:1, :4], 2).shape == (1, 6)

    baseline = Transformer(lm_config("baseline", 8192)).num_parameters()
    latent = Transformer(lm_config("latent", CODEBOOK_SIZE)).num_parameters()
    assert 9_000_000 <= baseline <= 11_000_000
    assert 9_000_000 <= latent <= 11_000_000
    print(
        json.dumps(
            {
                "status": "ok",
                "compression": f"{RAW_BLOCK} BPE -> {CODES_PER_BLOCK} codes",
                "baseline_parameters": baseline,
                "latent_parameters": latent,
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

    command = commands.add_parser("train-ae", help="train the 64-token to 16-code VQ autoencoder")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--out", default="checkpoints/autoencoder")
    command.add_argument("--steps", type=int, default=0, help="0 means one pass over the training tokens")
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--lr", type=float, default=3e-4)
    command.add_argument("--eval-every", type=int, default=500)
    command.add_argument("--eval-batches", type=int, default=20)
    command.add_argument("--save-every", type=int, default=500)
    add_device(command)
    command.set_defaults(function=train_ae)

    command = commands.add_parser("encode", help="freeze the autoencoder and convert BPE data to latent codes")
    command.add_argument("--checkpoint", default="checkpoints/autoencoder/best.pt")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--out-dir", default="data/latent")
    command.add_argument("--batch-size", type=int, default=256)
    add_device(command)
    command.set_defaults(function=encode_dataset)

    command = commands.add_parser("train-lm", help="train one equal-text pass of a 10M LM")
    command.add_argument("kind", choices=("latent", "baseline"))
    command.add_argument("--data-dir", default="")
    command.add_argument("--out", default="")
    command.add_argument("--steps", type=int, default=0, help="0 means one pass over the dataset")
    command.add_argument("--batch-size", type=int, default=16)
    command.add_argument("--grad-accum", type=int, default=1)
    command.add_argument("--lr", type=float, default=3e-4)
    command.add_argument("--eval-every", type=int, default=500)
    command.add_argument("--eval-batches", type=int, default=20)
    command.add_argument("--save-every", type=int, default=500)
    add_device(command)
    command.set_defaults(function=train_lm)

    command = commands.add_parser("report", help="write the six requested metrics and blinded samples")
    command.add_argument("--data-dir", default="data/tinystories")
    command.add_argument("--latent-dir", default="data/latent")
    command.add_argument("--autoencoder", default="checkpoints/autoencoder/best.pt")
    command.add_argument("--baseline", default="checkpoints/baseline/best.pt")
    command.add_argument("--latent", default="checkpoints/latent/best.pt")
    command.add_argument("--out-dir", default="results")
    command.add_argument("--samples", type=int, default=3)
    command.add_argument("--new-tokens", type=int, default=256)
    command.add_argument("--eval-batches", type=int, default=50)
    command.add_argument("--eval-batch-size", type=int, default=32)
    add_device(command)
    command.set_defaults(function=report)

    command = commands.add_parser("smoke", help="run the smallest shape, gradient, and parameter-count checks")
    command.set_defaults(function=smoke)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
