# Lethe

Minimal experiment: does a learned 4x discrete representation make a small story model cheaper **after paying for the representation**?

## Fixed MVP

- TinyStories: 50M train BPE tokens, 1M validation tokens
- byte-level BPE: 8,192 entries
- VQ autoencoder: 64 BPE tokens -> 16 codes, 1,024-entry codebook -> 64 reconstructed tokens
- latent LM: 10.3M parameters, 64-code context (256 BPE-token equivalent)
- baseline LM: 10.0M parameters, 256-token context
- one equal-text pass for the autoencoder and each LM unless `--steps` is given

The two LMs consume the same amount of original text per optimizer step. The latent LM performs attention over 64 positions while the baseline performs attention over 256.

## Run

```powershell
uv sync
uv run python lethe.py smoke
uv run python lethe.py prepare
uv run python lethe.py train-ae
uv run python lethe.py encode
uv run python lethe.py train-lm latent
uv run python lethe.py train-lm baseline
uv run python lethe.py report
```

Every training command resumes from `latest.pt`. No command above is launched automatically.

For a quick end-to-end plumbing check, use small separate paths so the real run remains untouched:

```powershell
uv run python lethe.py prepare --data-dir data/smoke --tokenizer-docs 200 --train-tokens 20000 --val-tokens 5000 --vocab-size 512
uv run python lethe.py train-ae --data-dir data/smoke --out checkpoints/smoke-ae --steps 2 --batch-size 2 --eval-every 1 --eval-batches 1 --save-every 1
uv run python lethe.py encode --data-dir data/smoke --out-dir data/smoke-latent --checkpoint checkpoints/smoke-ae/best.pt --batch-size 8
uv run python lethe.py train-lm latent --data-dir data/smoke-latent --out checkpoints/smoke-latent --steps 2 --batch-size 2 --eval-every 1 --eval-batches 1 --save-every 1
uv run python lethe.py train-lm baseline --data-dir data/smoke --out checkpoints/smoke-baseline --steps 2 --batch-size 2 --eval-every 1 --eval-batches 1 --save-every 1
uv run python lethe.py report --data-dir data/smoke --latent-dir data/smoke-latent --autoencoder checkpoints/smoke-ae/best.pt --latent checkpoints/smoke-latent/best.pt --baseline checkpoints/smoke-baseline/best.pt --out-dir results/smoke --samples 1 --new-tokens 64 --eval-batches 1 --eval-batch-size 2
```

## Results

`report` writes only the requested evidence:

- `report.md` / `report.json`: reconstruction, compression, training time, total time, and generation latency
- `reconstructions.md`: original vs reconstructed validation text
- `samples_blind.md`: blinded baseline/latent continuations for final quality scoring
- `sample_key.json`: reveal only after rating the samples

Latent and BPE validation losses are intentionally not compared: their vocabularies and prediction targets differ. The experiment succeeds only if blinded story quality is comparable and `latent_total` is materially faster than baseline training.
