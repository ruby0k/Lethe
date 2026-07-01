# Lethe

Minimal experiment: does a learned 4x discrete representation make a small story model cheaper **after paying for the representation**?

## Fixed MVP

- TinyStories: 50M train BPE tokens, 1M validation tokens
- byte-level BPE: 8,192 entries
- FSQ encoder: 64 BPE tokens -> 16 ten-bit codes with 1,024 possible values
- contextual autoregressive decoder: 16 codes + previous 64 decoded tokens -> next 64 tokens
- latent LM: 10.3M parameters, 64-code context (256 BPE-token equivalent)
- baseline LM: 10.0M parameters, 256-token context
- one equal-text pass for the autoencoder and each LM unless `--steps` is given

Both LMs consume 50M original-token equivalents. The latent LM uses batch 64 and finishes in 3,052 steps; the baseline uses batch 16 and needs 12,208 steps.

## Run

```powershell
uv sync
uv run python lethe.py smoke
uv run python lethe.py prepare
uv run python lethe.py train-ae
uv run python lethe.py encode
uv run python lethe.py train-decoder
uv run python lethe.py train-lm latent
uv run python lethe.py train-lm baseline  # skip when checkpoints/baseline already exists
uv run python lethe.py report
```

Every training command resumes from `latest.pt`. No command above is launched automatically.
`train-decoder` freezes the encoder and reuses `data/latent-v2`; it does not re-encode data or retrain generators.

For a quick end-to-end plumbing check, use small separate paths so the real run remains untouched:

```powershell
uv run python lethe.py prepare --data-dir data/smoke --tokenizer-docs 200 --train-tokens 20000 --val-tokens 5000 --vocab-size 512
uv run python lethe.py train-ae --data-dir data/smoke --out checkpoints/smoke-ae-v2 --steps 2 --batch-size 2 --eval-every 1 --eval-batches 1 --save-every 1
uv run python lethe.py encode --data-dir data/smoke --out-dir data/smoke-latent-v2 --checkpoint checkpoints/smoke-ae-v2/best.pt --batch-size 8
uv run python lethe.py train-decoder --autoencoder checkpoints/smoke-ae-v2/best.pt --data-dir data/smoke --latent-dir data/smoke-latent-v2 --out checkpoints/smoke-decoder-v3 --steps 2 --batch-size 2 --eval-every 1 --eval-batches 1 --save-every 1
uv run python lethe.py train-lm latent --data-dir data/smoke-latent-v2 --out checkpoints/smoke-generator-v2 --steps 2 --batch-size 2 --eval-every 1 --eval-batches 1 --save-every 1
uv run python lethe.py train-lm baseline --data-dir data/smoke --out checkpoints/smoke-baseline --steps 2 --batch-size 2 --eval-every 1 --eval-batches 1 --save-every 1
uv run python lethe.py report --data-dir data/smoke --latent-dir data/smoke-latent-v2 --autoencoder checkpoints/smoke-decoder-v3/best.pt --latent checkpoints/smoke-generator-v2/best.pt --baseline checkpoints/smoke-baseline/best.pt --out-dir results/smoke-v3 --samples 1 --new-tokens 64 --eval-batches 1 --eval-batch-size 2
```

## Results

`report` writes only the requested evidence:

- `report.md` / `report.json`: reconstruction, compression, training time, total time, and generation latency
- `reconstructions.md`: original vs reconstructed validation text
- `samples_blind.md`: blinded baseline/latent continuations for final quality scoring
- `sample_key.json`: reveal only after rating the samples

Latent and BPE validation losses are intentionally not compared: their vocabularies and prediction targets differ. The codec cost is amortized, so the report includes the number of generator runs needed to break even.

## Adaptive-rate experiment

The frozen v4 refiner can be reused at different rates; only encoding and the generator need rerunning:

```powershell
uv run python lethe.py sweep-refinement
uv run python lethe.py encode-refined --rate 0.125 --out-dir data/latent-v4-rate2
uv run python lethe.py train-lm refined --data-dir data/latent-v4-rate2 --out checkpoints/latent-v4-rate2
uv run python lethe.py report-refined --refined-dir data/latent-v4-rate2 --refined-model checkpoints/latent-v4-rate2/best.pt --out-dir results/v4-rate2
uv run python lethe.py sweep-literals
```

Two extras per 64-token block was the measured knee: 53.61% greedy reconstruction, 3.24x compression, and 146.2s generator training (4.03x faster than baseline). Literal rare-token escapes restore selected tokens exactly but reduce compression too sharply; see `results/v4-literal-sweep/literal-sweep.md`.

## Joint fixed-rate experiment

The v3.1 experiment jointly fine-tunes the encoder and contextual decoder for two passes, applies capped rare-token loss weighting, and corrupts 25% of decoder prefixes with its own teacher-forced predictions:

```powershell
uv run python lethe.py train-joint-codec
uv run python lethe.py encode --checkpoint checkpoints/autoencoder-v3.1-joint/best.pt --out-dir data/latent-v3.1
uv run python lethe.py train-lm latent --data-dir data/latent-v3.1 --out checkpoints/latent-v3.1
uv run python lethe.py report --latent-dir data/latent-v3.1 --autoencoder checkpoints/autoencoder-v3.1-joint/best.pt --latent checkpoints/latent-v3.1/best.pt --out-dir results/v3.1-joint
uv run python lethe.py beam-diagnostic
```

This is a documented negative result: teacher-forced accuracy improved from 67.54% to 69.56%, but greedy reconstruction fell from 51.11% to 49.12%. Width-4 beam search did not catch original v3 on the same blocks. Keep v3 as the default.

## General rarity-priority experiment

This experiment deliberately contains no name, capitalization, entity, or domain-specific rules. It weights tokens by inverse corpus frequency, gives a capped boost to the hardest 20% of occurrences, and adds a local code-only auxiliary prediction head:

```powershell
uv run python lethe.py train-priority-codec
uv run python lethe.py encode --checkpoint checkpoints/autoencoder-v3.2-priority/best.pt --out-dir data/latent-v3.2-priority
uv run python lethe.py rare-diagnostic --blocks 1600
```

On the same 1,600 validation blocks, accuracy for tokens occurring at most 1,000 times improved from 3.30% to 5.92%, while overall greedy reconstruction fell from 55.64% to 52.67%. The mechanism reallocates fixed-rate capacity toward rare occurrences, but the tradeoff is not good enough to train another generator. See `results/v3.2-priority/assessment.md`.

## Fixed correction-code experiment

This is the successful "reverse lossy" codec test. Original v3 still emits 16 semantic codes; each block then stores two lossless 20-bit `(position, original BPE token)` corrections chosen by general frequency and reconstruction surprisal:

```powershell
uv run python lethe.py correction-diagnostic
uv run python lethe.py encode-corrected
uv run python lethe.py train-lm corrected
uv run python lethe.py report-corrected
```

Two corrections achieve 56.41% greedy reconstruction and 58.56% rare-token accuracy at 3.20x compression. A controlled equal-work benchmark measured corrected-generator training 2.62x faster than baseline. Generated stories remain worse because the model sometimes emits implausible correction pairs; see `results/v3.3-corrections/assessment.md`.

## Structured correction-head experiment

The suggested separate apply/position/token heads were tested both over a frozen v3 generator and jointly with the semantic-code objective:

```powershell
uv run python lethe.py train-correction-head
uv run python lethe.py structured-correction-diagnostic
uv run python lethe.py train-correction-head --joint --out checkpoints/correction-head-v3.4-joint
```

The head can classify whether a rare correction is needed (about 70% F1), but cannot recover its content from v3 states: final exact pair accuracy was 0.068% frozen and 0% joint. Applying predicted corrections reduced reconstruction accuracy, so free-generation testing was skipped. See `results/v3.4-structured-head/assessment.md`.

## Controlled semantic-language pilot

The pilot strips function words, marks event boundaries, and replaces repeated entities with typed definition/reference symbols. It then trains a 9.97M autoregressive surface realizer on 1M TinyStories BPE tokens:

```powershell
uv run python lethe.py prepare-semantic-ir
uv run python lethe.py train-semantic-realizer
uv run python lethe.py report-semantic-realizer
```

The representation compressed validation sequences only 1.82x. Teacher-forced accuracy reached 71.18%, but free greedy reconstruction was 9.50% exact with 76.43% semantic lexical recall. Stories remained readable while names and fine facts drifted. This is useful as a lossy planning language, not yet a replacement for v3's fixed codec, so an IR generator was not trained. See `results/semantic-ir-pilot/assessment.md`.

## Longer v3 training

Continuing the frozen-code v3 decoder and latent generator from one to two passes improved controlled greedy reconstruction from 55.64% to 56.93% and latent validation loss from 5.7481 to 5.6619. Rare-token accuracy remained poor at 4.02%, and fixed-prompt samples showed no consistent quality gain. Keep the two-pass decoder when reconstruction matters; extra generator training is not the default. See `results/v3-2pass/assessment.md`.

## 40M latent-generator experiment

`train-lm latent40` trains a 38.99M-parameter generator on the unchanged v3 code stream. One pass reached 5.7084 validation loss versus 5.7481 for the one-pass 10M model, but took 1,279.5s and retained the same repetition and entity confusion. The BPE baseline was preferable in about four of five fixed-prompt pairs. Larger generator capacity is not the missing piece; see `results/v3-latent40/assessment.md`.
