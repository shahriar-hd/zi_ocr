# PaddleOCR-VL Persian LoRA — Training Report
## RTX PRO 6000 run (`rtx6000pro16_run`, PaddleOCR-VL-1.6, tiny mode)

> Sources: `notebooks/notebook_run.ipynb` (26 cells; repo has no `notebooks/notebook.ipynb` —
> `notebook_run.ipynb` is the full-run notebook), stage driver `scripts/train_molab.sh`,
> config `configs/lora_config_rtx6000pro_16.yaml` @ commit `b7bd463`,
> `configs/lora_config*.yaml`, `temp/*sim*.json`, `temp/*config_on*.txt`,
> `configs/field_validation.yaml`, `docs/decisions.md`.
> All numbers below are quoted from cell outputs, not recomputed.

## 1. Run identity

| Item | Value |
|---|---|
| Notebook | `notebooks/notebook_run.ipynb`, 26 cells (marimo) |
| Config | `configs/lora_config_rtx6000pro_16.yaml` |
| `output_dir` / Drive folder | `data/training/rtx6000pro16_run` → Drive `fa-ocr-lora / rtx6000pro16_run` |
| Base model | `PaddlePaddle/PaddleOCR-VL-1.6` @ `c5630abae1d940eafe0697512a0325494b02ab42` (2026-08-08), `auto_class: image_text_to_text`, `trust_remote_code: false` (native transformers-5 path, no remote code) |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, `sm_120`, 95.0 GiB total (97,251 MiB), 93.9 GiB free at start; bf16 supported |
| Software | torch `2.11.0+cu128` (CUDA 12.8), transformers `5.14.1`, peft `0.21.2`, datasets (installed), sentencepiece `0.2.2`, `Trainer` imports cleanly, `tf-keras` reconciled |
| Data mode | `DATA_MODE="tiny"` (~1% of each corpus): **164 letters pages** of 26,848 ceiling + **531 pixel rows** of 31,000 ceiling; images on disk 216 MiB + 211 MiB |
| Seed / splits | `seed: 1337`, `validation_size: 800` (capped by corpus), `test_fraction: 0.05`; group-hash holdout so splits are comparable across configs |
| LoRA shape | `r:16, alpha:32, dropout:0.05`, targets `q/k/v/o/gate/up/down_proj` + `adapt_embeddings + adapt_lm_head` → **9.39M trainable / 915M (1.03%)**, 128 modules, 439M visual params frozen |
| Optim | `AdamW fused`, `lr 2.0e-4` (projections) / `7.0e-5` (embed+head), `weight_decay 0.01`, `warmup_ratio 0.08`, `cosine`, `max_grad_norm 1.0`, `epochs 2.0`, **effective batch 16 = 16×1** |
| Image budget | `max_pixels: 1,003,520` (= 28×28×1280 = **1,280 image tokens**, the 1.6 processor's own `longest_edge`; larger values are rejected with "predicted 2772 but processor produced 1218"), `min_pixels: 147,384`, `max_total_seq_len: 4096`, `loss_chunk_rows: 1024`, `on_oversize: shrink_image`, `autoprobe_vram: false`, `quantization: none` (bf16 weights ~1.73 GiB), `attn: sdpa`, `grad_checkpointing: false`, `workers: 4`, `save_steps/eval_steps: 500` |
| Eval | `max_samples: null` (whole split), `max_new_tokens: 1024`, `beams: 1`, `confusion_top_k: 20`, field audit on `configs/field_validation.yaml` (5 fields) |

## 2. Stage-by-stage outputs

### 2.1 `check` (cell 18)
- Interpreter/config/VRAM banner: torch/transformers/peft/GPU/bf16/Trainer/sentencepiece all pass.
- `reserved_vram_mib: 93,700` projected peak (`none, vision bf16`), spendable 95,938 MiB after 200 MiB margin; holding `python 1100 MiB ×2`.
- Manifest does not exist yet; output dir + Drive `on → fa-ocr-lora/rtx6000pro16_run`; Drive authenticated, no remote checkpoint.

### 2.2 `manifest` (cell 18)
- Wrote `unified_manifest.parquet` + `split_summary.json`.
- **1,074 records across 67 groups**: train **913** (`letter_page=130, letter_crop=473, persian_pixel_page=310`), validation **103** (`22+81`, no pixel), test **58** (`12+46`, no pixel).
- Digit oversampling: **229 train records** at density ≥ 0.105 repeated → **1,142 samples/epoch**.
- Crop budget taken/avail: `metadata_field 181/210`, `list_item 68/526`, `paragraph 68/235`, `footer_field 45/333`, `page_number 45/69`, `recipient 45/66`, `subject 45/66`, `signatory_title 33/66`, `sender 21/21`, `header_company_name 16/155`, `chart 11/16`, `header_slogan 11/84`, `religious_opener 11/32`.
- Rejected 53,948: `letters_image_missing 26,684`, `crops_image_missing 26,684`, `crops_excluded_type 554`, `crops_text_rejected 25`, `source_underfilled 1`.
- Caveat: tiny mode → validation/test are **letters-only**; pixel (310 pages) is train-only here. Full run expects 800-record holdout + ~4,342 test with pixel included.

### 2.3 `smoke` — 2 steps (cell 18)
- 16 records → 18 samples/epoch, batch 16. Frozen 439M, 9.39M trainable, bf16 1,763 MiB.
- `loss 1.2144`, 2 steps in 7 s (3.29 s/step, 0.21 s/sample), **peak VRAM 51,591 MiB** vs 93,700 reserved.
- Purpose: wiring proof only.

### 2.4 `probe` — 30 steps (cell 18)
- Full tiny train split: 913 records → 1,142 samples/epoch.
- `loss 1.6676`, 30 steps in 26 s (**0.87 s/step, 0.05 s/sample**), **peak VRAM 56,082 MiB**.
- Projection: 0.02 h/epoch over 1,142 samples; 2 epochs ≈ 0.0 h at this corpus size. Full corpus (~102k samples/epoch, 6,383 steps/epoch) is what `temp/*sim*.json` costs out (see §4).

### 2.5 `baseline` — base model, no adapter (cell 18, 100 validation records, 125 s)
- Weights 1,727 MiB, base-only. **5 empty predictions**, output 2.2× too long (1,647 expected chars → 3,685 produced).
- CER **0.4179** (folded 0.4077, 2% convention) · WER **0.5648** (folded 0.5063) · digit CER **1.9678** (69 rec) · digit exact **0.0580** · digit runs **138/546 (0.2527)**.
- Top confusions: `ی→ي 119`, `ZWNJ→space 50`, `؛→: 17`, `۰→۵ 16`, `ی→ى 14`, `.→، 10` — i.e. Arabic-form letters, ZWNJ/space, digit misreads, over-generation.
- Field recall: `registration_number 0.5000 (exact 0.0833)`, `letter_number 0.5227 (exact 0.0000)`, `date 0.1111 (exact 0.0000)`, `national_id 0.0000`, `amount 0.0000`. Faults: missing 59, mismatch 28, low_conf 14, spurious 6, malformed 3.

### 2.6 `train` (cells 19–20, pid 1806, detached nohup)
- Started via `scripts/train_molab.sh train`; monitor polled GPU + console tail.
- Finished: **144 steps in 110 s (0.76 s/step, 0.05 s/sample)**, `train_loss 1.0893`, `eval_loss 0.5753`, **peak VRAM 56,156 MiB**, `eval_runtime 5.0 s (20.5 samples/s)`.
- Adapter written to `adapter_only/`; Drive sync: `checkpoint-144` + `adapter_only` mirrored (14 files, **161,731,861 bytes**, 0 skipped, 0 failures).
- Console shows warmup from ~6.6 s/it (compile) to ~4.9 it/s steady; 94% GPU util while stepping, 0% + 559 MiB after.

### 2.7 `eval` — trained adapter, validation (cell 20 auto-report, 103 records, 224 s)
- Weights 1,763 MiB + `adapter_only`. Output length now correct: 1,647 expected → **1,698 produced (1.0×)**; **0 empty predictions**.
- CER **0.2619** (folded 0.2611, 0% convention) · WER **0.3473** (folded 0.3339) · digit CER **0.4706** · digit exact **0.2174** · digit runs **313/546 (0.5733)**.
- Top confusions now: `؛→: 20`, `ی→ب 5`, `۳→۴ 4`, `۰→۵ 4`, `ی→ي 4`, `ZWNJ→space 4` — Arabic-form and repetition errors largely gone; residual is digit/letter confusions + punctuation.
- Field audit (103 rec): `registration_number 0.5833/0.5000/0.5000`, `letter_number 0.7955/0.2955/0.2955`, `date 0.6667/0.5556/0.5556`, `national_id 0.5000/0.0000/0.0000`, `amount 0.0000/0.0000/0.0000`; confidences mean 0.81–0.87; faults missing 30, low_conf 25, mismatch 25, spurious 18, malformed 2.
- Base→trained delta (100→103 samples, +3): **CER −0.1559, WER −0.2174, digit CER −1.4973, digit exact +0.1594, empty −5**; field recall reg 0.50→0.58, letter 0.52→0.80, date 0.11→0.67, national 0.00→0.50, amount 0.00→0.00.

### 2.8 `compare` — base vs adapter, test (cells 20/24, 58 records, base 67 s + adapter 119 s)
- Same 58 test records both variants, letters-only in tiny mode.
- Overall: base **CER 0.2978 / WER 0.4924** → adapter **CER 0.1576 / WER 0.2401** = **Δ −0.1403 (−47%)**; folded CER 0.2864→0.1557 (Δ −0.1307), so genuine reading gain, not orthographic normalisation.
- Digits: CER 0.9978→0.2910, exact 0.0500→0.2500; runs 87/304 (0.286) → 202/304 (0.665).
- Per-dataset: `letters n=58` −0.1403; verdict "adapter better on 1 of 1 datasets".
- Artefacts: `compare_test.json`, `compare_test_*_predictions.jsonl`, `eval_validation.json` (+ `eval_validation_base.json` backup), `eval_predictions.jsonl`, `training_summary.json`, `console.log`, `train.pid`.

## 3. What the numbers mean
1. **Adapter works on tiny data:** −37% CER on validation, −47% on test, digit CER 1.97→0.47 (val) / 1.00→0.29 (test), over-generation 2.2×→1.0×, empties 5→0.
2. **Digits/fields move most:** letter_number recall 0.52→0.80, date 0.11→0.67; digit runs 25%→57% (val), 29%→66% (test).
3. **Remaining gaps:** `amount` 0/6 (table amounts under grouped-separator pattern — no learning signal in tiny run); `national_id` n=2 (underpowered); letter_number exact only 0.30 (recall without exactness — finds the field, mistranscribes digits); date confidence weakest-token 0.25–0.60 on failures (review queue signal works).
4. **VRAM headroom is large:** measured train peak 56.2 GiB vs 93.7 GiB reserved (from original-model measurement) vs 95.9 GiB spendable. The reserve is conservative for 1.6 (1,280 vs 2,775 image tokens); re-probe value should replace it before a full run.
5. **Tiny ≠ full:** 1,074 records / 144 steps / ~2 min train vs full-corpus 86,845 records / 12,766 steps / ~4–6 h (see §4). Val/test lack pixel pages; amount/national_id conclusions need the full split.

---

# Compare Report — configs, hardware, and cost models

## A. Config matrix (all `seed: 1337`, LoRA `r16/α32/drop0.05` + embed/head, 7 proj modules)

| Config | Target | Base model | Batch (per×accum=eff) | `max_pixels` (tokens) | `seq_len` | Quant | GradCkpt | Optim | Workers | Drive | Output dir |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `lora_config.yaml` | RTX 3050 Ti Laptop 4 GB (dev) | `PaddleOCR-VL` @2b77538, `causal_lm`, remote code | 1×16=16 | 1,003,520 (1,280) | 3,072 | **nf4 + vision quant** (870 MiB) | **on** | auto (paged 8-bit) | 2 | off | `local_run` |
| `lora_config_a100.yaml` | A100 40 GB (32 vCPU/64 RAM/200 disk) | same original | **8×2=16** | **2,175,600 (2,775)** full-page unscaled | 4,096 | none (1,828 MiB) | off | fused | 8 | on | `a100_run` |
| `lora_config_l40s.yaml` | L40S 45 GB (12 vCPU/126 RAM/497 disk) | same original | 8×2=16 | 2,175,600 (2,775) | 4,096 | none (1,828 MiB) | off | fused | **6** | on | `l40s_run` |
| `lora_config_rtx6000pro.yaml` | RTX PRO 6000 96 GB (4 vCPU/32 RAM, molab) | same original | **16×1=16** | 2,175,600 (2,775) | 4,096 | none (~1.73 GiB) | off | fused | 4 | on | `rtx6000pro_run` |
| `lora_config_rtx6000pro_16.yaml` (**this run**) | same RTX PRO 6000 | **`PaddleOCR-VL-1.6` @c5630ab, `image_text_to_text`, NO remote code** | 16×1=16 | **1,003,520 (1,280)** — 1.6 clamp | 4,096 | none (1,763 MiB) | off | fused | 4 | on | `rtx6000pro16_run` |
| `lora_config_rtx6000pro_16.yaml` vs sibling | — | **different base models: never compare one base against the other's adapter; use `compare` (base vs own adapter)** | same eff. batch → same LR 2e-4/7e-5 valid | −1,495 tokens/page vs original | inherited, comfortable | same | same | same | same | same | isolated so checkpoints can't cross-resume |

Key design invariants: effective batch is **16 everywhere** (LRs carry over); `reserved_vram_mib` 30,000 (A100/L40S, modelled) vs 93,700 (RTX PRO 6000, measured on original model); `save_steps 500`, `epochs 2`, `loss_chunk_rows 1024` (datacentre) vs 128 (local); full-run corpus ceilings 26,848 pages + 60,000 crops + 31,000 pixel everywhere.

## B. Cost-model comparison (`temp/*.json/.txt`, full corpus 86,845 rec → 102,129 samples/epoch, 6,383 steps/epoch, 12,766 steps)

| Target | Peak worst-batch | Fraction | Mean batch | Headroom | Supply vs demand | Expected run | Notes |
|---|---|---|---|---|---|---|---|
| A100 40 GB | 30,089 MiB (29.4 GiB) | 73.5% | 10,754 MiB | 10.6 GiB | 297 vs 12.2/s (24×, compute-bound) | **4.65 h** (3.62–6.51) | Decoder activations 23,279 MiB dominate; dense-vision counterfactual 61.6 GiB would OOM — block-diagonal verified by `probe_vision_attention.py` (exponent 1.00) |
| L40S 45 GB | **same 30,089 MiB** (MiB don't scale with card) | **65.3%** | same | **15.6 GiB** | 223 vs 14.2/s (16×) | **4.01 h** (3.12–5.61) | 1.16× bf16 TFLOPS (362 vs 312) is an upper bound; 0.9× bandwidth (625 vs 1,555 GB/s) — LoRA streams frozen weights; `l40s_config_on_a100` / `a100_config_on_l40s` cross-checks identical |
| RTX PRO 6000 96 GB (original model) | **93,643 MiB measured** (probe 93,559) | ~96% | — | ~2.2 GiB | — | hours (6,800 steps/epoch @16) | Old 30,000 reserve understated 3×; corrected to 93,700 |
| RTX PRO 6000 96 GB (**1.6, this run, tiny**) | **56,156 MiB measured** (smoke 51,591, probe 56,082) | ~58% of 97,251 | — | ~40 GiB | GPU 94% while stepping | **110 s for 144 steps** (tiny); full-run hours scale with §B | Lower because 1,280 not 2,775 image tokens; 93,700 banner is inherited — replace with 1.6's own probe reading |
| Local 3050 Ti 4 GB | ~1,287 MiB measured (proj 1,387) | ~65% of ~2 GiB free | — | ~1.3 GiB | — | ~5.5 h (2 epochs, 11.5k samples/epoch, 0.86 s/sample) | NF4+vision-quant mandatory; bf16 (~1,730 MiB weights) never fit |

Storage/RAM are non-binding everywhere: ~43.5 GiB disk (37.4 images + 3.6 base + 0.5 manifest + 2.0 headroom), 9.5–11.6 GiB RAM.

## C. Model-quality comparison (what exists today)
- **Only measured adapter:** this 1.6 tiny run (§2.7–2.8). No A100/L40S/original-model full-run metrics exist yet — only `fa-ocr-simulate` projections (no model loaded, no CUDA).
- **Local baseline reference** (`docs/decisions.md`): original base CER **1.2220** / WER 1.6410 / digit CER 4.2063 on 50 validation records (repetition loops; Devanagari/Arabic-form/ZWNJ confusions). Not comparable to this run's baseline (different base 1.6, different tiny split: CER 0.4179 val / 0.2978 test) — compare **within** base only.
- **Full-run expectation:** same adapter shape, same LRs, ~100× more samples, full 800-record holdout + 4,342 test *with* pixel pages; amount/national_id need that scale before judging.

## D. Recommendations
1. **Promote 1.6 path; keep original profile frozen.** Transformers-5-native load, no remote code, and −47% test CER in 2 minutes of training is the supported path.
2. **Re-probe on 1.6 and lower `reserved_vram_mib`** from 93,700 to the measured ~56–60 GiB + margin; consider `per_device_batch_size` 16→24 or `r` 16→32 only if eval shows rank-limit (VRAM allows, throughput may not need it).
3. **Run full corpus next** (`DATA_MODE=full`): expect ~6,800 steps/epoch, ~4–6 h, checkpoint every 500 steps with Drive mirror; watch `amount` recall and `letter_number` exact-match as the scale signals.
4. **Keep `seed 1337` + validation/test carving unchanged** so tiny→full and card→card CERs stay directly comparable.
5. **Tighten `field_validation.yaml` only per-deployment** (organisation-specific number schemes); keep tri-script digit acceptance (script error ≠ malformation).
