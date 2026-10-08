# Full Training Report — PaddleOCR-VL-1.6 Persian LoRA (full corpus)

Run `rtx6000pro16_run`, source of record `notebooks/notebook_last.ipynb`
(`notebooks/notebook_full.ipynb` is the same pipeline whose training was cut off at
step 4,424/17,374; every number below comes from the completed run unless marked).

All figures are quoted from cell outputs, not recomputed.

---

## 1. Run identity

| Item | Value |
|---|---|
| Base model | `PaddlePaddle/PaddleOCR-VL-1.6` @ `c5630abae1d940eafe0697512a0325494b02ab42` |
| Load path | `AutoModelForImageTextToText`, `trust_remote_code: false` (transformers-5 native) |
| Config | `configs/lora_config_rtx6000pro_16.yaml` @ commit `b7bd463` |
| GPU | RTX PRO 6000 Blackwell Server Edition, `sm_120`, 97,251 MiB (96,138 MiB free) |
| Stack | torch 2.11.0+cu128 / CUDA 12.8, transformers 5.14.1, peft 0.21.2, sentencepiece 0.2.2 |
| Precision | bf16, `quantization: none`, weights 1,727 MiB base / 1,763 MiB with adapter |
| Attention | `sdpa` (no sm_120 flash-attn wheel) |
| Adapter | LoRA r16 / α32 / dropout 0.05 on `q,k,v,o,gate,up,down_proj` + embeddings + lm_head |
| Trainable | 9.39 M of 915 M (**1.03 %**), 128 adapted modules, 439 M vision params frozen |
| Batch | 16 × 1 = **16 effective** |
| LR | 2.0e-4 projections / 7.0e-5 embed+head, cosine, warmup 0.08, clip 1.0, wd 0.01 |
| Epochs | 2.0 → **17,374 optimiser steps** |
| Image budget | 1,003,520 px = **1,280 image tokens** (1.6 processor's own `longest_edge`), `max_total_seq_len` 4,096 |
| Data mode | `full` — letters **26,848 / 26,848**, Persian Pixel **31,108 / 31,000**; 37 GB + 12 GB on disk |

## 2. Dataset as built

`unified_manifest.parquet` — **117,845 records across 10,056 groups**:

| Split | Records | letter_page | letter_crop | persian_pixel_page |
|---|---|---|---|---|
| train | **111,187** | 25,150 | 56,079 | 29,958 |
| validation | **803** | 229 | 574 | — |
| test | **5,855** | 1,466 | 3,347 | 1,042 |

- Digit oversampling: **27,799** train records at density ≥ 0.138 → **138,986 samples/epoch**.
- Groups never straddle splits (hash of `group_id`), so no letter's page is both trained and scored.
- Validation holds no Persian Pixel records; the test split does — which is why §6 is the
  honest comparison and §5 is only a within-split check.

**Crop budget (taken / available)** — the weighted quota pushing the metadata block up:

| Region type | Taken | Available |
|---|---|---|
| metadata_field | 15,355 | 31,550 |
| table_cell | 7,676 | 13,064 |
| list_item | 5,757 | 93,976 |
| paragraph | 5,756 | 34,754 |
| footer_field | 3,837 | 60,468 |
| page_number | 3,837 | 12,830 |
| recipient | 3,837 | 10,000 |
| subject | 3,837 | 10,000 |
| sender | 2,877 | 3,709 |
| signatory_title | 2,877 | 9,941 |
| header_company_name | 1,438 | 24,892 |
| chart | 959 | 1,325 |
| header_slogan | 959 | 14,598 |
| religious_opener | 959 | 4,445 |
| heading | 39 | 39 |

Rejected 96,273: `crops_excluded_type` 88,808 (by design), `crops_text_rejected` 7,190,
`crops_too_thin` 269, `letters_text_rejected` 3, `crops_too_small` 2, `source_underfilled` 1.

## 3. Pre-flight stages

| Stage | Steps | Loss | Throughput | Peak VRAM |
|---|---|---|---|---|
| `smoke` | 2 | 0.8425 | 3.47 s/step, 0.22 s/sample | 52,515 MiB |
| `probe` | 30 | 1.5588 | 0.93 s/step, 0.06 s/sample | 54,145 MiB |

`check` passed with projected 93,700 MiB vs spendable 95,938 MiB; Drive authenticated,
no remote checkpoint at the start.

## 4. Training

| Metric | Value |
|---|---|
| Total steps | **17,374** (2 epochs over 138,986 samples/epoch) |
| Final `train_loss` | **0.0279** |
| Final `eval_loss` | **0.1695** (was 0.3203 at step 2,500) |
| Peak VRAM | **72,706 MiB** of 97,251 (75 %) |
| Eval throughput in-loop | 29.59 samples/s, 27.1 s per validation pass |
| Resume | from Drive `checkpoint-16000`, mirrored back after 7 downloads |
| Drive mirror | `checkpoint-17374` + `adapter_only`; 4 checkpoints, 21 files, **339,385,110 bytes**, 0 failures |
| Retention | remote `checkpoint-14000` / `checkpoint-15000` pruned per `keep_remote: 5` |

Loss trajectory (logged every 20 steps): probe 1.56 → epoch 0.12–0.40 band 0.33–0.84 →
epoch 1.92–2.00 band 0.22–0.53, with `grad_norm` falling from 50+ spikes early to 0.7–5.7
at the end. `eval_loss` 0.3203 → 0.1695 across the run, i.e. still improving when the
schedule ended — no overfitting signal at 2 epochs.

VRAM per step oscillates 17–72 GB because `DigitAwareLengthGroupedSampler` sorts each
megabatch longest-first; the peak arrives early and often, which is the intended property.

---

# Comparison Report — base vs base + LoRA

## 5. Validation split (within-split check)

Base was scored on 100 records (`stage_baseline` caps at 100), the adapter on all 803, so
the sample sets differ. Treat the direction as sound and the magnitude as indicative.

| Metric | Base (n=100) | Base + LoRA (n=803) | Δ |
|---|---|---|---|
| CER | 0.5747 | **0.0502** | −0.5246 |
| CER (folded) | 0.5670 | 0.0500 | −0.5170 |
| WER | 0.8048 | **0.0644** | −0.7403 |
| digit CER | 1.7950 | **0.0792** | −1.7158 |
| digit exact match | 0.0000 | **0.6927** | +0.6927 |
| digit runs exact | 499/1,890 (0.2640) | 4,389/4,856 (**0.9038**) | +0.64 |
| digit char ratio | 6,258 → 12,315 (**2.0×**) | balanced | over-generation gone |
| empty predictions | 0 | 0 | 0 |
| convention share of error | 1 % | 0 % | — |

### Sensitive-field audit (the project's priority metric)

| Field | Base recall | LoRA recall | Base exact | LoRA exact | LoRA digits |
|---|---|---|---|---|---|
| `registration_number` | 0.3077 | **0.9783** | 0.0000 | **0.9565** | 0.9565 |
| `letter_number` | 0.6356 | **0.9747** | 0.0339 | **0.8892** | 0.8892 |
| `date` | 0.1444 | **0.9573** | 0.0000 | **0.9231** | 0.9231 |
| `national_id` | 0.1667 | **1.0000** | 0.0000 | **1.0000** | 0.9259 |
| `amount` | 0.1538 | **0.9725** | 0.0000 | **0.8165** | 0.8165 |

Mean confidence rose on every field — `registration_number` 0.805 → 0.985,
`letter_number` 0.659 → 0.974, `date` 0.718 → 0.992, `amount` 0.821 → 0.966,
`national_id` → 0.999 — and the fault profile inverted:

| Fault | Base | LoRA |
|---|---|---|
| missing | 172 | 22 |
| mismatch | 86 | 52 |
| low_confidence | 60 | 24 |
| spurious | 9 | 26 |
| malformed | 8 | 3 |
| repeated_digits | 3 | 8 |
| script_only | 2 | 1 |

Base failed by *not finding* fields (172 missing); the adapter finds them and occasionally
over-reports (spurious 9 → 26). That is a far cheaper failure mode: a spurious field is
visible to the audit, a missing one is silent.

## 6. Test split — the headline result

**5,855 records, identical set for both variants**, greedy, full field audit.

| Dataset | n | Base CER | Base WER | LoRA CER | LoRA WER | Δ CER |
|---|---|---|---|---|---|---|
| letters | 4,813 | 0.6366 | 0.7926 | **0.0907** | **0.1133** | **−0.5459 (−86 %)** |
| persian_pixel | 1,042 | 0.4531 | 0.6497 | **0.4041** | **0.4865** | −0.0490 (−11 %) |
| **ALL** | **5,855** | **0.5636** | **0.7326** | **0.2154** | **0.2701** | **−0.3482 (−62 %)** |

| Digit metric | Base | Base + LoRA |
|---|---|---|
| digit CER | 1.5358 | **0.2563** |
| digit exact match | 0.0404 | **0.6031** |
| digit runs exact | 9,001/35,505 (0.2535) | 30,545/35,505 (**0.8603**) |

Folded CER (orthographic conventions normalised): base 0.5532 → adapter 0.2144,
**Δ −0.3388**. Because the folded gain tracks the raw gain, the improvement is genuine
reading, not normalisation of ی/ي and ZWNJ.

**Verdict as printed:** *adapter better on 2 of 2 datasets (letters, persian_pixel);
overall −0.3482 CER (−62 %); folded CER −0.3388 too, so reading genuinely improved.*

### Where the remaining error lives

The adapter is near-solved on the trained domain (letters CER 0.0907) and only marginally
better on Persian Pixel (0.4041). Persian Pixel is 18 % of the test split but carries most
of the residual error — it is a different distribution (scanned/handwritten-style pixel
pages) that the letters-weighted mix under-trains. Raising its share, or training a
separate adapter for it, is the obvious next lever.

### Confusion profile shift

| Base top confusions | Count | | LoRA top confusions | Count |
|---|---|---|---|---|
| `ی → ي` | 273 | | `0 → ۰` | 41 |
| `ZWNJ → space` | 145 | | `۰ → 0` | 20 |
| `؛ → :` | 37 | | `1 → ۱` | 17 |
| `۰ → -` | 26 | | `۲ → ۳` | 14 |
| `ی → ب` | 25 | | `۳ → ۲` | 11 |
| `: → ،` | 24 | | `، → ؛` | 11 |
| `ک → ك` | 21 | | `. → ؛` | 10 |

The base model's failures are *language-level* — Persian letters emitted in Arabic forms,
ZWNJ collapsed to space, plus 2.0× over-generation from repetition loops. The adapter's
remaining failures are *numeral-script* (Latin ↔ Persian digit family) and punctuation,
which are one-character, locally-fixable errors.

## 7. Cost of the evaluation itself

| Pass | Records | Wall clock | Per record |
|---|---|---|---|
| baseline (validation) | 100 | 491 s | 4.9 s |
| eval (validation, adapter) | 803 | 1,732 s | 2.2 s |
| compare base (test) | 5,855 | ~24,000 s | ~4.1 s |
| compare adapter (test) | 5,855 | 17,274 s | 2.95 s |

**Total ≈ 9.5 h of GPU time for one test comparison**, on a card that reported
~3 GB VRAM and ~2 % utilisation during the pass. The cause is structural, not
hardware: `score_split` generated **one record at a time** (`unsqueeze(0)`), so a
1,024-token greedy decode ran 1,024 sequential kernel launches over a batch of 1,
and the CPU-side image decode + processor call sat on the critical path between
records. This is addressed in §8.

## 8. What changed after this run

The cost above is structural. `score_split` generated **one record at a time**
(`unsqueeze(0)`), so a 1,024-token greedy decode ran 1,024 sequential launches of
kernels sized for a single row — launch latency, not arithmetic, which is exactly
what 2 % utilisation at 4.9 s/record looks like. The CPU-side image decode and
processor call also sat on the main thread *between* records, so the GPU idled
through them.

`eval.py` now dispatches through `run_score_split`, which picks the batched
scorer in `training/batch_eval.py` whenever `eval.batch_size > 1`:

- **Left-padded batched generation** — `eval.batch_size` prompts per `generate`
  call. Left padding is not a preference: the answer grows rightwards, so only
  left padding puts every row's first new token at the same index, which is what
  makes the single slice `generated[:, prompt_width:]` correct for all rows.
- **Length-bucketed ordering** — records sorted longest-first so a batch pads to
  its own longest row, and so an OOM surfaces in the first minutes rather than at
  03:00. Predictions are keyed by manifest position and written at the end, so
  `eval_predictions.jsonl` stays byte-comparable with an unbatched run's.
- **Background encode** — a `DataLoader` with `eval.dataloader_workers` worker
  processes runs Pillow and the processor ahead of the GPU, two batches of
  lookahead each.
- **Per-row confidence** — `BatchConfidenceRecorder` keeps one list of steps per
  row, so the field audit's weakest-token figure survives batching. The
  single-row `ConfidenceRecorder` still rejects a wide batch, correctly.

Unchanged: same encoder, same template, same greedy decoding, same metric
arithmetic. A batched CER is comparable to a batch-1 CER because only the
scheduling moved.

| Profile | `eval.batch_size` | `eval.dataloader_workers` |
|---|---|---|
| `lora_config_rtx6000pro_16.yaml` | 32 | 4 |
| `lora_config_a100.yaml` / `_l40s` / `_rtx6000pro` | 16 | 4 |
| `lora_config.yaml` (4 GB laptop) | 1 | 2 |

The 4 GB profile stays at 1 deliberately: a batched decode holds one KV cache of
`max_new_tokens` **per row**, and two will not fit in ~2 GB of spendable VRAM.
`--batch-size 1` reaches the original loop from any profile, which is how the two
paths get bisected against each other.

## 9. Reproducing / extending

```bash
# fast test comparison, base vs adapter, both variants over identical records
CONFIG=configs/lora_config_rtx6000pro_16.yaml \
  bash scripts/train_molab.sh compare --split test --batch-size 32 --eval-workers 4

# report bundle (CSV + JSON + Markdown + charts) from the comparison just written
CONFIG=configs/lora_config_rtx6000pro_16.yaml \
  bash scripts/train_molab.sh report --split test
```

`notebooks/eval_report.ipynb` does the whole thing from a fresh molab session:
pulls the corpora, rebuilds the identical manifest from the seed, restores
`adapter_only` from Drive via `CheckpointSync.fetch_folder`, scores both variants
on the test split, renders the bundle, and pushes it back to Drive with
`CheckpointSync.push_directory`.

Bundle contents, written to `<output_dir>/report_<split>/`:

| File | Contents |
|---|---|
| `summary.json` | Every headline number, one flat object |
| `overall.csv` | Base vs adapter vs delta, one row per metric |
| `by_dataset.csv` | The same split per dataset (letters, persian_pixel) |
| `by_source.csv` | And per manifest source tag |
| `fields.csv` | Per-field recall / exact / digits, both variants |
| `confusions.csv` | Top substitution pairs, both variants |
| `training_curve.csv` | Loss and eval-loss against step, when available |
| `report.md` | The tables above, rendered for a human |
| `charts/*.png` | Error rates, per-dataset CER, field recall, training curve |
