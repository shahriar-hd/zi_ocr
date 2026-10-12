"""Persian diverse OCR dataset pipeline.

A second, deliberately heterogeneous corpus, built because the first one was
too narrow. The administrative-letter dataset taught a LoRA-tuned
PaddleOCR-VL to read letters well (CER about 0.09) and barely moved a
different distribution (persian_pixel stayed near 0.40): a model trained on
one generator learns that generator, not Persian.

So here **diversity is the product, not volume**. The stages are:

====  ======================================================  ==================
  00  harvest, tag and band a Persian lexicon                 ``lexicon.parquet``
  01  sample plans across independent axes                    ``plans.jsonl``
  02  generate text with several NVIDIA-hosted models         ``texts/``
  03  normalise, filter and dedupe                            ``texts_clean.jsonl``
  04  choose format, template, fonts, degradation             ``layouts.jsonl``
  05  render through Chromium with exact box harvest           images + annotations
  06  reject illegible, duplicate or mislabelled samples      ``accepted.jsonl``
  07  group-aware splits and sharded Parquet                  ``hf/``
  08  parallel, resumable upload                              Hugging Face Hub
====  ======================================================  ==================

Stages 00 to 03 run on the laptop, because they need the API keys and cost
money; 04 to 08 run on a marimo/molab runtime, because they need Chromium and
bandwidth. The handoff is the text artifacts, mirrored to Google Drive.
"""
