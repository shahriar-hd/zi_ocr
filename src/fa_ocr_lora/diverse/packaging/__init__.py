"""Stage 07 - group-aware splits, Parquet shards and the dataset card.

A page and every crop taken from it share a ``group_id`` and therefore a
split, so a model can never meet a line in training and its own parent page in
test. Three hard tests - unseen fonts, unseen templates, unseen generator -
measure what a random split cannot.
"""
