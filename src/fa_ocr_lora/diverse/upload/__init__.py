"""Stage 08 - upload to the Hugging Face Hub, resumably.

Every file is verified against the size the Hub reports and then recorded in
``upload_ledger.jsonl``, so an interrupted push resumes exactly where it
stopped and nothing local is deleted until the remote copy is confirmed.
"""
