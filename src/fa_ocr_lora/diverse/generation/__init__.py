"""Stage 02 - text generation against the NVIDIA-hosted endpoint.

One worker thread per API key, each holding at most one request in flight, so a
throttled or quota-exhausted key sleeps alone while its siblings keep working.
Replies are cached on disk by prompt hash and written one file per sample, which
is what makes the only stage that costs money also the most resumable.
"""
