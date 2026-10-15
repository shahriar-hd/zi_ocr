"""Stage 06 - image QC.

Every check answers one question: would this sample teach the model something
false? A blurry page is fine, because real scans are blurry. A page whose
label claims a line the pixels do not show is not, and volume does not
compensate for it.
"""
