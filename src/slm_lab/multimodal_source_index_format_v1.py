"""Compact complete native/source references, including explicit corrections.

This schema describes source preparation, not neural training admission. Raw
metadata and original source bytes remain immutable and separately SHA-bound.
"""
import numpy as np

FORMAT = 'own-complete-multimodal-native-source-index-with-qualified-corrections-v1'
DTYPE = np.dtype([
    ('metadata_offset','<u8'),('metadata_bytes','<u4'),('metadata_sha256','V32'),
    ('source_offset','<u8'),('source_bytes','<u4'),('source_sha256','V32'),
    ('family','<u2'),('original_local_position','<u4'),
    ('context_tokens','<u2'),('target_start','<u2'),('response_EOS_targets','<u2'),
    ('input_ids_sha256','V32'),('correction_index','<i2'),
])
