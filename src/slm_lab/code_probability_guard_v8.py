"""Runtime rounding guard tied to the actual full-size own-weight diagnostic."""
from dataclasses import asdict
import json
from pathlib import Path

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash


def load(root, model):
    path = Path(root)/'reports/code-probability-rounding-v1/admission.json'
    data = json.loads(path.read_text())
    if (data.get('format') != 'own-actual-FP32-cache-full-rounding-guard-v1'
            or data.get('actual_counts') != model.counts() or data.get('config') != asdict(model.config)
            or data.get('full_size_four_L40_diagnostic_used') is not True
            or data.get('raw_vector_and_original_logit_aggregates_independently_reconstructed') is not True
            or data.get('maximum_observed_FP64_cache_full_logp_difference',1.) > 1e-8
            or not 0 < data.get('admitted_absolute_action_logp_bound',1.) <= 1e-3
            or data.get('guard_is_runtime_fail_closed_not_a_future_error_or_accuracy_certificate') is not True
            or data.get('production_use_requires_fresh_full_profile_GPU_capacity') is not True):
        raise ValueError('Exact same-architecture full-size four-L40 numerical calibration and bounded runtime guard required')
    FileBindings(data['source_files_sha256']).check(full=True)
    return dict(bound=data['admitted_absolute_action_logp_bound'],sha256=file_hash(path))
