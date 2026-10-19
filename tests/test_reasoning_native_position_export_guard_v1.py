"""Reject the actually observed old-position source before native packaging."""
import json
from pathlib import Path

import pytest
import torch

from slm_lab.corpus import file_hash
from slm_lab.parallel_repair_durable_state_v1 import read_latest
from slm_lab.public_reasoning_train_data_v1 import PublicReasoningTrainData
from slm_lab.reasoning_native_export_v2 import export
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_batched_reasoning_capacity_controller_v2 import make_student

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/public-reasoning-curriculum-v1'
PROOFS = ROOT / 'reports/batched-reasoning-training-controller-v1-pytest/test_actual_four_replica_capac0'


@pytest.mark.parametrize('profile', ['anchor_CE', 'anchor_RL'])
def test_actual_uninstalled_trained_small_actor_cannot_claim_canonical_native_position_protocol(profile, tmp_path):
    torch.set_num_threads(1)
    data = PublicReasoningTrainData(DATA, ROOT / 'reports/public-reasoning-curriculum-v1-readback-admission/complete.json')
    state, _, _ = read_latest(PROOFS / profile / 'complete/durable')
    receipt = json.loads((PROOFS / (profile + '-rank-0.json')).read_text())['actual_complete_training']
    model = make_student(data).eval()
    model.import_export(state['model'], expected_sources=state['model']['sources'])
    before, rng = tensor_digest(model.named_parameters()), torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='deterministic native vision/word position protocol'):
        export(model, state, receipt, sources=state['model']['sources'], tokenizer_sha256=file_hash(DATA / 'tokenizer.json'),
            working_budget_bytes=256 * 1024**2, allow_mechanics=True)
    assert tensor_digest(model.named_parameters()) == before and torch.equal(torch.get_rng_state(), rng)
    (tmp_path / 'actual-old-position-native-export-rejection.json').write_text(json.dumps(dict(profile=profile,
        actual_small_research_values=model.counts()['actual_unique'], actual_original_complete_training_groups=3,
        actual_old_position_actor_rejected_before_fold_or_packaging=True, no_source_weight_or_RNG_mutation=True,
        new_neural_forwards=0, optimizer_updates=0, actual_full_size_GPUadmission=False), indent=2))
