from dataclasses import asdict
import copy
import json

import pytest
import torch
from tokenizers import Tokenizer, models

from slm_lab.corpus import file_hash
from slm_lab.frozen_base_pointer_inference_v1 import export_payload, load
from slm_lab.frozen_base_pointer_stage_v1 import FrozenBasePointerStage, make_optimizer
from test_frozen_base_pointer_stage_v1 import own_head, settings
from test_deterministic_vision_resize_v1 import ResizeFixture


def test_trained_reference_only_head_reloads_exact_predictions_and_rejects_false_original_base(tmp_path):
    torch.set_num_threads(1); head = own_head(); opts = settings()
    token = tmp_path / 'tokenizer.json'
    Tokenizer(models.WordLevel({'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3,
        **{str(i): i for i in range(4, 32)}}, unk_token='<unk>')).save(str(token))
    base = tmp_path / 'base.pt'
    torch.save(dict(model=head.base.state_dict(), config=asdict(head.base.config), tokenizer_sha256=file_hash(token)), base)
    plan = dict(training_options=opts, head_rank=8, base_checkpoint_sha256=file_hash(base), data_admission_sha256='a' * 64)
    run_plan = tmp_path / 'plan.json'; run_plan.write_text(json.dumps(plan))
    sources = dict(base_checkpoint_sha256=file_hash(base), tokenizer_sha256=file_hash(token),
                   run_plan_sha256=file_hash(run_plan), data_admission_sha256='a' * 64)
    kernel = FrozenBasePointerStage(head, make_optimizer(head, opts), ResizeFixture(), opts, sources, 'cpu')
    for _ in range(kernel.steps):
        kernel.update()
    checkpoint = tmp_path / 'head.pt'
    payload = export_payload(head, opts, sources, kernel.completed, kernel.frozen_base_digest)
    torch.save(payload, checkpoint)
    kwargs = dict(expected_head_sha256=file_hash(checkpoint), expected_base_sha256=file_hash(base),
                  expected_tokenizer_sha256=file_hash(token), expected_plan_sha256=file_hash(run_plan),
                  expected_data_admission_sha256='a' * 64)
    actual, metadata = load(checkpoint, base, token, run_plan, **kwargs)
    ids = torch.tensor([[1, 4, 7, 2]]); pixels = torch.linspace(-1, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
    head.eval(); actual.eval()
    with torch.no_grad():
        assert torch.equal(head(ids, pixels)['logits'], actual(ids, pixels)['logits'])
        assert torch.equal(head.base.generate(ids, pixels, max_new_tokens=3), actual.base.generate(ids, pixels, max_new_tokens=3))
    assert metadata['entire_original_backbone_frozen'] and len(payload['projections']) == 2
    assert checkpoint.stat().st_size < 32 * 1024
    bad = copy.deepcopy(payload); bad['sources']['base_checkpoint_sha256'] = '0' * 64
    wrong = tmp_path / 'wrong.pt'; torch.save(bad, wrong)
    with pytest.raises(ValueError, match='frozen-backbone'):
        load(wrong, base, token, run_plan, **{**kwargs, 'expected_head_sha256': file_hash(wrong)})
