import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from slm_lab.apps_native_metrics_v1 import summarize


def test_actual_original_domain_rows_recount_and_corrupt_or_incomplete_sources_still_reject():
    path = Path('scripts/review_own_code_training_endpoint_v2.py').resolve()
    spec = importlib.util.spec_from_file_location('checked_code_endpoint_v2', path)
    module = importlib.util.module_from_spec(spec); sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    questions = list(map(json.loads, Path('data/knowledge-eval-v1/questions.jsonl').read_text().splitlines()))
    wrapped = module.bound_summary(summarize, questions)
    report = json.loads(Path('reports/own-code-generation-repair-v1/candidate/knowledge.json').read_text())
    assert wrapped(report, 'knowledge.json')['questions'] == 559
    bad = copy.deepcopy(report); bad['predictions'][0]['generated_correct'] = not bad['predictions'][0]['generated_correct']
    with pytest.raises(ValueError, match='reproduce every domain outcome'):
        wrapped(bad, 'knowledge.json')
    with pytest.raises(ValueError, match='Every actual original domain question'):
        module.bound_summary(summarize, questions[:-1])(report, 'knowledge.json')
    with pytest.raises(ValueError, match='unique original domain-question'):
        module.bound_summary(summarize, questions + questions[:1])
