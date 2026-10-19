import copy
import json

import pytest

from slm_lab.held_apps_native_shards_v1 import select, reconcile, FORMAT
from slm_lab.corpus import file_hash


def records():
    return [dict(id=str(i), publisher_split='test', purpose='evaluation_only', optimization_forbidden=True,
        supported_by_declared_local_protocol=True, complete_prompt_tokens=80+i, difficulty=['introductory','interview','competition'][i%3],
        mode=['function','stdin'][i%2], generation_budget=1024, full_context_window=4096, input_or_case_clipping_permitted=False)
        for i in range(90)]


def test_source_only_stratified_selection_is_fixed_distinct_and_keeps_original_budget():
    rows = records(); chosen = select(rows)
    assert len(chosen) == len({r['id'] for r in chosen}) == 64
    assert chosen == select(list(reversed(rows)))
    assert {(r['difficulty'],r['mode']) for r in chosen} == {(r['difficulty'],r['mode']) for r in rows}
    assert all(r['generation_budget'] == 1024 for r in chosen)


def test_train_roles_changed_budgets_and_incomplete_cohorts_reject():
    for change in [dict(publisher_split='train'), dict(optimization_forbidden=False),
                   dict(generation_budget=512), dict(input_or_case_clipping_permitted=True)]:
        rows = records(); rows[0].update(change)
        with pytest.raises(ValueError): select(rows)
    with pytest.raises(ValueError): select(records()[:10])


def test_all_shards_required_and_duplicated_or_wrong_actual_cohort_reject(tmp_path):
    cohort = tmp_path/'cohort.json'; cohort.write_text(json.dumps(dict(selected_original_test_ids=['a','b','c','d'], manifest_sha256='b'*64)))
    paths = []
    for rank, identity in enumerate(['a','b','c','d']):
        shard = dict(format=FORMAT, rank=rank, world=4, checkpoint_sha256='a'*64,
            cohort_sha256=file_hash(cohort), original_manifest_sha256='b'*64, generation_budget=1024,
            inference_device_type='cpu', inference_precision='float32', optimization_forbidden=True,
            results=[dict(id=identity, stopped_on_EOS=True, execution=dict(all_finite_published_cases_pass=False,
                syntax_valid=True, all_cases_observed=True))])
        path = tmp_path/f'{rank}.json'; path.write_text(json.dumps(shard)); paths.append(path)
    reconcile(paths, cohort, tmp_path/'complete.json')
    with pytest.raises(ValueError): reconcile(paths[:3], cohort, tmp_path/'missing.json')
    changed = json.loads(paths[3].read_text()); changed['results'][0]['id'] = 'a'; paths[3].write_text(json.dumps(changed))
    with pytest.raises(ValueError): reconcile(paths, cohort, tmp_path/'duplicate.json')
    changed['results'][0]['id'] = 'd'; paths[3].write_text(json.dumps(changed))
    cohort.write_text(json.dumps(dict(selected_original_test_ids=['a','b','c','d'], manifest_sha256='c'*64)))
    with pytest.raises(ValueError): reconcile(paths, cohort, tmp_path/'wrong-cohort.json')
