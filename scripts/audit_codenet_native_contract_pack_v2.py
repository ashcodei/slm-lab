"""Read every revised row/mask against the original independently audited pack."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_complete_context_v1 import CompleteCodeEncoder
from slm_lab.codenet_native_contracts_v1 import CONTRACT
from slm_lab.codenet_token_pack_v1 import CompleteCodePack, ROLES
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    old_folder, new_folder = ROOT / 'data/codenet-durable-pack-v1', ROOT / 'data/codenet-durable-pack-v2'
    reports = ROOT / 'reports/codenet-native-contract-pack-v2'
    result_path = reports / 'independent-readback.json'
    if result_path.exists():
        raise ValueError('Preserve earlier independent readback evidence')
    status = json.loads((reports / 'status.json').read_text())
    if (status['status'] != 'complete runtime-contract revision prepared; independent full readback and admission pending'
            or status['manifest_sha256'] != file_hash(new_folder / 'manifest.json')):
        raise ValueError('Completed exact candidate revision required')
    plan = json.loads((reports / 'plan.json').read_text())
    for path, digest in plan['input_files_sha256'].items():
        if file_hash(path) != digest:
            raise ValueError('Original revision input or implementation changed')
    old, new = CompleteCodePack(old_folder, 'train'), CompleteCodePack(new_folder, 'train')
    qpath = ROOT / 'reports/codenet-numba-original-references-v1/source-declared-aot-qualification-v1.json'
    q = json.loads(qpath.read_text())
    qualified = {r['original_archive_member']: r for r in q['source_declared_native_preparation_qualifications']}
    catalog_path = reports / 'native-contract-inventory.json'
    catalog = json.loads(catalog_path.read_text())
    if (len(qualified) != 6 or new.manifest['provenance']['native_contract_inventory_sha256'] != file_hash(catalog_path)
            or set(plan['qualified_members']) != set(qualified)):
        raise ValueError('Exact predeclared six-source contract inventory required')
    sources = [Path(__file__), ROOT / 'src/slm_lab/codenet_native_contracts_v1.py',
        ROOT / 'src/slm_lab/codenet_complete_context_v1.py', ROOT / 'src/slm_lab/codenet_token_pack_v1.py',
        ROOT / 'src/slm_lab/training.py', ROOT / 'src/slm_lab/model.py',
        ROOT / 'src/slm_lab/codenet_checkpoint_v1.py', qpath, catalog_path, reports / 'plan.json',
        old_folder / 'manifest.json', new_folder / 'manifest.json']
    sources += [folder / name for folder, pack in [(old_folder, old), (new_folder, new)]
                for name in pack.manifest['files_sha256']]
    bound = {str(p): file_hash(p) for p in sources}; guard = FileBindings(bound)
    encoder = CompleteCodeEncoder(Tokenizer.from_file(str(new_folder / 'tokenizer.json')),
                                  ModelConfig.load(new_folder / 'model-config.json'), ROOT)
    counts, changed, unchanged = Counter(), [], 0
    token_delta = 0
    if len(new.index) != len(old.index):
        raise ValueError('Entire original population must be retained')
    with gzip.open(old_folder / 'original-rows.jsonl.gz', 'rt', encoding='utf-8') as before, gzip.open(new_folder / 'original-rows.jsonl.gz', 'rt', encoding='utf-8') as after:
        for position, pair in enumerate(itertools.zip_longest(before, after)):
            if any(line is None for line in pair):
                raise ValueError('Original/revised row populations differ')
            a, b = map(json.loads, pair); member = a['original_archive_member']
            ia, ib = old.index[position], new.index[position]
            if (a['token_pack_index'] != position or b['token_pack_index'] != position
                    or ia['role'] != ib['role'] or a['split'] != b['split']):
                raise ValueError('Original complete-source order or roles differ')
            skip = {'prompt', 'complete_input_ids_sha256'}
            if any(a[k] != b.get(k) for k in set(a) - skip):
                raise ValueError('Original source bytes, labels or metadata changed')
            if hashlib.sha256(b['response'].encode()).hexdigest() != b['original_source_body_sha256']:
                raise ValueError('Original programme body digest differs')
            sa, la, ta = int(ia['offset']), int(ia['length']), int(ia['target_start'])
            sb, lb, tb = int(ib['offset']), int(ib['length']), int(ib['target_start'])
            xa, xb = old.ids[sa:sa+la], new.ids[sb:sb+lb]
            if (hashlib.sha256(xb.tobytes()).hexdigest() != b['complete_input_ids_sha256']
                    or not np.array_equal(xa[ta:], xb[tb:]) or lb - tb != b['target_tokens']):
                raise ValueError('Complete unchanged original answer/EOS target mask differs')
            if member in qualified:
                if b['split'] != 'train' or set(b) != set(a) | {'execution_contract'}:
                    raise ValueError('Only exact qualified training prompts may gain a contract')
                head, separator, statement = a['prompt'].partition('\n\n')
                if b['prompt'] != head + '\n' + CONTRACT + separator + statement:
                    raise ValueError('Original problem body or declared contract framing differs')
                expected_contract = dict(kind='source-declared-numba-aot-v1',
                    preparation_final_argument='ONLINE_JUDGE', generated_module='my_module',
                    preparation_may_continue_into_input_handling=True,
                    source_qualification_report_sha256=file_hash(qpath),
                    actual_native_preparation_verified=False, original_response_modified=False)
                if b['execution_contract'] != expected_contract:
                    raise ValueError('Runtime qualification metadata differs')
                for key in ['original_archive_member', 'original_source_body_sha256', 'source_problem_id', 'group_id', 'split']:
                    if b[key] != qualified[member][key]:
                        raise ValueError('Qualified source identity differs')
                ids, labels = encoder.sequence(b, 'response')
                if (not np.array_equal(xb, np.asarray(ids, dtype='<u2'))
                        or labels != [-100] * tb + ids[tb:]):
                    raise ValueError('Actual revised complete prompt/mask encoding differs')
                changed.append(dict(original_archive_member=member, old_context_tokens=la,
                    new_context_tokens=lb, target_tokens=lb-tb, split=b['split']))
            else:
                if set(a) != set(b) or a['prompt'] != b['prompt'] or not np.array_equal(xa, xb) or ta != tb:
                    raise ValueError('Unqualified source prompt or any original IDs changed')
                unchanged += 1
            for key, value in [('examples', 1), ('context_tokens', lb), ('target_tokens', lb-tb)]:
                counts[b['split'], key] += value
            token_delta += lb - la
            if (position + 1) % 5000 == 0:
                guard.check(); print('read_back', position+1, flush=True)
    if ({r['original_archive_member'] for r in changed} != set(qualified)
            or unchanged + len(changed) != new.manifest['source_examples'] or encoder.truncated):
        raise ValueError('All original sources and complete revised contexts required')
    actual_counts = {role: {k: counts[role, k] for k in ['examples', 'context_tokens', 'target_tokens']} for role in ROLES}
    if actual_counts != new.manifest['counts']:
        raise ValueError('Reconstructed full role/token counts differ')
    if (len(catalog['records']) != 8
            or {r['original_archive_member'] for r in catalog['records'] if r['required_contract_revision']} != set(qualified)
            or len([r for r in catalog['records'] if not r['required_contract_revision']]) != 2):
        raise ValueError('Optional/native import inventory differs')
    if any(new.manifest['files_sha256'][name] != old.manifest['files_sha256'][name]
           for name in ['tokenizer.json', 'model-config.json', 'DATASET-LICENSE.txt', 'publisher-README.md']):
        raise ValueError('Original vocabulary, architecture or licensing changed')
    guard.check(full=True)
    atomic_json(result_path, dict(completed_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=bound, original_complete_examples_read_back=unchanged+len(changed),
        entirely_unchanged_prompt_and_token_sequences=unchanged, revised_source_prompts=changed,
        added_prompt_tokens=token_delta, actual_counts=actual_counts,
        all_original_programme_bytes_and_full_answer_EOS_tokens_preserved=True,
        all_original_order_roles_groups_and_metadata_preserved=True,
        two_other_native_import_source_prompts_unchanged=True,
        original_dataset_files_modified=False, actual_native_preparation_executed=False,
        dataset_admitted_to_training=False, new_model_training_started=False, goal_achieved=False,
        scope='Exact complete-source/runtime-contract data reconstruction only. Actual compilation, broader execution, model learning, final starting-policy bindings and frontier capability remain unverified.'))
    print('full_readback_complete', unchanged+len(changed), 'six_contracts', len(changed), flush=True)


if __name__ == '__main__':
    main()
