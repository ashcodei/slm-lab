"""Check32retained real observations with the separate reader while work runs.

This is a fixed early integration check, not whole-source admission or a
quality evaluation. Original source order sorts the available observations;
the full reader still requires every one of3959original dispositions.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

from tokenizers import Tokenizer

from slm_lab.code_contests_CE_view_v1 import CodeContestsCEView
from slm_lab.code_contests_contrast_readback_v1 import reconstruct_pair, reconstruct_witness
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.source_jsonl_records_v1 import records

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/CodeContests-contrast-early-independent-readback-v1.json'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('A fresh CPU-only early integration readback is required')
    if '2 passed' not in (ROOT / 'reports/code-contests-contrast-readback-v1-tests.log').read_text():
        raise ValueError('The actual separate reconstruction controls must pass first')
    preparation = ROOT / 'reports/CodeContests-execution-contrast-candidate-v1'
    selected = sorted((preparation / 'source-observations').glob('*.json'))[:32]
    if len(selected) != 32:
        raise ValueError('At least32complete immutable original observations are required')
    evidence = {str(p): file_hash(p) for p in selected}
    evidence[str(Path(__file__))] = file_hash(__file__)
    schema_path = ROOT / 'data/code-contests-protobuf-schema-v1'
    schema = json.loads((schema_path / 'manifest.json').read_text())
    sys.path.insert(0, schema['isolated_reader_site_packages']); sys.path.insert(0, str(schema_path))
    import contest_problem_pb2 as proto
    view = CodeContestsCEView(ROOT); tokenizer = Tokenizer.from_file(str(view.tokenizer_path))
    sources = {r['id']: r for r in records(ROOT / 'data/code-contests-native-train-records-v1/all-original-problem-records.jsonl')}
    counts = Counter(); items = []
    for path in selected:
        actual = json.loads(path.read_text()); source = sources[actual['original_problem_id']]
        with open(source['extracted_record_file'], 'rb') as stream:
            stream.seek(source['extracted_record_offset']); raw = stream.read(source['extracted_record_bytes'])
        if hashlib.sha256(raw).hexdigest() != actual['original_record_sha256']:
            raise ValueError('Every entire original source record must match')
        problem = proto.ContestProblem()
        if problem.ParseFromString(raw) != len(raw):
            raise ValueError('All original source fields must parse')
        position = actual['original_reference_position']
        reference_path = ROOT / 'reports/CodeContests-full-Python-references-v1/original-problem-executions' / f'{position:05d}.json'
        reference = json.loads(reference_path.read_text())
        pair, why = reconstruct_pair(view.row(actual['original_view_position']), problem, reference, tokenizer)
        if pair is None or pair != actual['pair']:
            raise ValueError('The separate reader must rebuild both real programs, every own-BPE token and full EOS mask')
        witness, why = reconstruct_witness(pair['rejected'], problem, actual['actual_negative_execution'])
        if witness != actual['counterexample'] or why != actual['actual_disposition']:
            raise ValueError('Every actual raw negative-output disposition must independently rebuild')
        counts[why] += 1
        items.append(dict(original_reference_position=position, actual_disposition=why,
                          chosen_context=len(pair['chosen_ids']), rejected_context=len(pair['rejected_ids'])))
        evidence[str(reference_path)] = file_hash(reference_path)
    if any(file_hash(path) != digest for path, digest in evidence.items()):
        raise ValueError('An original retained observation changed during readback')
    atomic_json(OUT, dict(at=datetime.now(timezone.utc).isoformat(), complete=True,
        original_observation_positions=[r['original_reference_position'] for r in items], actual_counts=dict(counts), items=items,
        every_observed_whole_program_native_EOS_mask_and_raw_counterexample_rebuilt=True,
        sorted_first32_available_observation_files_not_full3959source_epoch=True,
        original_programs_not_rerun_and_OS_confinement_not_independently_certified=True,
        full_separate_source_readback_and_training_admission_still_pending=True,
        source_files_sha256=evidence, models_constructed=0, model_updates=0, goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT), actual_counts=dict(counts), training_admitted=False)), flush=True)


if __name__ == '__main__':
    main()
