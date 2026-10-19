"""Complete public-code source/score readback and reexecution of every pass."""
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from slm_lab.code_quality import execute
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/whole-human-SFT-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError('Public code reader must explicitly hide GPUs')
    output = REPORTS/'independent-public-comparison-v1.json'
    if output.exists(): raise ValueError('Preserve the first complete public readback')
    source_path = ROOT/'data/public-code-assessment-v1/questions.jsonl'
    questions = {r['id']:r for r in map(json.loads,source_path.open())}
    if len(questions)!=175: raise ValueError('Every original supported development problem required')
    sources = {str(source_path):file_hash(source_path),str(Path(__file__).resolve()):file_hash(__file__)}
    summaries = {}
    for arm in ('initial','rate_5e5'):
        rows = []; passing = []; executions = []
        for shard in range(4):
            path = REPORTS/arm/f'public-{shard}.json'; sources[str(path)] = file_hash(path)
            d = json.loads(path.read_text()); records = d['results']; rows.extend(records)
            shard_source = ROOT/f'reports/own-code-delta-public-v1/inputs/shard-{shard}/questions.jsonl'
            sources[str(shard_source)] = file_hash(shard_source)
            originals = list(map(json.loads,shard_source.open()))
            if (d['data_sha256']!=file_hash(shard_source) or d['supported_problems']!=len(records)
                    or {r['id'] for r in originals}!={r['id'] for r in records}
                    or any(questions[r['id']]!=r for r in originals)
                    or round(d['syntax_rate_on_supported_subset']*len(records))!=sum(r['execution'].get('syntax_valid',False) for r in records)
                    or round(d['pass_at_1_on_supported_subset']*len(records))!=sum(r['execution'].get('functional_passed',False) for r in records)):
                raise ValueError('Every original shard source and finite-case/syntax aggregate must reconstruct')
        if len(rows)!=175 or {r['id'] for r in rows}!=set(questions): raise ValueError('Each original supported problem exactly once required')
        for row in rows:
            original = questions[row['id']]
            try: ast.parse(row['code']); parsed = True
            except SyntaxError: parsed = False
            if parsed!=row['execution'].get('syntax_valid',False): raise ValueError('Every delivered whole-program syntax outcome must reconstruct')
            if row['execution'].get('functional_passed',False):
                actual = execute(dict(language='Python',code=row['code'],function=original['function'],cases=original['cases'],stress=[]))
                if not actual.get('functional_passed') or actual['passed_cases']!=len(original['cases']):
                    raise ValueError('Every delivered passing program must pass all complete original cases again')
                passing.append(row['id']); executions.append(dict(id=row['id'],code=row['code'],execution=actual,original_cases=len(original['cases'])))
        summaries[arm] = dict(supported_problems=175,syntax=sum(r['execution'].get('syntax_valid',False) for r in rows),
            finite_case_passes=len(passing),independently_reexecuted_passing_ids=passing,passing_programs_and_actual_executions=executions)
    before,after = (set(summaries[a]['independently_reexecuted_passing_ids']) for a in ('initial','rate_5e5'))
    if any(file_hash(p)!=h for p,h in sources.items()): raise ValueError('Original source/reports changed during readback')
    atomic_json(output,dict(observed_at=datetime.now(timezone.utc).isoformat(),summaries=summaries,
        every_original_problem_and_delivered_syntax_reconstructed=True,all_candidate_passes_reexecuted_on_all_original_cases=True,
        gained_passing_ids=sorted(after-before),lost_passing_ids=sorted(before-after),retained_passing_ids=sorted(before&after),
        source_files_sha256=sources,native_CUDA_BF16_inference_not_replayed=True,
        complete_native_math_and_final_training_reconciliation_still_pending=True,
        supported_public_development_subset_not_all_code_or_independent_final_benchmark=True,
        candidate_promoted=False,goal_achieved=False))
    print({arm:{k:v for k,v in record.items() if k!='passing_programs_and_actual_executions'} for arm,record in summaries.items()})


if __name__=='__main__': main()
