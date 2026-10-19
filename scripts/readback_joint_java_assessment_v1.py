"""After the exact owner ends, independently compare retained complete Java reports."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exercise_java_evaluation import compare_reports

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/joint-vs-language-Java-v1'


def live(actor):
    try:
        fields=(Path('/proc')/str(actor['pid'])/'stat').read_text().split(') ',1)[1].split()
        return fields[0] not in {'Z','X'} and fields[19]==actor['ticks']
    except FileNotFoundError:
        return False


def main():
    output=REPORTS/'independent-comparison-readback.json'
    failure=REPORTS/'independent-readback-failure.json'
    if output.exists() or failure.exists():
        raise ValueError('Preserve prior complete or failed independent readback')
    plan_path=REPORTS/'plan.json'
    plan=json.loads(plan_path.read_text())
    bindings={**plan['source_files_sha256'],str(plan_path):file_hash(plan_path),str(Path(__file__)):file_hash(Path(__file__))}
    guard=FileBindings(bindings)
    try:
        while live(plan['owner']):
            guard.check()
            time.sleep(2)
        guard.check(full=True)
        before=json.loads((REPORTS/'language.json').read_text())
        after=json.loads((REPORTS/'joint.json').read_text())
        for name,report in [('language',before),('joint',after)]:
            subject=next(s for s in plan['subjects'] if s['name']==name)
            if report['checkpoint_sha256']!=subject['checkpoint_sha256'] or len(report['results'])!=35:
                raise ValueError('Exact checkpoint and complete35-module population required')
            evidence=REPORTS/(name+'-evidence')
            for field,path in [('generations_sha256',evidence/'generations.jsonl'),
                    ('generation_seal_sha256',evidence/'generation-seal.json')]:
                if file_hash(path)!=report[field]:
                    raise ValueError('Exact sealed original Java generations required')
            if not report['all35_generations_sealed_before_test_execution'] or not report['all_model_bytes_unchanged']:
                raise ValueError('Actual frozen complete generation-before-execution required')
        comparison=compare_reports(before,after)
        comparison['scope']='Own706M completed language versus completed joint; exact full35-module publisher development cases. No final-policy, universal code, asymptotic efficiency or frontier claim.'
        comparison.update(reviewed_at=datetime.now(timezone.utc).isoformat(),candidate_promoted=False,
            original_owner_terminal=True,driver_postprocessing_failure_preserved=(REPORTS/'failure.json').exists(),
            driver_known_defect='The v1 parent passes scope twice when assembling its final dict. This independent reader recomputes the comparison from both complete sealed reports; no model generation or scoring is rerun or repaired.',
            source_files_sha256=bindings,report_files_sha256={str(p):file_hash(p) for p in REPORTS.rglob('*')
                if p.is_file() and p.suffix in {'.json','.jsonl'}})
        atomic_json(output,comparison)
    except Exception as error:
        atomic_json(failure,dict(recorded_at=datetime.now(timezone.utc).isoformat(),error=str(error),
            automatic_retry=False,goal_achieved=False))
        raise


if __name__=='__main__':
    main()
