"""Prepare fresh operational routes without mutating a live/frozen controller."""
import json
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
MAP=ROOT/'reports/function-repair-GPU-lineage-v1-file-map.json'
HUMAN_IMPORT='from slm_lab.function_repair_GPU_contract_v1 import study_released\n'
HUMAN_GATE='def previous_ready(identity):\n    if not study_released(ROOT):return False\n'
HUMAN_BINDINGS="additions = [Path(__file__), ROOT/'src/slm_lab/function_repair_GPU_contract_v1.py', ROOT/'tests/test_function_repair_GPU_lineage_v1.py', ROOT/'reports/function-repair-GPU-lineage-v1-tests.log',"


def replacements(root=ROOT):
    old=json.loads((root/'reports/instance-lineage-recovery-v2-file-map.json').read_text())
    routes=[];literals={}
    for i,row in enumerate(old['stages']):
        new=dict(stage=row['stage'],previous_script=row['recovery_script'],previous_queue=row['queue'],
            recovery_script=Path(row['original_script']).stem+'_repair_gate_v1.py',
            queue='reports/'+row['stage']+'-repair-gate-v1-queue.json')
        literals[row['queue']]=new['queue'];routes.append(new)
    literals['reports/broad-anchored-code-v7-recovery-v2-held-APPS-cohort.json']='reports/broad-anchored-code-v7-repair-gate-v1-held-APPS-cohort.json'
    literals['reports/AI2D-fixed-exposure-strata-recovery-v1']='reports/AI2D-fixed-exposure-strata-repair-gate-v1'
    return routes,literals


def transform(text,literals,*,human=False):
    for before,after in literals.items():text=text.replace(before,after)
    if human:
        if text.count('def previous_ready(identity):\n')!=1 or text.count('additions = [Path(__file__),')!=1:
            raise ValueError('Exact existing human gate and immutable source-binding anchor required')
        text=text.replace('from slm_lab.codenet_checkpoint_v1 import FileBindings\n',
            'from slm_lab.codenet_checkpoint_v1 import FileBindings\n'+HUMAN_IMPORT)
        text=text.replace('def previous_ready(identity):\n',HUMAN_GATE)
        text=text.replace('additions = [Path(__file__),',HUMAN_BINDINGS)
    return text


def main():
    if MAP.exists():raise ValueError('Never overwrite a prepared operational lineage')
    routes,literals=replacements()
    if any((ROOT/'scripts'/r['recovery_script']).exists() or (ROOT/r['queue']).exists() for r in routes):
        raise ValueError('Every replacement source and actual queue must be fresh')
    files={}
    for i,row in enumerate(routes):
        source=ROOT/'scripts'/row['previous_script'];target=ROOT/'scripts'/row['recovery_script']
        target.write_text(transform(source.read_text(),literals,human=i==0));files[str(source)]=file_hash(source);files[str(target)]=file_hash(target)
    atomic_json(MAP,dict(stages=routes,permitted_literal_replacements=literals,source_files_sha256=files,
        only_first_human_readiness_has_additional_optional_study_restoration_gate=True,
        every_original_neural_function_factory_and_full_benchmark_population_preserved=True,goal_achieved=False))


if __name__=='__main__':main()
