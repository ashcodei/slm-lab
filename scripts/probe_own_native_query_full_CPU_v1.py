"""One real full-size text-interface smoke query; no quality score or updates."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/own-native-query-full-CPU-v1'
BASE=ROOT/'checkpoints/code-first-continuation-v3/standalone/model.pt'
EXPECTED='759bcd549a6226a1cae00a5dd2eb0968355b445ad7d77629c84a6b3a9ec6e496'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh GPU-hidden full-size inference smoke required')
    tests=ROOT/'reports/own-native-query-v1-tests.log'
    if '6 passed' not in tests.read_text():raise ValueError('Actual native text/image/identity/context/cold/CLI controls required')
    source=ROOT/'reports/completed-code-public-function-rollouts-CPU-v1/plan.json'
    mapping=ROOT/'reports/own-native-query-v1-smoke-input-sources.json'
    if json.loads(mapping.read_text())!=json.loads(source.read_text())['expected_standalone_sources']:
        raise ValueError('Externally recorded complete source map must remain exact')
    paths=[BASE,BASE.parent/'tokenizer.json',source,mapping,tests,
        *python_sources(ROOT,[Path(__file__),ROOT/'src/slm_lab/own_native_query_v1.py',ROOT/'tests/test_own_native_query_v1.py'])]
    files={str(p):file_hash(p) for p in paths}
    if files[str(BASE)]!=EXPECTED:raise ValueError('The exact own completed7558-update full standalone source is required')
    guard=FileBindings(files);OUT.mkdir()
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
        source_files_sha256=files,CPU_threads=1,nice=os.getpriority(os.PRIO_PROCESS,0),max_new_tokens=1,raw_prompt='Hello',
        fixed_original_model_not_new_repair_actor=True,no_training_or_optimizer=True,no_quality_evaluation=True,
        no_GPU_allocation=True,no_original_training_or_evaluation_job_modified=True,goal_achieved=False))
    command=[sys.executable,'-m','slm_lab.own_native_query_v1','--checkpoint',str(BASE),'--expected-sha256',EXPECTED,
        '--expected-tokenizer-sha256',files[str(BASE.parent/'tokenizer.json')],'--expected-sources',str(mapping),
        '--prompt','Hello','--raw-prompt','--max-new-tokens','1','--threads','1','--precision','bf16','--json']
    start=time.perf_counter()
    with (OUT/'actual-query.json').open('x') as output,(OUT/'stderr.log').open('x') as errors:
        subprocess.run(command,cwd=ROOT,stdout=output,stderr=errors,check=True)
    actual=json.loads((OUT/'actual-query.json').read_text())
    if (actual['actual_registered_parameters']!=984399690 or actual['actual_native_actions']!=1
            or len(actual['actual_generated_token_ids'])!=1 or actual['completed_checkpoint_updates']!=7558
            or actual['checkpoint_sha256']!=EXPECTED or actual['inference_device']!='cpu'
            or actual['inference_precision']!='bf16' or not actual['source_map_verified_against_external_input']):
        raise ValueError('Actual full-size independent own native text-interface proof must agree')
    guard.check(full=True)
    atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_registered_neural_parameters=984399690,
        actual_generated_native_actions=1,actual_native_response=actual['response'],
        actual_full_size_self_contained_text_generation_without_ancestor_optimizer_or_teacher=True,
        end_to_end_seconds=time.perf_counter()-start,actual_generation_seconds=actual['generation_seconds'],
        actual_complete_source_map_and_model_tokenizer_identities_verified=True,
        image_interface_exercised_on_small_actual_neural_fixture_only=True,
        new_training_updates=0,no_new_generated_quality_metric=True,no_existing_job_modified=True,
        source_files_sha256={**files,str(OUT/'plan.json'):file_hash(OUT/'plan.json'),str(OUT/'actual-query.json'):file_hash(OUT/'actual-query.json')},goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_neural_parameters=984399690,generated_actions=1,no_quality_metric=True)),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        target=OUT/'failure.json' if OUT.exists() else ROOT/'reports/own-native-query-full-CPU-v1-parent-failure.json'
        atomic_json(target,dict(error=repr(error),traceback=traceback.format_exc(),goal_achieved=False));raise
