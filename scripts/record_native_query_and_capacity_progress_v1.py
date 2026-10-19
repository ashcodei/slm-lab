"""Keep the full goal open while recording real standalone usability checks."""
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
import own_training_gpu_lease_v2 as owner

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/native-query-and-capacity-v1-review'


def read(path):return json.loads(Path(path).read_text())


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only full-scope review required')
    master_path=ROOT/'reports/completion-audit.json';master=read(master_path);previous=ROOT/master['latest_full_scope_review']
    requirements=deepcopy(read(previous)['requirements']);runtime=ROOT/'reports/public-pretraining-recovery-v1'
    main_plan=read(runtime/'plan.json');owner.REPORTS=runtime;owner.resources.REPORTS=runtime;owner.owned(main_plan)
    ranks=[r for r in read(runtime/'children.json') if r['subject'].startswith('train-rank-') and not terminal(r)]
    handoff=read(ROOT/'reports/instance-lineage-recovery-v2/complete.json')
    if len(ranks)!=4 or any(terminal(r['actual_owner']) for r in handoff['actual_ten_new_controllers']):
        raise ValueError('All original GPU ranks and full human/RL/evaluation controllers must remain live')
    smoke_path=ROOT/'reports/own-native-query-full-CPU-v1/complete.json';smoke=read(smoke_path)
    if (not smoke['complete'] or smoke['actual_registered_neural_parameters']!=984399690 or smoke['actual_generated_native_actions']!=1
            or not smoke['no_new_generated_quality_metric'] or '6 passed' not in (ROOT/'reports/own-native-query-v1-tests.log').read_text()):
        raise ValueError('Completed actual full-size one-action interface proof and all actual unit controls required')
    image_path=ROOT/'reports/own-native-query-image-full-CPU-v1/complete.json';image=read(image_path)
    if (not image['complete'] or image['actual_registered_neural_parameters']!=984399690 or image['actual_image_size']!=448
            or not image['original_Encoder_and_frontend_image_preprocessing_bit_exact'] or not image['no_new_generated_quality_metric']):
        raise ValueError('Completed actual full-size448px image-interface and exact original preprocessing proof required')
    research_root=ROOT/'reports/full-function-repair-comparison-CPU-v1';research=read(research_root/'plan.json')
    if terminal(research['owner']) or (research_root/'failure.json').exists():raise ValueError('Actual original research owner must still be live without a failure')
    capacities={};paths=[Path(__file__),previous,smoke_path,image_path,ROOT/'reports/own-native-query-v1-tests.log',research_root/'plan.json',
        runtime/'plan.json',ROOT/'reports/instance-lineage-recovery-v2/complete.json']
    for profile in research['profiles']:
        path=research_root/('capacity-'+profile)/'complete.json'
        if path.exists():
            actual=read(path)
            if (not actual['complete'] or not actual['every_actor_Adam_metric_and_private_RNG_next_update_bit_exact']
                    or not actual['cold_export_import']['actual_CPU_BF16_function_logits_bit_exact']):
                raise ValueError('Every claimed completed actual profile must pass next-step and cold proofs')
            capacities[profile]=actual;paths.append(path)
    targets=[master_path,ROOT/'README.md',ROOT/'RESEARCH.md'];args=['rg','-l','-F','-g','*.json']
    for path in targets:args.extend(['-e','"'+str(path)+'"','-e','"'+str(path.relative_to(ROOT))+'"'])
    scan=subprocess.run([*args,'reports'],cwd=ROOT,text=True,capture_output=True)
    if scan.returncode not in [0,1]:raise ValueError(scan.stderr)
    protected=set()
    for name in scan.stdout.splitlines():
        value=json.loads((ROOT/name).read_text(),parse_int=str)
        if isinstance(value,dict):protected.update(str(Path(p).resolve()) for p in value.get('source_files_sha256',{}))
    if any(str(path) in protected for path in targets):raise ValueError('Source-bound root summaries cannot be edited')
    prefix=[]
    for line in (runtime/'training-metrics.jsonl').read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'):break
        prefix.append(line)
    rows=[json.loads(line) for line in prefix]
    if [r['step'] for r in rows]!=list(range(1,len(rows)+1)):raise ValueError('The complete actual main source prefix must remain contiguous')
    at=datetime.now(timezone.utc).isoformat();OUT.mkdir();(OUT/'captured-main-training-prefix.jsonl').write_bytes(b''.join(prefix))
    snapshot=read(research_root/'progress.json')
    runtime_snapshot=dict(at=at,actual_main_updates=len(rows),planned_main_updates=4447,
        actual_main_response_EOS_targets=int(sum(r['response_EOS_targets'] for r in rows)),actual_four_training_ranks=ranks,
        actual_GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        actual_research_owner=research['owner'],research_progress=snapshot,completed_actual_full_size_capacity_profiles=list(capacities),
        all_original_ten_downstream_controllers_live=True,new_interface_updates=0,new_interface_quality_scores=0,
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress')
    atomic_json(OUT/'runtime.json',runtime_snapshot)
    requirements['own_architecture']['evidence']+=' A new CPU native query interface supports our complete independently stored dense and policy-folded formats, with optional image input and no external model or repair service. This adds no learned parameters.'
    requirements['under_1b_parameters']['evidence']+=' The fresh actual native CLI load independently counts984399690registered parameters and generates one real native action using the fixed own7558-update standalone file, without an old base/actor/optimizer checkpoint.'
    requirements['serious_full_model_training']['evidence']=(f'The original four-L40 public/raw/function/retention epoch has {len(rows)}/4447 complete actual updates and '
        f'{runtime_snapshot["actual_main_response_EOS_targets"]} response/EOS targets in this captured prefix. '
        f'The separate actual999706442parameter full-source repair comparison remains live in phase {snapshot["phase"]}. '
        f'Completed full-size capacity profiles are {list(capacities)}; each has2logical/3physical updates, every selected Adam/actor/metric/private-RNG next update bit exact, '
        'corrupt source/RNG rejection before mutation, complete long/tail backward probes and exact cold full-context CPU BF16 logits. '
        'All three profiles must pass before the fixed306updates/4851presentations/2287110original targets per arm. The new interface smoke adds no updates or quality score.')
    requirements['high_end_llm_capability']['evidence']+=' The new one-action Hello query proves standalone usability only; it is deliberately not scored as a language or coding accuracy improvement.'
    requirements['text_and_images']['evidence']+=' The new query frontend passes actual tiny-neural text/image-output checks and separate actual984399690parameter text and448px image native-generation smoke queries. The full-size image pixels rebuild bit exactly through the original training Encoder. The synthetic image has no expected semantic answer and is not a training/validation/benchmark case, so this proves interface execution only.'
    requirements['gpu_runtime']['evidence']+=' The one-off native interface smoke uses one low-priority CPU thread and no visible GPU, leaving every original GPU owner unchanged.'
    for value in requirements.values():value['status']='incomplete_for_requested_final_model'
    if len(requirements)!=10:raise ValueError('All ten original final requirements must remain in scope')
    paths.extend([OUT/'runtime.json',OUT/'captured-main-training-prefix.jsonl'])
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,source_files_sha256={str(p):file_hash(p) for p in paths},
        previous_full_scope_review=str(previous.relative_to(ROOT)),previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        original_full_scope_preserved=True,all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False))
    (OUT/'previous-completion-audit.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:v['evidence'] for k,v in requirements.items()},latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),
        latest_actual_main_runtime=runtime_snapshot,all_ten_final_requirements_incomplete=True,current_goal_turn_classification='progress',candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Native standalone query interface, **{at}**: a new CPU command loads our own complete model from **model.pt + tokenizer.json**, '
        'accepts text or a local image, preserves the complete prompt, and returns the actual native response without repair. '
        '**Six tests pass**, including actual small-neural text/image output, source/checksum/context guards, both standalone formats and the CLI. '
        'Fresh **984,399,690-parameter** full-size loads complete actual text and448px image-generation actions with independently supplied model/tokenizer/source identities. '
        'The image preprocessing matches the original training Encoder bit exactly. '
        '**This is usability evidence, not a quality score.** The last whole-code-epoch coding result remains0/48functional. '
        f'The same four GPUs have {len(rows)}/4,447updates; completed CPU comparison capacity profiles are {list(capacities)}. '
        'All original human/RL/final-evaluation queues remain live. **All ten final requirements remain incomplete.** '
        '[Actual full-size query](reports/own-native-query-full-CPU-v1/complete.json), '
        '[actual448px query](reports/own-native-query-image-full-CPU-v1/complete.json), '
        '[full review](reports/native-query-and-capacity-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for path in [ROOT/'README.md',ROOT/'RESEARCH.md']:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    usage=('\n\nUse the current self-contained model from `/home/jovyan/slm-lab`:\n\n'
        '```bash\nCUDA_VISIBLE_DEVICES=\'\' .venv/bin/python -m slm_lab.own_native_query_v1 \\\n'
        '  --checkpoint checkpoints/code-first-continuation-v3/standalone/model.pt \\\n'
        '  --prompt "Write a Python function to add two integers." --max-new-tokens 128\n```\n\n'
        'Add `--image /absolute/path/to/image.png --image-size 448` for an image question, '
        '`--raw-prompt` for verbatim completion, or `--json` for native token and runtime records. '
        '`--inspect` checks the artifact identity and declared metadata without constructing the full neural model. '
        'The named checkpoint is the completed code epoch; ongoing public pretraining writes separate resumable states. '
        'The interface does not establish improved model accuracy.\n')
    path=ROOT/'README.md';path.write_text(path.read_text()+usage)
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_main_updates=len(rows),completed_capacity_profiles=list(capacities),all_ten_incomplete=True)),flush=True)


if __name__=='__main__':main()
