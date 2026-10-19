"""Record actual admitted sources, complete queue handoff and unfinished quality."""
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/public-pretraining-lineage-progress-v1-review'
STAGES=['code-first-continuation-v3','public-pretraining-v1','phase-human-preference-v8','broad-anchored-code-v7',
    'public-pretraining-final-standalone-CPU-v1','full-held-QA-pipeline-v5','full-AI2D-pipeline-v5',
    'full-original-MBPP-pipeline-v5','AI2D-fixed-exposure-strata-v5','full-machine-preference-v2',
    'machine-preference-final-standalone-CPU-v2','machine-standalone-full-assessment-v2',
    'public-billion-causal-candidate-v2-full-readback-v2','public-pretraining-sources-CPU-v3','public-pretraining-sources-CPU-v4']


def read(path):return json.loads(Path(path).read_text())


def live(identity):
    if terminal(identity):raise ValueError('Exact active queue/training birth must remain live for this snapshot')
    fields=(Path('/proc')/str(identity['pid'])/'stat').read_text().split(') ',1)[1].split()
    return dict(identity,state=fields[0])


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-hidden full-scope review required')
    master_path=ROOT/'reports/completion-audit.json';master=read(master_path);previous=ROOT/master['latest_full_scope_review']
    requirements=deepcopy(read(previous)['requirements']);controllers=[];protected=set()
    for name in STAGES:
        p=ROOT/'reports'/name/'plan.json'
        if not p.exists():p=ROOT/'reports'/(name+'-queue.json')
        plan=read(p);controllers.append(dict(stage=name,owner=live(plan['owner'])));protected.update(plan['source_files_sha256'])
        if (ROOT/'reports'/name/'failure.json').exists():raise ValueError('No active source/training failure may be hidden')
    handoff_path=ROOT/'reports/public-pretraining-lineage-handoff-v1/complete.json';handoff=read(handoff_path)
    if (not handoff['complete'] or len(handoff['retired_unstarted_controllers'])!=10
            or any(not terminal(p) for p in handoff['retired_unstarted_controllers'])
            or not handoff['original_training_parent_and_all_workers_not_signalled']):raise ValueError('Actual successful exact unstarted-controller handoff required')
    source=ROOT/'reports/code-first-continuation-v3';progress=read(source/'progress.json')
    ranks=[live(p) for p in read(source/'children.json') if p['subject'].startswith('train-rank-')]
    if len(ranks)!=4:raise ValueError('The original four exact production ranks must remain active')
    function_path=ROOT/'reports/direct-algorithm-function-contracts-v1-readback-admission/complete.json';function=read(function_path)
    raw_path=ROOT/'reports/public-billion-causal-candidate-CPU-v3/complete.json';raw=read(raw_path)
    reader=read(ROOT/'reports/public-billion-causal-candidate-v2-full-readback-v2/progress.json')
    if (function.get('dataset_admitted_to_function_CE_training') is not True
            or function['actual_verified_roles']['train_functions']!=826
            or function['actual_verified_roles']['valid_functions']!=15
            or raw['actual_train_targets']!=1000085544 or raw['all_training_quotas_satisfied'] is not True
            or function['neural_training_updates']!=0):raise ValueError('Every actual full-source collection and admitted function count must reconcile')
    failures=[ROOT/'reports'/n/'failure.json' for n in ['public-billion-causal-candidate-CPU-v2',
        'public-billion-causal-candidate-v2-full-readback-v1','public-pretraining-sources-CPU-v1','public-pretraining-sources-CPU-v2']]
    if any(not p.exists() for p in failures):raise ValueError('The metadata failure and strict original downstream failures must remain explicit')
    if 'optimization_admitted' not in read(failures[0])['error']:raise ValueError('Only the actual duplicate metadata field was finalized')
    tests={'git-function-readback-v1-final2-tests':4,'public-pretraining-data-v2-tests':2,
        'public-pretraining-source-preparation-v2-tests':2,'public-pretraining-native-v1-final-tests':2,
        'public-billion-metadata-v1-tests':2,'public-pretraining-source-recovery-v1-tests':2,
        'public-pretraining-run-contract-v1-final-tests':3,'public-pretraining-driver-v1-tests':3,
        'public-pretraining-human-policy-v3-tests':2,'public-pretraining-broad-policy-v3-tests':2,
        'public-pretraining-machine-policy-v2-tests':2,'public-pretraining-lineage-drivers-v1-final2-tests':16}
    for name,count in tests.items():
        if str(count)+' passed' not in (ROOT/'reports'/(name+'.log')).read_text():raise ValueError('All current actual source/factory/driver controls required')
    loss0=read(source/'held-packed-initial.json');loss6144=read(source/'held-packed-6144.json')
    for key in ['selected_original_positions','complete_causal_targets','original_manifest_sha256','evaluated_windows']:
        if loss0[key]!=loss6144[key]:raise ValueError('The exact earlier held loss population must remain matched')
    loss=dict(initial=loss0['token_mean_NLL'],step6144=loss6144['token_mean_NLL'],targets=loss6144['complete_causal_targets'],
        windows=loss6144['evaluated_windows'],relative_reduction=1-loss6144['token_mean_NLL']/loss0['token_mean_NLL'],
        producer_observations_no_fresh_neural_readback=True,not_generated_code_correctness=True)
    notes={
        'own_architecture':'Own scratch-origin dense text/image backbone, original BPE and byte/native word heads remain. The new full-weight raw/function CE stage removes the training-only neural reference forward; image and word updates keep the language decoder fixed. It adds no parameters, external weights or live teacher. The source, exact all-worker recovery and own standalone format are implemented and CPU-tested; no globally novel architecture or frontier capability claim is established.',
        'public_data':'The complete larger corpus now contains1,000,085,544train and10,148,261valid text/EOS targets across523,103whole documents:500Mcode across8languages,250Mmath and250Meducational web. The collector failed only on a duplicate optimization_admitted field in its final metadata constructor after every stream closed. A separate metadata-only finalizer reconciles all25source receipts and existing native bytes without recollection or neural training; the original exception and strict downstream failures remain. A new exact full independent source reader is active and the raw corpus remains unadmitted. The separate full pinned direct-Git function readback has completed all906eligible-candidate fresh replays twice, all1737Git blobs/1608Pythonfiles, every whole native row, masks and public-example result. It admits826train/15validfunctions,389458train/6563validtargets; the original roles remain. Finite public examples are not hidden benchmark accuracy, all-history contamination freedom or optimal complexity.',
        'under_1b_parameters':'The complete original dense model has984399690actual unique values including all text/image/word/auxiliary/critic leaves. The new native factory exports one self-contained dense student plus its own tokenizer without an ancestor needed for final neural inference. Actual small-network text/image cold-load and source/cursor corruption tests pass. Human/code/machine actor branches remain999706442values, and final folded exports984399690. New public full-size four-L40 native/capacity proof is still pending; counts and unit tests do not establish quality.',
        'text_and_images':'The new schedule preserves every original selected code/output/state and all12retention-family source occurrence, including real generic photos, visual tasks and word crops. All826new whole functions are additionally present once. Compute bucketing changes grouping without clipping or changing original targets. Actual source preparation waits for full raw admission; the GPU capacity plan requires real4096text,448photo, and real word-image backward/recovery and cold outputs. Original proper/blank/wrong visual and word assessments retain their full populations. No new image capability gain has been measured.',
        'human_feedback':'The actual new post-training queue now starts only from complete own public pretraining, then runs all85786original public human-ranked pairs in both chosen-CE and human-DPO arms, all3721original code sources in matched anchored-CE/execution-RL arms, and all1620machine pairs in matched chosen-CE/machine-DPO arms. Small actual gradient/cold-load/corruption tests verify the complete new factory chain. Human labels concern public other-model responses, not new human judgments of our outputs; machine tests are not relabelled human feedback. These epochs have not yet run on the new foundation.',
        'serious_full_model_training':f'The same original four-L40 code/state/replay epoch continues at{progress["actual_updates"]}/7558updates. An actual new full public-pretraining CPU queue and ten complete downstream replacement controllers are now live. After all replacements passed and were actually queued, only ten childless CUDA-hidden unstarted old controllers were retired through exact process descriptors; original queue/scientific files and the live training parent/ranks were preserved. The explicit handoff receipt is complete. The new stage waits for complete original training/native controls/restoration, full original-source admission and exact preparation. Nine actual modality/source capacity batches, all445Adam leaves, two actual next-step private-RNG recoveries, every cold-export value and real text/photo/word outputs must pass on all four L40s before production. New raw/function corpus neural updates remain0.',
        'high_end_llm_capability':f'NOT ACHIEVED. The latest completed matched native code assessment remains0/48functional at both initial andstep2048, with syntax29/48to14/48and ordinaryPython0/24at both. Step6144held packed NLL is{loss["step6144"]:.9f}versus{loss["initial"]:.9f}on the same128windows/524160targets, a{100*loss["relative_reduction"]:.3f}%reduction; these are producer neural observations, not fresh readback or generated correctness. The earlier true-EOS auxiliary experiment has no free-generation gain and worsens unoptimized train-pool NLL, so production auxiliary weight stays0. Neither source verification, a queue handoff nor lower loss satisfies the requested high-end target.',
        'broad_independent_benchmarks':'Every original eight endpoint name and original benchmark population remains, with a ninth public_pretrained endpoint added before human alignment to measure the new stage separately. Complete16961QA questions,3088AI2Dquestions under3imageconditions,600originalMBPPproblems and all original cases,4642held human pairs and3429held words under3imageconditions remain scheduled. The revised interstage checks require9*16961QA,9*3088*3visual-condition and9*600MBPPdeliveries, including every new endpoint shard; the original held-source constructors are exercised in tests. These broad final assessments are pending, not passed or inferred from code loss.',
        'gpu_runtime':'All four original NVIDIA L40training workers and their exact parent remain live. The new pretraining stage is kernel-flock gated and waits for all original children to end and ownership to be restored; no GPU worker was interrupted. Actual RAM checks account for tmpfs against the256GiBcgroup limit, keeping96GiBworking headroom and a34GiBcapacity-file upper bound. Capacity snapshots are explicitly volatile; the persistent atomic full-state resume and initial/final native exports require26GiBreserve and sufficient actual free storage. No capacity scratch is allocated yet and no new GPU capacity test or new corpus optimization has run.',
        'any_code_correctness_and_efficiency':'NOT ACHIEVED. The independently admitted whole-function data retain original complete implementations/docstrings/stdlib context and source-visible public answers, and all originally positive examples pass two additional fresh runs. This is supervised source quality rather than generated-code correctness. The actual source factory chain preserves exact prompt/response/EOS masks, held roles and source digests. The full new1Braw/function curriculum and human/execution/machine-feedback pipeline are now concretely implemented and queued. Generated syntactic, functional and measured runtime gains still require unchanged actual native assessments; finite public cases cannot prove arbitrary-code correctness or asymptotic optimality.'}
    if set(notes)!=set(requirements):raise ValueError('Every original final requirement must remain in scope')
    for name,note in notes.items():
        if requirements[name]['status']!='incomplete_for_requested_final_model':raise ValueError('No final requirement may be declared complete')
        requirements[name]['evidence']=note
    # Check every immutable plan/queue source map before touching root summary files.
    names=subprocess.check_output(['rg','--files','reports','-g','plan.json','-g','*-queue.json'],cwd=ROOT,text=True).splitlines()
    for name in names:
        record=read(ROOT/name)
        if isinstance(record,dict):protected.update(record.get('source_files_sha256',{}))
    for path in [master_path,ROOT/'README.md',ROOT/'RESEARCH.md']:
        if str(path) in protected:raise ValueError('An immutable scientific plan binds this summary file: '+str(path))
    at=datetime.now(timezone.utc).isoformat()
    runtime=dict(at=at,actual_code_first_updates=progress['actual_updates'],planned_updates=7558,
        actual_same_four_GPU_workers=ranks,all_fifteen_active_training_pipeline_source_and_preparation_controllers=controllers,
        actual_retired_unstarted_old_controllers=handoff['retired_unstarted_controllers'],
        original_GPU_parent_and_ranks_not_signalled=True,new_public_raw_and_function_neural_updates=0,
        current_raw_independent_reader=reader,
        actual_GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        actual_filesystem_observation=subprocess.check_output(['df','-h',str(ROOT),'/dev/shm'],text=True))
    OUT.mkdir();atomic_json(OUT/'runtime.json',runtime);atomic_json(OUT/'same-held-loss-producer-observations.json',loss)
    paths=[Path(__file__),previous,function_path,raw_path,handoff_path,OUT/'runtime.json',OUT/'same-held-loss-producer-observations.json',
        ROOT/'reports/public-pretraining-v1-queue.json',*failures,*[ROOT/'reports'/(n+'.log') for n in tests]]
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,previous_full_scope_review=str(previous.relative_to(ROOT)),
        source_files_sha256={str(p):file_hash(p) for p in paths},previous_goal_turn_classification='progress',
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,original_full_scope_preserved=True,
        candidate_promoted=False,goal_achieved=False))
    (OUT/'previous-completion-audit.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={n:r['evidence'] for n,r in requirements.items()},latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),
        latest_actual_main_runtime=runtime,latest_independently_admitted_Git_functions=str(function_path.relative_to(ROOT)),
        latest_complete_raw_corpus_metadata=str(raw_path.relative_to(ROOT)),current_billion_public_independent_reader='reports/public-billion-causal-candidate-v2-full-readback-v2/plan.json',
        queued_public_pretraining_source_preparation='reports/public-pretraining-sources-CPU-v4/plan.json',
        latest_public_pretraining_GPU_queue='reports/public-pretraining-v1-queue.json',latest_complete_pretraining_lineage_handoff=str(handoff_path.relative_to(ROOT)),
        next_required_work=['Finish the same original7558update GPU epoch and its complete unchanged native assessments.',
            'Finish the full independent1Braw source/native readback and exact CPUv3/v4source preparation; admit no prefix.',
            'Run the new actual full-size four-L40capacity, all-private-RNG recovery and native cold-load controls, then the complete declared new raw/function/retention epoch.',
            'Run all complete original human preference, anchored-code/execution-RL and machine comparison epochs and every original broad/native/standalone assessment at all declared endpoints.',
            'Demonstrate actual generated correct/efficient coding and broad multimodal gains before any promotion; all ten final requirements remain incomplete.'],
        all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False,current_goal_turn_classification='progress')
    atomic_json(master_path,master)
    paragraph=(f'Public pretraining and full pipeline handoff, **{at}**: the same original four-L40 training run is at '
        f'**{progress["actual_updates"]}/7,558 updates**. The full **1,000,085,544-target** raw corpus has closed; a metadata-only '
        'finalizer repaired a duplicate-field manifest failure while preserving all original token files and failure records. '
        'Its new full independent reader is active; **raw data are not yet admitted or trained**. Separately, full independent '
        'Git/source/native/fresh-example readback now admits **826 train and 15 validation whole functions**.\n\n'
        'A new full public-pretraining GPU controller and all ten downstream human-feedback/RL/evaluation controllers are '
        '**implemented, tested and actually queued**. A completed process-descriptor handoff retires only ten unstarted CPU queues; '
        '**the live training parent and four GPU ranks are untouched**. The ninth pre-alignment endpoint preserves all original '
        'benchmark populations and isolates the new pretraining stage. Full new-source admission, actual four-L40capacity/recovery '
        'and the new full source epoch remain pending. **New corpus neural updates: 0.**\n\n'
        f'Same-protocol held code NLL at step6,144 is **{loss["step6144"]:.6f}** versus **{loss["initial"]:.6f}** '
        f'(**{100*loss["relative_reduction"]:.2f}%lower**,128windows/524160targets; producer observations). '
        '**This is not generated-code accuracy:** the latest completed matched code result remains **0/48functional**, '
        'with29to14syntax passes at step2,048. **All ten final requirements remain incomplete.** '
        '[Function admission](reports/direct-algorithm-function-contracts-v1-readback-admission/complete.json), '
        '[actual new GPU queue](reports/public-pretraining-v1-queue.json), '
        '[complete lineage handoff](reports/public-pretraining-lineage-handoff-v1/complete.json), '
        '[full review](reports/public-pretraining-lineage-progress-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for path in [ROOT/'README.md',ROOT/'RESEARCH.md']:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_original_GPU_updates=progress['actual_updates'],
        complete_raw_targets=raw['actual_train_targets'],admitted_train_functions=826,new_corpus_neural_updates=0,all_ten_final_requirements_incomplete=True)),flush=True)


if __name__=='__main__':main()
