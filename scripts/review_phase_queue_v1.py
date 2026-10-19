"""Current full objective review: real RL, failed warm-up, queued full sources."""
from datetime import datetime,timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Review must hide every GPU')
    import assess_joint_java_comparison_v1 as resources
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.knowledge_evaluation import letter
    from slm_lab.math_corpus import final_answer
    from slm_lab.ocr_evaluation import transcription_scores
    def read(name):return json.loads((ROOT/name).read_text())
    run=ROOT/'reports/verified-views-execution-v1';plan=read('reports/verified-views-execution-v1/plan.json')
    queued=read('reports/phase-joint-v1-queue.json')
    if not resources.same(plan['owner']) or not resources.same(queued['owner']):raise ValueError('Actual current and queued owners must be live now')
    live=[a for a in read('reports/verified-views-execution-v1/children.json') if resources.same(a)]
    if sum(a['subject'].startswith('RL-rank-') for a in live)!=4:raise ValueError('All four actual code-RL/control workers required')
    warm=read('reports/verified-views-warmup88-readback-v1/complete.json');prefix=read('reports/execution-prefix22-readback-v1/complete.json')
    expanded=read('reports/expanded-joint-sources-v1/complete.json');cpu=read('reports/phase-joint-full-CPU-v2/complete.json')
    outcomes={};evidence=[]
    for arm in ['initial','warmup']:
        folder=run/arm;knowledge=json.loads((folder/'knowledge.json').read_text());maths=json.loads((folder/'math.json').read_text());ocr=json.loads((folder/'ocr.json').read_text())
        for row in knowledge['predictions']:
            actual=letter(row['generated_response'],len(row['candidate_scores']));best=max(range(len(row['candidate_scores'])),key=lambda i:row['candidate_scores'][i])
            if row['generated_index']!=actual or row['generated_correct']!=(actual==row['answer_index']) or row['likelihood_index']!=best:
                raise ValueError('Every actual generated/likelihood knowledge decision must reconstruct')
        for row in maths['results']:
            if row['correct']!=(final_answer(row['raw_response'])==Fraction(row['expected_numerator'],row['expected_denominator'])):
                raise ValueError('Every actual exact rational math verdict must reconstruct')
        chars=edits=0
        for row in ocr['results']:
            values=transcription_scores(row['expected'],row['generated'])
            if any(row[k]!=v for k,v in values.items()):raise ValueError('Every actual OCR character/word verdict must reconstruct')
            chars+=values['reference_characters'];edits+=values['character_edits']
        if edits/chars!=ocr['overall']['character_error_rate']:raise ValueError('Actual OCR total differs')
        outcomes[arm]=dict(code=warm['matched_full512_code_scores'][arm],knowledge_generated=sum(r['generated_correct'] for r in knowledge['predictions']),
            knowledge_likelihood=sum(r['likelihood_correct'] for r in knowledge['predictions']),knowledge_questions=len(knowledge['predictions']),
            math_correct=sum(r['correct'] for r in maths['results']),math_examples=len(maths['results']),OCR=ocr['overall'],
            reported_objects=json.loads((folder/'objects.json').read_text())['overall'],
            reported_caption_retrieval=json.loads((folder/'vision.json').read_text())['caption_retrieval'])
        evidence.extend(str((folder/(name+'.json')).relative_to(ROOT)) for name in ['knowledge','math','ocr','objects','vision'])
    progress={arm:read('reports/verified-views-execution-v1/'+arm+'-progress.json') for arm in ['verified_solution','execution_hybrid']}
    requirements={
      'own_architecture':'Own scratch-origin35-layer backbone, tokenizer, vision and auxiliary heads remain. Current counted own actor uses functional frozen-base policy deltas and the same counted base as reference. New phase-separated full-weight raw/book/code then instruction/vision learning, source batching, reference-bound checkpoints and sharded native code assessment are implemented/tested; final useful architecture remains unverified. No external pretrained model or model teacher is used.',
      'public_data':'The complete expanded native loader now enumerates28families/669863source-family records/577662698context positions/467581587actual next/response/EOS targets in35284updates. It integrates the existing298093434packed train tokens, including52970300additionalOS/network/security/crypto tokens,48584later unique source rows with50372explicit references,87925human factual questions, existing education/dialogue/math/photos/words and321derived completion/repair views. Native extrema and every four-rank source draw/count are checked; all2003packed-tail targets remain. These are existing audited sources, not new raw collection. Header variants, four exercise repetitions, cross-family overlap and repeated functions are explicit. Full original data exposure, further public-source variety and all-domain coverage remain unfinished.',
      'under_1b_parameters':'Current exported code policy counts999706442unique values, including984399690frozen own base/head values,15305216actor values and1536separate critic values. All four actual saved actor/Adam states match. The queued full-weight graph counts984399690; a second own foundation reference exists only in its later training phase and is excluded from exported policy, making later resident policy+reference1968799380values. No final trained unified counted model is qualified yet.',
      'text_and_images':'Current warm-up gives no useful multimodal improvement: OCR0/20remains0/20andCER0.95417to1.05; object proper4/15to3/15with blank4/15both; captionI2T1/32both,T2I3/32to2/32. Base vision/word/head bytes remain fixed, but changing the shared language adapter changes image answers. The queued full-weight phase learns word/vision through a fixed decoder and later text anchors to its newly trained own foundation. Generic vision, whole/cropped text reading, genuine photo reasoning, held word controls and full final multimodal quality remain incomplete.',
      'human_feedback':'Execution feedback uses finite public training tests and adds0human judgments; it is not genuine human-preference RLHF. The87925original human factual answers are supervised references, not ranked human preferences. Existing human rankings/reward/DPO/PPO evidence remains preserved. Useful genuine-human preference/PPO policy training and own-output reward fidelity are still required.',
      'serious_full_model_training':'All88actual warm-up updates are separately read back:561selected actor tensors changed, all561Adam clocks equal88on all4replicas, all173functions occur4times, and2076complete presentations contain273128targets. Four epoch online training CE falls0.73748to0.52184without a native code gain. Current173-function controlled training is genuinely running; the22-step prefix contains176actual own responses and all rewards independently replayed. New full-size phase CPU control runs5logical/6physical updates across original code, human QA, JavaScript, a real photo and a word, with445Adam states, positive newly trained own-reference KL, exact next state and text/image/word cold reload. Those five controls are not the467M-target epoch. The full35284-step curriculum is queued behind the live experiment, with its own full configured four-L40 admission still required.',
      'high_end_llm_capability':'NOT ACHIEVED. Matched full512-token code completion changes20/24syntaxand1/24correctto18/24and0/24; instruction changes20/24syntaxand0/24correctto19/24and0/24. Efficiency0/24both frames/both endpoints. Generated knowledge0/56to1/56andlikelihood3/56to4/56are tiny mixed changes; maths1/16to0/16. No current method demonstrates broad high-end competence or frontier parity. No model is promoted.',
      'broad_independent_benchmarks':'All96actual full512code outputs across2frames/2endpoints were independently reexecuted for syntax/function. All raw knowledge/math/OCR decisions were read back. Development subsets and training-reward gains are not independent final generalization. Native code sharding tests ensure complete unchanged24-case512-token reconciliation and reject missing/changed shards, enabling future assessment across all4GPUs. New evaluation-only packed valid-stream NLL is wired at initial/checkpoints/phase boundaries. Its bounded128window measure is not whole-domain quality. Original full559knowledge,24codebothframes,12C++,175publiccode,1319math,170OCR,164objects,128retrieval and held word checks plus genuinely independent final evaluation remain required.',
      'gpu_runtime':'The exact current owner and all4actualRL/control workers are live on the original four local L40s. Current full-profile512-token capacity/recovery already passed. The queued successor is confirmed live, GPU-hidden and childless while waiting; it will acquire only after current owner/workers finish and restore original work. Fresh actual raw4096/later-text/photo/word capacity plus own trained-reference identity, exact all-worker Adam/RNG recovery and invalid-RNG rejection must pass before new production. No source/test/log/plan bound to current or queued work was edited. Long raw pretraining is measured using valid token loss and completed instruction phases; early code-format scores are not an automatic mid-pretraining stop rule.',
      'any_code_correctness_and_efficiency':'NOT ACHIEVED. On the matched22problem prefix, supervised control has2/88positive samples(meanreward0.009375), execution hybrid5/88(mean0.0210633), and both0fully passing programs. All176actual applied rewards reexecute exactly. Hybrid has5nonzero applied policy-surrogate updates, so it has real execution-policy learning rather than only zero-reward CE fallback; this remains public training feedback, not held code improvement. Complete173-function controlled comparison and posttraining native syntax/function/efficiency/broad/vision assessments are running/pending. Wider multilingual,OS/network/security and efficient correct arbitrary code remain unproved.'
    }
    evidence.extend(['reports/verified-views-warmup88-readback-v1/complete.json','reports/verified-views-warmup88-readback-v1/private-numpy-readback.json',
        'reports/execution-prefix22-readback-v1/complete.json','reports/expanded-joint-sources-v1/complete.json','reports/phase-joint-full-CPU-v2/complete.json',
        'reports/phase-joint-update-v1-tests.log','reports/phase-joint-checkpoint-v1-tests.log','reports/phase-joint-driver-v1-tests.log',
        'reports/native-code-shards-v1-tests.log','reports/packed-validation-v1-tests.log'])
    destination=ROOT/'reports/phase-joint-queued-v1-review.json'
    if destination.exists():raise ValueError('Preserve earlier full-scope reviews')
    review=dict(at=datetime.now(timezone.utc).isoformat(),objective_preserved=True,previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        current_goal_turn_progress=['Actual full88-update warm-up/native/Adam/all-source/private-RNG readback and96actual code reexecutions reveal no coding gain.',
            'Genuine controlled production execution learning starts; all176own responses/rewards from a matched22-step prefix independently replay.',
            'Existing additional systems code, multilingual interfaces, human QA and full original multimodal sources integrated into actual native28-family batches.',
            'Complete467581587target foundation/instruction source pass enumerated with all rank views/tails; five actual full-size phase updates/recovery/cold-load prove new kernels.',
            'New phase/reference-bound checkpoint validation and complete parallel code-case reconciliation tested; full staged experiment queued behind current exact live owner.'],
        requirements={k:dict(status='incomplete_for_requested_final_model',current_evidence_and_limits=v,prior_evidence_retained=True) for k,v in requirements.items()},
        current_native_outcomes=outcomes,actual_current_RL_progress=progress,actual_matched_training_prefix=prefix['actual_matched_training_prefix_summary'],
        queued_successor=dict(owner=queued['owner'],planned_steps=35284,planned_targets=467581587,production_updates=0,actual_GPU_capacity_pending=True),
        all_ten_final_requirements_incomplete=True,no_remaining_scope_removed=True,held_coding_gain_proven=False,genuine_human_preference_RLHF_started=False,
        source_files_sha256={str(ROOT/p):file_hash(ROOT/p) for p in evidence},candidate_promoted=False,goal_achieved=False)
    for name in ['README.md','RESEARCH.md','reports/completion-audit.json']:
        if str(ROOT/name) in plan['source_files_sha256'] or str(ROOT/name) in queued['source_files_sha256']:raise ValueError('Never modify bound live/queued documentation')
    atomic_json(destination,review);audit_path=ROOT/'reports/completion-audit.json';snapshot=ROOT/'reports/completion-audit-before-phase-joint-queue-v1.json'
    if snapshot.exists():raise ValueError('Preserve previous full-scope audit')
    shutil.copyfile(audit_path,snapshot);audit=json.loads(audit_path.read_text())
    audit.update(requirements=requirements,updated_at=review['at'],latest_full_scope_review=str(destination.relative_to(ROOT)),
        current_posttraining=dict(phase='controlled_execution_RL_and_supervised_comparison',warmup_updates=88,
            actual_execution_updates=progress['execution_hybrid']['actual_updates'],actual_control_updates=progress['verified_solution']['actual_updates'],
            model_unique_parameters=999706442,production_execution_RL_started=True,held_coding_gain_proven=False),
        queued_full_source_phase_training=review['queued_successor'],all_ten_final_requirements_incomplete=True,goal_achieved=False)
    atomic_json(audit_path,audit)
    message=f'''Real execution learning and full-source expansion, **2026-10-17**: the current **999,706,442-value** own policy finished all**88supervised warm-up updates** and is now running the matched execution-hybrid/control experiment on all4localL40s. Current observed counts are**{progress['execution_hybrid']['actual_updates']}execution updates** and**{progress['verified_solution']['actual_updates']}supervised-control updates**. The matched first22problems give**5/88positive partial rewards forRLversus2/88forcontrol**, means0.0210633versus0.009375; **neither has a fully passing training program**. All176actual own-response rewards were independently reexecuted. Five hybrid updates have a nonzero applied policy surrogate. This is genuine machine-execution policy learning, **not genuine human-preference RLHF or held correctness**.\n\nWarm-up source fitting did **not** yield native coding improvement. All96programs below were independently reexecuted; budgets/frames/initial weights match:\n\n| Full512-token development measure |Initial|After88updates|\n|---|---:|---:|\n| Completion syntax |20/24|18/24|\n| Completion correctness |1/24|0/24|\n| Instruction syntax |20/24|19/24|\n| Instruction correctness |0/24|0/24|\n| Efficiency, either frame |0/24|0/24|\n| Generated knowledge |0/56|1/56|\n| Maths exact |1/16|0/16|\n| OCR exact |0/20|0/20|\n| OCR character error |0.954|1.050|\n\nAll4saved actor/Adam states agree; all561actor tensors changed and all clocks are88. Every173function appears4times, with2,076complete presentations and273,128targets. Epoch online trainingCE falls0.73748to0.52184; this is source fitting, not a quality gain. Objects4/15to3/15with blank4/15both and captionI2T1/32both/T2I3/32to2/32 also show no useful broad visual improvement. No candidate is promoted.\n\nA **complete broader curriculum is now queued**, preserving the live experiment: **28families,669,863source-family records,577,662,698context positions and467,581,587targets over35,284updates**. It integrates the already prepared298.1Mpacked code tokens(including52.97MadditionalOS/network/security/crypto),49klater instruction sources with explicit reference repetitions,87,925human factual questions and the original books/dialogues/maths/photos/text-image/word sources. This is **not fresh raw collection**; variants, repeated exercises and cross-source overlap are explicit. Every source-family record/rank view and original packed tail is enumerated, and actual native source extremes are read back.\n\nThe queued full-weight model is**984,399,690values**. Raw/book/code foundation learning runs first without a reference; the completed own foundation becomes the later text anchor. Image/word updates keep the decoder fixed. The extra own reference is training-only, making later resident policy+reference1,968,799,380values; the export remains under1B. Five short full-sizeCPUcontrols(6physicaloptimizer calls) prove original code/QA/JS/photo/word learning,445Adam states,positiveown-foundationKL,exact next weights/Adam/RNG and text/image/word cold reload. **These are not the full epoch**: new actual full-profile four-GPU admission still gates production. Phase/reference checkpoint validation, evaluation-only valid packed NLL and unchanged512-token code sharding acrossall4GPUs are tested/wired.\n\nThe complete467.6M-target pass, later useful genuine-human preference training, complete independent broad/code/multimodal benchmarks and a qualified unified final model remain required. **All ten final-model requirements remain incomplete; no high-end parity is claimed.** See [liveRL](reports/verified-views-execution-v1/execution_hybrid-progress.json), [matched reward replay](reports/execution-prefix22-readback-v1/complete.json), [warm-up readback](reports/verified-views-warmup88-readback-v1/complete.json), [expanded native sources](reports/expanded-joint-sources-v1/complete.json), [queued full pass](reports/phase-joint-v1-queue.json) and [full-scope review](reports/phase-joint-queued-v1-review.json). Earlier dated entries are historical.\n\n'''
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;first,_,rest=path.read_text().partition('\n\n');path.write_text(first+'\n\n'+message+rest)
    print(json.dumps(dict(review=str(destination),actual_execution_updates=progress['execution_hybrid']['actual_updates'],queued_targets=467581587,all_ten_final_requirements_incomplete=True)),flush=True)


if __name__=='__main__':main()
