"""Admit independently reconstructed original training attention; no model work."""
from datetime import datetime,timezone
import json
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.point_grounding_data_v1 import PointerData
from slm_lab.point_grounding_protocol_v1 import fit_position_prior,control_pairs

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/localized-pointer-candidate-v1'
REPORTS=ROOT/'reports/localized-pointer-candidate-v1'


def main():
    targets=[DATA/'admission.json',DATA/'training-position-prior.json',DATA/'validation-control-pairs.jsonl',
        REPORTS/'admitted-reader-readback.json']
    if any(p.exists() for p in targets):
        raise ValueError('Preserve previous original-pointer admission or partial evidence')
    proof=REPORTS/'independent-source-readback.json'; evidence=json.loads(proof.read_text())
    for group in ['source_candidate_files_sha256','audit_files_sha256']:
        for path,sha in evidence[group].items():
            if file_hash(path)!=sha:
                raise ValueError('Original independent source readback bindings changed')
    if evidence['candidate_manifest_sha256']!=file_hash(DATA/'manifest.json'):
        raise ValueError('Actual complete independently checked candidate required')
    tests={'point-grounding-data-v1-tests.log':'2 passed','point-grounding-protocol-v1-tests.log':'2 passed',
        'point-grounding-evaluation-v1-tests.log':'1 passed'}
    for name,passed in tests.items():
        if passed not in (ROOT/'reports'/name).read_text():
            raise ValueError('Actual own-input, matched-control and evaluator integration tests required')
    train=PointerData(DATA,'train','audit'); held=PointerData(DATA,'validation','evaluation')
    prior=fit_position_prior(train.rows)
    prior.update(original_train_file_sha256=file_hash(DATA/'train.jsonl'),
        candidate_manifest_sha256=file_hash(DATA/'manifest.json'))
    atomic_json(DATA/'training-position-prior.json',prior)
    controls=control_pairs(held.rows,held.photos)
    (DATA/'validation-control-pairs.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in controls))
    paths=[Path(__file__).resolve(),ROOT/'src/slm_lab/point_grounding_data_v1.py',
        ROOT/'src/slm_lab/point_grounding_protocol_v1.py',ROOT/'src/slm_lab/point_grounding_evaluation_v1.py',
        ROOT/'src/slm_lab/point_grounding_head_v1.py',ROOT/'src/slm_lab/localized_pointer_evidence_v1.py',
        ROOT/'data/localized-pointer-source-v1/alignment-qualification.json',
        DATA/'training-position-prior.json',DATA/'validation-control-pairs.jsonl']
    paths.extend(ROOT/'reports'/name for name in tests)
    paths.extend(ROOT/'tests'/name for name in ['test_point_grounding_data_v1.py',
        'test_point_grounding_protocol_v1.py','test_point_grounding_evaluation_v1.py'])
    admission=dict(admitted_at=datetime.now(timezone.utc).isoformat(),training_admitted=True,
        scope='original-human-attention-distribution-supervision-only',
        candidate_manifest_sha256=file_hash(DATA/'manifest.json'),independent_source_readback_sha256=file_hash(proof),
        original_validation_optimization_forbidden=True,original_train_queries=len(train),original_validation_queries=len(held),
        annotation_license='CC-BY-4.0',original_photo_license='CC-BY-2.0',creator_attribution_retained=True,
        original_temporal_alignment_speech_model_assisted=True,
        approximate_annotator_attention_not_object_boxes_OCR_or_preferences=True,
        image_size=448,grid_side=14,maximum_whole_query_tokens=34,
        prior_fitted_on_original_train_queries_only=True,
        validation_wrong_image_controls=len(controls),same_photo_disjoint_query_controls=sum(r['wrong_query_id'] is not None for r in controls),
        runtime_and_control_files_sha256={str(p):file_hash(p) for p in paths},
        no_change_to_preexisting_training_data_or_curricula=True,
        model_training_started=False,GPU_capacity_still_required=True,model_updates=0,goal_achieved=False)
    atomic_json(DATA/'admission.json',admission)
    admitted=PointerData(DATA,'train','optimization')
    try:
        PointerData(DATA,'validation','optimization')
    except ValueError:
        rejects_held=True
    else:
        raise ValueError('Original validation may never supply gradients')
    atomic_json(REPORTS/'admitted-reader-readback.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
        admitted_original_train_queries=len(admitted),original_validation_queries=len(held),
        all_original_query_BPE_framing_and_dense_targets_reconciled=True,
        original_validation_optimization_rejected=rejects_held,
        train_only_prior_original_queries=prior['fitted_original_training_queries'],
        presealed_wrong_image_controls=len(controls),same_photo_disjoint_query_controls=admission['same_photo_disjoint_query_controls'],
        admission_sha256=file_hash(DATA/'admission.json'),independent_source_readback_sha256=file_hash(proof),
        model_updates=0,capability_gain_established=False,goal_achieved=False))
    print('original_train_only_pointer_supervision_admitted',len(admitted),'validation',len(held),flush=True)


if __name__=='__main__':
    main()
