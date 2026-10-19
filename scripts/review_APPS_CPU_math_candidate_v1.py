"""Reconstruct the completed older706M CPU math source/answer score records."""
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path

from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.math_corpus import final_answer,prompt

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/apps-full-language-CPU-math-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Raw score/source reader must explicitly hide GPUs')
    output=REPORTS/'candidate-source-score-readback-v1.json'
    if output.exists():raise ValueError('Preserve the completed older math readback')
    path=REPORTS/'candidate.json';report=json.loads(path.read_text());launch=json.loads((REPORTS/'candidate-launch.json').read_text())
    stat=Path('/proc')/str(launch['pid'])/'stat'
    if stat.exists():
        fields=stat.read_text().rsplit(') ',1)[1].split()
        if fields[19]==launch['ticks'] and fields[0] not in ('Z','X'):raise ValueError('The exact candidate math actor is still live')
    checkpoint=ROOT/'checkpoints/apps-full-language-v1/model.pt';tokenizer_path=checkpoint.parent/'tokenizer.json'
    data=ROOT/'data/math-reasoning-v2/test.jsonl';manifest=data.parent/'manifest.json'
    guard=FileBindings({str(path):file_hash(path),str(checkpoint):report['checkpoint_sha256'],
        str(data):report['data_sha256'],str(manifest):report['manifest_sha256'],str(tokenizer_path):report['tokenizer_sha256'],
        **report['evaluation_provenance']['files_sha256']})
    if (report['checkpoint_sha256']!=launch['checkpoint_sha256'] or report['inference_device_type']!='cpu'
            or report['inference_precision']!='fp32' or report['generation_budget']!=512 or report['evaluated_examples']!=1319
            or report['official_test_examples']!=1319 or report['full_official_test'] is not True):
        raise ValueError('Exact completed original1319question CPU/FP32/512token protocol required')
    source=[json.loads(line) for line in data.open()];source.sort(key=lambda row:hashlib.sha256(row['question'].encode()).digest())
    tokenizer=Tokenizer.from_file(str(tokenizer_path));records=report['results']
    if len(records)!=len(source) or len({r['id'] for r in records})!=1319:raise ValueError('Every original question exactly once required')
    correct=formatted=0
    for row,result in zip(source,records):
        gold=Fraction(row['answer_numerator'],row['answer_denominator']);answer=final_answer(result['raw_response'])
        ids=[1,*tokenizer.encode('User: '+prompt(row['question'])+'\nAssistant:').ids]
        expected=dict(id=row['id'],answer_numerator=answer.numerator if answer is not None else None,
            answer_denominator=answer.denominator if answer is not None else None,expected_numerator=gold.numerator,
            expected_denominator=gold.denominator,correct=answer==gold,format_compliant=answer is not None,context_tokens=len(ids))
        if any(result[k]!=value for k,value in expected.items()) or len(ids)+512>4096:
            raise ValueError('Every original prompt context, original gold and delivered raw-response numeric score must reconstruct')
        correct+=answer==gold;formatted+=answer is not None
    if (correct!=report['correct'] or correct/1319!=report['numeric_exact_accuracy'] or formatted/1319!=report['format_compliance']):
        raise ValueError('Every complete all-source numeric/format aggregate must reconstruct')
    guard.check(full=True)
    atomic_json(output,dict(observed_at=datetime.now(timezone.utc).isoformat(),reader_sha256=file_hash(__file__),
        source_report_sha256=file_hash(path),checkpoint_sha256=report['checkpoint_sha256'],
        model='older fully trained706M APPS candidate; not the new984M dense-growth or999M aligned model',
        evaluated_original_questions=1319,correct=correct,numeric_exact_accuracy=correct/1319,format_compliant=formatted,
        inference_device='cpu',precision='fp32',declared_generation_budget=512,
        every_original_ID_gold_context_and_raw_response_answer_score_reconstructed=True,
        reconstruction_uses_unchanged_pinned_official_scorer=True,evaluator_source_and_runtime_hashes_verified=True,
        actual_neural_generation_not_replayed=True,raw_generated_token_IDs_not_retained_by_original_evaluator=True,
        paired_CPU_baseline_comparison_pending=not (REPORTS/'base.json').exists(),
        earlier_same_candidate_CUDA_BF16_math_correct=18,CPU_and_GPU_precision_protocols_kept_separate=True,
        numeric_matches_do_not_prove_reasoning_or_high_end_capability=True,model_updates=0,candidate_promoted=False,goal_achieved=False))
    print('complete_older_APPS_CPU_math',correct,'/',len(source))


if __name__=='__main__':main()
