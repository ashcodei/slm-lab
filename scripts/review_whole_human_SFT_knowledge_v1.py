"""Recompute the complete matched native knowledge scores from retained answers."""
import json
import math
from pathlib import Path

from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.knowledge_evaluation import letter, interval
from slm_lab.knowledge_chat_diagnostic_v1 import question_prompt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/whole-human-SFT-v1'


def main():
    destination = REPORTS/'independent-native-knowledge-readback-v1.json'
    if destination.exists(): raise ValueError('Preserve prior native knowledge readbacks')
    questions = ROOT/'data/knowledge-eval-v1/questions.jsonl'
    original = [json.loads(line) for line in questions.read_text().splitlines()]
    selected = json.loads((REPORTS/'selection.json').read_text())['selected_arm']; summaries = {}; predictions = {}
    def summarize(items):
        count = len(items); generated = sum(r['generated_correct'] for r in items); choice = sum(r['likelihood_correct'] for r in items)
        return dict(questions=count, generated_accuracy=generated/count, likelihood_accuracy=choice/count,
            format_compliance=sum(r['generated_index'] is not None for r in items)/count,
            chance_accuracy=sum(r['chance_accuracy'] for r in items)/count,
            generated_accuracy_95ci=interval(generated,count), likelihood_accuracy_95ci=interval(choice,count))
    for arm in ('initial', selected):
        path = REPORTS/arm/'knowledge.json'; report = json.loads(path.read_text()); rows = report['predictions']
        checkpoint = ROOT/'checkpoints/whole-human-SFT-v1'/arm/'model.pt'
        tokenizer = checkpoint.parent/'tokenizer.json'; tok = Tokenizer.from_file(str(tokenizer))
        if (len(rows) != len(original) or len(rows) != 559 or report['questions_sha256'] != file_hash(questions)
                or report['checkpoint_sha256'] != file_hash(checkpoint) or report['tokenizer_sha256'] != file_hash(tokenizer)
                or report['generation_budget'] != 32 or report['skipped']
                or report['inference_device_type'] != 'cuda' or report['inference_precision'] != 'bf16'):
            raise ValueError('Complete original matched native CUDA/BF16 knowledge protocol required')
        for question, row in zip(original, rows):
            scores = row['candidate_scores']; generated = letter(row['generated_response'], len(question['options']))
            best = max(range(len(scores)), key=lambda i:scores[i]); gold = int(question['answer_index'])
            if (row['id'] != question['id'] or row['domain'] != question['domain'] or row['answer_index'] != gold
                    or len(scores) != len(question['options']) or any(not math.isfinite(v) or v > 0 for v in scores)
                    or row['context_tokens'] != len([1, *tok.encode(question_prompt(question, 'plain')).ids])
                    or row['chance_accuracy'] != 1/len(scores) or row['generated_index'] != generated
                    or row['generated_correct'] != (generated == gold) or row['likelihood_index'] != best
                    or row['likelihood_correct'] != (best == gold)):
                raise ValueError('Every original domain/prompt/gold choice/answer parse and likelihood must reconcile')
        if report['overall'] != summarize(rows) or report['by_domain'] != {
                domain:summarize([r for r in rows if r['domain'] == domain]) for domain in sorted({r['domain'] for r in rows})}:
            raise ValueError('Every original raw-answer aggregate and confidence interval must reconstruct')
        summaries[arm] = dict(generated_correct=sum(r['generated_correct'] for r in rows),
            likelihood_correct=sum(r['likelihood_correct'] for r in rows),
            format_compliant=sum(r['generated_index'] is not None for r in rows),
            overall=report['overall'], by_domain=report['by_domain'], retained_report_sha256=file_hash(path))
        predictions[arm] = rows
    changes = {}
    for field in ('generated_correct', 'likelihood_correct'):
        changes[field] = dict(gained=[b['id'] for a,b in zip(predictions['initial'], predictions[selected]) if b[field] and not a[field]],
                              lost=[b['id'] for a,b in zip(predictions['initial'], predictions[selected]) if a[field] and not b[field]])
    atomic_json(destination, dict(selected_arm=selected, matched_native_knowledge=summaries, paired_changes=changes,
        every_original_question_prompt_gold_answer_parse_score_and_domain_aggregate_reconstructed=True,
        retained_generated_text_not_raw_generation_tokens=True, numerical_GPU_forwards_not_independently_replayed=True,
        teacher_response_improvement_not_broad_native_capability_improvement=True,
        model_updates=0, candidate_promoted=False, goal_achieved=False))


if __name__ == '__main__': main()
