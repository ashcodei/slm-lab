"""Read all96retained actions and actual train prompts; no new inference."""
from collections import Counter
from datetime import datetime, timezone
import ast
import json
import os
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reports/code-first-step2048-native-code-CPU-v1'
OUT = ROOT/'reports/code-first-repetition-and-prompt-diagnostic-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('Fresh GPU-hidden completed-action diagnostic required')
    from tokenizers import Tokenizer
    from slm_lab.code_contests_CE_view_v1 import CodeContestsCEView
    reader = ROOT/'reports/code-first-step2048-native-code-CPU-v1-readback/complete.json'
    complete = json.loads(reader.read_text())
    if not complete['complete']:raise ValueError('All96original actual actions and source/outcome checks must finish')
    tokenizer_path = ROOT/'data/tokenizer-corpus/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    files = [Path(__file__),reader,SOURCE/'complete.json',tokenizer_path]
    profiles = [];deliveries = []
    for endpoint in ['initial','step2048']:
        for frame in ['completion','instruction']:
            action_path = SOURCE/endpoint/(frame+'-all24-actual-generation-actions.jsonl')
            report_path = SOURCE/endpoint/('code-'+frame+'.json')
            files.extend([action_path,report_path])
            actions = [json.loads(line) for line in action_path.open()]
            rows = json.loads(report_path.read_text())['results']
            if len(actions) != 24 or len(rows) != 24:raise ValueError('Every original whole frame required')
            counts = Counter();repetition = []
            for action,row in zip(actions,rows):
                ids = action['actual_generated_token_ids']
                if len(ids) != action['actual_native_action_count']:raise ValueError('Actual action count differs')
                stops = action['actual_stopped_on_EOS']
                if stops != bool(ids and ids[-1] == 2):raise ValueError('The actual final EOS action must remain exact')
                grams = Counter(tuple(ids[i:i+16]) for i in range(max(0,len(ids)-15)))
                repeat = 1-len(grams)/sum(grams.values()) if grams else 0.
                parsed = None
                if row['language'] == 'Python':
                    try:
                        tree = ast.parse(row['code'])
                        names = [n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
                        parsed = dict(top_level_functions=len(names),repeated_top_level_function_names=len(names)-len(set(names)),
                            requested_function_present=row['task'] in names)
                    except SyntaxError as error:
                        parsed = dict(syntax_error=error.msg,line=error.lineno)
                counts.update(deliveries=1,generated_actions=len(ids),stopped_on_EOS=int(stops),
                    exhausted_512_action_budget=int(not stops and len(ids)==512),
                    repeated_16gram_majority=int(repeat >= .5),syntax=int(row['candidate'].get('syntax_valid',False)),
                    functional=int(row['candidate'].get('functional_passed',False)))
                repetition.append(repeat)
                deliveries.append(dict(endpoint=endpoint,frame=frame,language=row['language'],task=row['task'],
                    actual_actions=len(ids),actual_EOS=stops,repeated_16gram_fraction=repeat,
                    most_frequent_16gram_occurrences=max(grams.values(),default=0),actual_Python_structure=parsed,
                    unmodified_generated_text_prefix=tokenizer.decode(ids[:100],skip_special_tokens=False)))
            profiles.append(dict(endpoint=endpoint,frame=frame,**dict(counts),mean_repeated_16gram_fraction=sum(repetition)/24))
    view = CodeContestsCEView(ROOT)
    positions = [i*(len(view)-1)//7 for i in range(8)]
    training = []
    for position in positions:
        ids,labels = view.sequence(position)
        start = next(i for i,v in enumerate(labels) if v != -100)
        row = view.row(position)
        text = row['prompt']
        prompt = text if text.rstrip().endswith('Assistant:') else 'User: '+text+'\nAssistant:'
        expected = [1,*tokenizer.encode(prompt).ids]
        decoded = tokenizer.decode(ids[1:start],skip_special_tokens=False)
        training.append(dict(original_admitted_train_position=position,prompt_tokens=start,
            whole_original_publisher_prompt_matches_same_User_Assistant_wrapper=ids[:start]==expected,
            actual_prompt_has_User_prefix=decoded.startswith('User:'),actual_prompt_has_Assistant_suffix=decoded.endswith('Assistant:'),
            complete_source_context_tokens=len(ids),complete_response_EOS_targets=len(ids)-start,
            actual_prompt_prefix=decoded[:300]))
    OUT.mkdir()
    atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
        actual_all96_action_profiles=profiles,actual_original_deliveries=deliveries,
        actual_eight_evenly_spaced_training_prompt_checks=training,
        all_eight_sampled_training_prompt_wrappers_match=all(r['whole_original_publisher_prompt_matches_same_User_Assistant_wrapper'] for r in training),
        eight_sampled_prompt_checks_not_an_all_history_format_certificate=True,
        repeat16_statistic_is_a_diagnostic_not_an_algorithmic_correctness_test=True,
        source_files_sha256={str(p):file_hash(p) for p in files},
        new_neural_forwards=0,new_program_executions=0,new_model_updates=0,
        programs_not_repaired_or_rescored=True,useful_model_gain_not_established=True,goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_profiles=profiles,
        sampled_training_wrappers_match=all(r['whole_original_publisher_prompt_matches_same_User_Assistant_wrapper'] for r in training))),flush=True)


if __name__ == '__main__':main()
