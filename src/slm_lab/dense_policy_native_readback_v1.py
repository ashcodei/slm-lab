"""Complete retained preference, word and generated native controls."""
import json
from pathlib import Path
from .dense_human_preference_evaluation_v1 import readback as preference_readback
from .shared_word_evidence_v1 import readback_held


def readback(root,reports,arms,held,words,tokenizer):
    root,reports=map(Path,(root,reports));preference={};reading={};native={}
    from .code_quality import execute
    questions={q['id']:q for q in map(json.loads,(root/'data/public-code-assessment-v1/questions.jsonl').open())}
    for arm in ['initial',*arms]:
        dest=reports/arm;preference[arm]=preference_readback(held,dest/'held-preferences.json')
        reading[arm]=readback_held(words,tokenizer,dest/'held-words.json',64)
        knowledge=json.loads((dest/'knowledge.json').read_text());maths=json.loads((dest/'math-complete.json').read_text())
        public=[r for i in range(4) for r in json.loads((dest/f'public-{i}.json').read_text())['results']]
        if (len(knowledge['predictions'])!=559 or len(public)!=175 or {r['id'] for r in public}!=set(questions)
                or maths['evaluated_examples']!=1319 or not maths['all_original_IDs_prompts_generated_tokens_numeric_answers_and_scores_reconciled']):
            raise ValueError('All original knowledge/public-code/native-math questions required')
        passing=[]
        for row in public:
            if row['execution'].get('functional_passed'):
                q=questions[row['id']];result=execute(dict(language='Python',code=row['code'],function=q['function'],cases=q['cases'],stress=[]))
                if not result.get('functional_passed'):raise ValueError('Every public passing program must independently execute all original cases')
                passing.append(row['id'])
        local={}
        for mode in ('completion','instruction'):
            rows=json.loads((dest/('code-'+mode+'.json')).read_text())['results']
            if len(rows)!=24:raise ValueError('Every original Python/JavaScript functional/efficiency task required')
            local[mode]=dict(syntax=sum(r['candidate'].get('syntax_valid',False) for r in rows),
                functional=sum(r['candidate'].get('functional_passed',False) for r in rows),efficiency=sum(r['efficiency_stress_passed'] for r in rows))
        cpp=json.loads((dest/'C++.json').read_text());vision=json.loads((dest/'vision.json').read_text())
        native[arm]=dict(knowledge=knowledge['overall'],public_code=dict(tasks=175,syntax=sum(r['execution'].get('syntax_valid',False) for r in public),
            functional=len(passing),independently_reexecuted_pass_ids=passing),local_code=local,
            CPP={k:v for k,v in cpp.items() if k not in ('results','evaluation_provenance')},
            math={k:v for k,v in maths.items() if k not in ('results','evaluation_provenance','shard_reports_sha256')},
            OCR=json.loads((dest/'ocr.json').read_text())['overall'],objects=json.loads((dest/'objects.json').read_text())['overall'],
            captions={k:vision[k] for k in ('images','proper_nll','blank_nll_gain','mismatch_nll_gain','caption_retrieval')})
    return dict(complete_held_preferences=preference,complete_held_word_image_controls=reading,matched_native_capability=native)
