import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from slm_lab import knowledge_evaluation as native
from slm_lab.corpus import file_hash
from slm_lab.knowledge_chat_diagnostic_v1 import evaluate_rows, question_prompt
from slm_lab.own_aligned_assistant_v2 import OwnAlignedAssistantV2
from test_own_aligned_assistant_v1 import fixture


def test_plain_full_native_model_scores_equal_original_evaluator_and_chat_keeps_question(tmp_path, monkeypatch):
    model = OwnAlignedAssistantV2(fixture(), rank=2, reward_hidden=8, word_temporal=4).eval().aligned
    vocab = {t:i for i,t in enumerate(['<pad>', '<bos>', '<eos>', '<unk>', 'A', 'B']+[f't{i}' for i in range(6,32)])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token='<unk>')); tok.pre_tokenizer = pre_tokenizers.Whitespace()
    token_path = tmp_path/'tokenizer.json'; tok.save(str(token_path))
    checkpoint = tmp_path/'model.pt'; checkpoint.write_bytes(b'loader supplied actual tiny full model')
    rows = [dict(id='original-1', domain='math', question='One?', options=['first', 'second'], answer_index=0)]
    path = tmp_path/'questions.jsonl'; path.write_text(json.dumps(rows[0])+'\n')
    path.with_suffix('.audit.json').write_text(json.dumps(dict(clean_questions_sha256=file_hash(path))))
    monkeypatch.setattr(native, 'load_checkpoint', lambda *a, **k:(model, dict(tokenizer_sha256=file_hash(token_path))))
    destination = tmp_path/'native.json'
    native.evaluate(checkpoint, path, destination, device='cpu', max_examples=1, max_new_tokens=4)
    original = json.loads(destination.read_text())['predictions'][0]
    evidence = []; plain = evaluate_rows(model, tok, rows, 'plain', max_new_tokens=4, record=evidence.append)
    for key,value in original.items(): assert plain['predictions'][0][key] == value
    assert evidence == plain['predictions']
    retained = plain['predictions'][0]
    assert tok.decode(retained['generated_ids']) == retained['generated_response']
    assert [sum(item['per_token_logprobs']) for item in retained['candidate_token_evidence']] == retained['candidate_scores']
    chat = evaluate_rows(model, tok, rows, 'chat', max_new_tokens=4)
    assert chat['predictions'][0]['prompt'] == 'User: '+question_prompt(rows[0], 'plain')+'\nAssistant:'
    assert chat['predictions'][0]['id'] == rows[0]['id']
    assert chat['overall']['questions'] == 1 and chat['generation_budget'] == 4
    with pytest.raises(ValueError, match='fit'):
        evaluate_rows(model, tok, rows, 'chat', max_new_tokens=model.config.max_text_length)
