from contextlib import nullcontext
import hashlib
from types import SimpleNamespace

import pytest
import torch

from slm_lab.codenet_model_assessment_v1 import generate_record, score_record, summarize, validate_question
from slm_lab.model import ModelConfig,TinyMultimodalLM
from slm_lab.recurrent_ablation import state_digest
from slm_lab.training import Encoder


class Tokenizer:
    def token_to_id(self,text):return {'<pad>':0,'<bos>':1,'<eos>':2}[text]
    def get_vocab_size(self):return 128
    def encode(self,text):return SimpleNamespace(ids=list(map(ord,text)))
    def decode(self,ids,skip_special_tokens=True):
        special={0:'<pad>',1:'<bos>',2:'<eos>'}
        return ''.join(special.get(i,chr(i)) for i in ids if not skip_special_tokens or i not in special)


def config():
    return ModelConfig(vocab_size=128,hidden_size=16,intermediate_size=32,num_layers=1,
        num_heads=4,num_kv_heads=2,max_text_length=128,vision_hidden_size=16,
        vision_intermediate_size=32,vision_layers=1,vision_heads=4,image_size=16,patch_size=4,pool_size=2)


def question(encoder,prompt='p',identity='q1',language='Python'):
    row={'id':identity,'problem_id':'p00001','group_id':'held-component','split':'test',
        'language':language,'target_runtime':'Python3.11' if language=='Python' else 'GNU C++17',
        'prompt':prompt,'original_html_sha256':'1'*64,'source':'test public statement','license':'test',
        'use_for_training_or_preferences':False,'complete_prompt_tokens':0}
    row['complete_prompt_tokens']=1+len(encoder.tokens(encoder.prompt(row)))
    return row


class ScriptedModel:
    config=config()
    def __init__(self,body):self.body=body;self.calls=[]
    def generate(self,ids,**options):
        self.calls.append((ids.clone(),options))
        return torch.cat((ids,torch.tensor([self.body],device=ids.device)),dim=1)


def test_generation_preserves_whole_prompt_and_reserved_body_tokens_removing_only_final_eos(tmp_path):
    tok=Tokenizer();encoder=Encoder(tok,config(),tmp_path,image_size=16)
    row=question(encoder);model=ScriptedModel([97,0,98,2])
    result=generate_record(model,tok,encoder,row,4,'cpu',nullcontext)
    assert result['program_text']=='a<pad>b'
    assert result['generated_token_ids']==[97,0,98,2] and result['terminal_eos']
    assert result['literal_reserved_body_token_count']==1
    assert model.calls[0][0][0].tolist()==[encoder.bos]+encoder.tokens(encoder.prompt(row))
    assert model.calls[0][1]=={'max_new_tokens':4,'temperature':0.,'eos_token_id':2,'use_cache':True}


def test_prompt_mismatch_or_overflow_stops_before_model_generation_without_clipping(tmp_path):
    tok=Tokenizer();encoder=Encoder(tok,config(),tmp_path,image_size=16)
    model=ScriptedModel([2])
    row=question(encoder)
    for invalid in [{**row,'complete_prompt_tokens':row['complete_prompt_tokens']+1},question(encoder,'p'*115)]:
        with pytest.raises(ValueError,match='no clipping'):
            generate_record(model,tok,encoder,invalid,16,'cpu',nullcontext)
    assert not model.calls and encoder.truncated==0


def test_question_schema_prevents_reference_or_hidden_case_fields_from_reaching_generation(tmp_path):
    tok=Tokenizer();encoder=Encoder(tok,config(),tmp_path,image_size=16);row=question(encoder)
    for update in [{'response':'gold program'},{'tests':['hidden input']},{'split':'train'},
                   {'use_for_training_or_preferences':True}]:
        model=ScriptedModel([2])
        with pytest.raises(ValueError,match='public held question'):
            generate_record(model,tok,encoder,{**row,**update},4,'cpu',nullcontext)
        assert not model.calls


def test_budget_exhaustion_is_explicit_and_early_or_post_eos_outputs_are_rejected(tmp_path):
    tok=Tokenizer();encoder=Encoder(tok,config(),tmp_path,image_size=16);row=question(encoder)
    result=generate_record(ScriptedModel([97,98]),tok,encoder,row,2,'cpu',nullcontext)
    assert result['stop_reason']=='token_budget' and result['program_text']=='ab'
    for body in [[97],[2,97]]:
        with pytest.raises(ValueError):generate_record(ScriptedModel(body),tok,encoder,row,2,'cpu',nullcontext)


def test_actual_own_small_model_generation_uses_complete_prompt_and_leaves_state_unchanged(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(73)
    model=TinyMultimodalLM(config()).eval();before=state_digest(model.state_dict())
    tok=Tokenizer();encoder=Encoder(tok,model.config,tmp_path,image_size=16);row=question(encoder)
    result=generate_record(model,tok,encoder,row,8,'cpu',nullcontext)
    assert before==state_digest(model.state_dict())
    assert result['prompt_tokens']==row['complete_prompt_tokens']
    assert 1<=result['generated_tokens_including_stop']<=8


def completion(q,text='print(0)\n'):
    return {**{key:q[key] for key in ['id','problem_id','group_id','split','language']},
            'program_text':text,'program_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'generation_budget':2048,'stop_reason':'eos'}


def cases(q,stress=True):
    result=[{'id':'public','problem_id':q['problem_id'],'split':'test','use_for_training_or_preferences':False,
             'kind':'published_sample'}]
    if stress:result.append({**result[0],'id':'large','kind':'stress'})
    return result


def test_complete_denominators_keep_compile_failures_and_separate_functional_from_stress(tmp_path):
    encoder=Encoder(Tokenizer(),config(),tmp_path,image_size=16)
    py=question(encoder);cpp=question(encoder,identity='q2',language='C++')
    gen_py,gen_cpp=completion(py),completion(cpp,'broken C++')
    def py_assessor(*args):
        return {'cases':[{'case_id':'public','oracle_passed':True},{'case_id':'large','oracle_passed':False}]}
    def cpp_assessor(*args):return {'compilation':{'status':'runtime_exit'},'cases':[]}
    result_py=score_record(py,gen_py,cases(py),None,py_assessor)
    result_cpp=score_record(cpp,gen_cpp,cases(cpp),None,cpp_assessor)
    result=summarize([py,cpp],[gen_py,gen_cpp],[result_py,result_cpp])
    assert result['questions']==2 and result['functional_correct']==1 and result['functional_pass_at_1']==.5
    assert result['overall_correct_including_defined_stress']==0 and result['bounded_stress_pass_rate']==0.
    assert result['case_counts']=={'published_sample':2,'stress':2}
    assert result['passed_case_counts']=={'published_sample':1}
    assert result_py['python_syntax_or_CPP_compilation_passed'] is True
    assert result_cpp['python_syntax_or_CPP_compilation_passed'] is False
    with pytest.raises(ValueError,match='Every predeclared'):
        summarize([py,cpp],[gen_py],[result_py])


def test_sealed_code_and_complete_case_population_cannot_be_changed_or_partially_scored(tmp_path):
    encoder=Encoder(Tokenizer(),config(),tmp_path,image_size=16);q=question(encoder);g=completion(q)
    with pytest.raises(ValueError,match='changed before execution'):
        score_record(q,{**g,'program_text':'print(1)'},cases(q),None,None)
    with pytest.raises(ValueError,match='partial executable'):
        score_record(q,g,cases(q),None,lambda *a:{'cases':[{'case_id':'public','oracle_passed':True}]})
    with pytest.raises(ValueError,match='duplicate/foreign'):
        score_record(q,g,cases(q),None,lambda *a:{'cases':[{'case_id':'foreign','oracle_passed':True}]})


def test_task_without_stress_is_marked_unrated_instead_of_claiming_efficiency(tmp_path):
    encoder=Encoder(Tokenizer(),config(),tmp_path,image_size=16);q=question(encoder);g=completion(q)
    r=score_record(q,g,cases(q,stress=False),None,lambda *a:{'cases':[{'case_id':'public','oracle_passed':True}]})
    report=summarize([q],[g],[r])
    assert r['functional_passed'] and r['overall_passed'] and r['bounded_stress_passed'] is None
    assert report['stress_eligible_questions']==0 and report['bounded_stress_pass_rate'] is None
