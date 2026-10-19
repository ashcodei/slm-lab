import json
import torch
import pytest
from tokenizers import Tokenizer,models,pre_tokenizers,decoders,trainers
from slm_lab.data import hh_pair,stable_id
from slm_lab.training import Encoder
from slm_lab.model import ModelConfig


def test_human_pairs_preserve_shared_prompt_and_source():
    row={'chosen':'\n\nHuman: Hi\n\nAssistant: Hello','rejected':'\n\nHuman: Hi\n\nAssistant: Goodbye'}
    pair=hh_pair(row)
    assert pair['chosen']=='Hello' and pair['rejected']=='Goodbye'
    assert pair['feedback_source']=='human'
    with pytest.raises(ValueError):hh_pair(dict(row,rejected='\n\nHuman: Different\n\nAssistant: Goodbye'))


def test_dedup_is_case_and_whitespace_insensitive():
    assert stable_id('Hello  WORLD')==stable_id(' hello\nworld ')


def test_sft_supervises_response_only_and_padding_is_ignored(tmp_path):
    tok=Tokenizer(models.BPE(unk_token='<unk>'));tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder=decoders.ByteLevel();tok.train_from_iterator(['hello world'],trainers.BpeTrainer(
        vocab_size=260,initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=['<pad>','<bos>','<eos>','<unk>']))
    c=ModelConfig(vocab_size=260,max_text_length=128)
    enc=Encoder(tok,c,tmp_path)
    rows=[{'prompt':'hi','response':'yes'},{'prompt':'long prompt','response':'short'}]
    batch=enc.batch(rows,'response','cpu')
    for i,row in enumerate(rows):
        targets=batch['labels'][i]
        actual=targets[targets!=-100].tolist()
        assert actual==enc.tokens(row['response'])+[enc.eos]
        assert torch.all(batch['labels'][i][~batch['attention_mask'][i]]==-100)


def test_preference_truncation_retains_the_same_prompt_for_both_responses(tmp_path):
    from tokenizers import Tokenizer,models,pre_tokenizers,trainers
    from slm_lab.training import Encoder
    from slm_lab.model import ModelConfig
    tok=Tokenizer(models.BPE(unk_token='<unk>'));tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train_from_iterator(['Common prompt with short and longer responses.'],trainers.BpeTrainer(vocab_size=260,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),special_tokens=['<pad>','<bos>','<eos>','<unk>']))
    encoder=Encoder(tok,ModelConfig(vocab_size=260,max_text_length=64),tmp_path)
    row={'prompt':'Common long question with context '*10,'chosen':'Yes.','rejected':'A much longer response with more context '*3}
    good,good_labels=encoder.sequence(row,'chosen');bad,bad_labels=encoder.sequence(row,'rejected')
    assert [token for token,label in zip(good,good_labels) if label==-100]==[token for token,label in zip(bad,bad_labels) if label==-100]
    assert len(good)<=64 and len(bad)<=64
