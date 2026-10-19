import json

from PIL import Image
import pytest
from tokenizers import Tokenizer
import torch

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import pixel_hash
from slm_lab.model import ModelConfig,TinyMultimodalLM
from slm_lab.point_grounding_data_v1 import PointerData
from slm_lab.point_grounding_head_v1 import PointGroundingHead
from slm_lab.point_grounding_evaluation_v1 import evaluate,state_digest
from slm_lab.point_grounding_protocol_v1 import fit_position_prior


TOKEN='checkpoints/dialogue-retention-v1/systems-variety-code/tokenizer.json'


def test_actual_own_head_retains_matching_inputs_logits_controls_and_no_updates(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(2179)
    root=tmp_path/'unit-data';root.mkdir();tokenizer=Tokenizer.from_file(TOKEN)
    photos=[];rows=[]
    for key,color in [('one',(220,20,30)),('two',(10,80,210))]:
        path=root/(key+'.png');image=Image.new('RGB',(400,200),color);image.save(path)
        photos.append(dict(image_id=key,split='validation',source_photo_aliases=[key],image=str(path),
            stored_dimensions=[400,200],stored_image_sha256=file_hash(path),model_pixel_sha256_448=pixel_hash(image,448)))
        for index,(query,cell) in enumerate([('red flowers',57),('a green tree',122)] if key=='one' else [('blue sky',57)]):
            rows.append(dict(id=key+':'+str(index),image_id=key,split='validation',optimization_forbidden=True,
                query=query,query_input_ids=[1,*tokenizer.encode(query).ids,2],image_size=448,grid_side=14,
                attention_grid_histogram=[[cell,4]],source_point_count=4))
    for name,values in [('photos.jsonl',photos),('validation.jsonl',rows)]:
        (root/name).write_text(''.join(json.dumps(r)+'\n' for r in values))
    manifest=dict(files_sha256={n:file_hash(root/n) for n in ['photos.jsonl','validation.jsonl']},
        tokenizer_sha256=file_hash(TOKEN),complete_query_rows=dict(validation=3))
    (root/'manifest.json').write_text(json.dumps(manifest))
    proof=root/'unit-fixture-proof.json'
    proof.write_text(json.dumps(dict(candidate_manifest_sha256=file_hash(root/'manifest.json'),
        every_whole_original_query_source_record_offset_sha_times_and_point_index_reconstructed=True,
        every_actual_photo_header_EXIF_dimensions_pixels_creator_and_role_rechecked=True,
        every_query_BPE_BOS_content_EOS_and_448px_14x14_grid_count_reconstructed=True)))
    data=PointerData(root,'validation','evaluation',proof=proof,tokenizer=TOKEN)
    prior=fit_position_prior([dict(rows[0],id='train-only',split='train',optimization_forbidden=False)])
    config=ModelConfig(vocab_size=tokenizer.get_vocab_size(),hidden_size=16,intermediate_size=32,num_layers=1,
        num_heads=4,num_kv_heads=2,max_text_length=64,vision_hidden_size=8,vision_intermediate_size=16,
        vision_layers=1,vision_heads=2,image_size=448,patch_size=16,pool_size=2)
    model=PointGroundingHead(TinyMultimodalLM(config),rank=8);model.train();before=state_digest(model)
    output=tmp_path/'actual-own-small-model-diagnostic'
    result=evaluate(model,data,prior,output,batch_size=2)
    assert before==state_digest(model) and model.training and all(p.grad is None for p in model.parameters())
    conditions=result['conditions']
    assert all(conditions[k]['queries']==3 for k in ['proper','blank_image','wrong_image','uniform','center','training_position'])
    assert conditions['wrong_query']['queries']==2
    observed=[json.loads(line) for line in (output/'observations.jsonl').read_text().splitlines()]
    by_id={(r['id'],r['condition']):r for r in observed}
    for row in rows:
        proper=by_id[row['id'],'proper'];wrong=by_id[row['id'],'wrong_image'];blank=by_id[row['id'],'blank_image']
        assert proper['actual_query_input_ids']==wrong['actual_query_input_ids']==blank['actual_query_input_ids']==row['query_input_ids']
        assert proper['original_target_valid_cells']==wrong['original_target_valid_cells']==blank['original_target_valid_cells']
        assert proper['actual_image_id']!=wrong['actual_image_id'] and blank['actual_image_id'] is None
        assert len(proper['logits'])==196 and proper['actual_448px_RGB_sha256']!=blank['actual_448px_RGB_sha256']
    assert by_id['one:0','wrong_query']['actual_query_id']=='one:1'
    assert by_id['one:0','wrong_query']['actual_image_id']=='one'
    with pytest.raises(ValueError,match='Preserve previous'):evaluate(model,data,prior,output)
    data.purpose='optimization'
    rejected=tmp_path/'must-not-be-created'
    with pytest.raises(ValueError,match='validation evaluation reader'):evaluate(model,data,prior,rejected)
    assert not rejected.exists()
