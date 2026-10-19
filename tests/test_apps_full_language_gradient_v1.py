"""Independent serial global-token scalar check for the four-rank updater."""
import copy
from contextlib import nullcontext
import importlib.util
import math
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from tokenizers import Tokenizer, models

from slm_lab.apps_full_language_v1 import FullLanguageStage, HumanProgramEncoder
from slm_lab.code_residual_stage_v3 import replay_field
from slm_lab.codenet_stage_v1 import EpochOrder, freeze_for_code_stage
from slm_lab.model import ModelConfig, TinyMultimodalLM


def worker(rank, folder, step):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + str(Path(folder) / 'rendezvous'), rank=rank, world_size=4)
    try:
        fixture_path=Path(__file__).with_name('test_apps_full_language_v1.py')
        spec=importlib.util.spec_from_file_location('full_language_scalar_fixtures',fixture_path)
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        torch.manual_seed(731)
        config=ModelConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_layers=2,num_heads=4,
            num_kv_heads=2,max_text_length=64,vision_hidden_size=8,vision_intermediate_size=16,
            vision_layers=1,vision_heads=2,image_size=16,patch_size=4,pool_size=2)
        original=TinyMultimodalLM(config);raw=copy.deepcopy(original);params,_=freeze_for_code_stage(raw)
        tokenizer=Tokenizer(models.WordLevel({'<pad>':0,'<bos>':1,'<eos>':2,'<unk>':3,
            **{f'token{i}':i for i in range(4,32)}},unk_token='<unk>'))
        options=dict(seed=931001,code_accumulation=2,replay_accumulation=1,image_size=16,learning_rate=.003,
            weight_decay=.01,replay_coefficient=.5,gradient_clip=1.)
        source=fixture.Source(folder);replay=fixture.Replay();code=HumanProgramEncoder(source,tokenizer,config)
        replay_encoder=fixture.ReplayEncoder()
        optimizer=torch.optim.AdamW(params,lr=.003,weight_decay=.01,foreach=False)
        stage=FullLanguageStage(raw,optimizer,source,code,replay,replay_encoder,options,
            dict(base_checkpoint_sha256='a'*64,tokenizer_sha256='b'*64,run_plan_sha256='c'*64),'cpu',nullcontext)
        # step28 is a disposable fresh-Adam probe of the causal packed branch;
        # no noncontiguous state is saved or offered as a training candidate.
        stage.completed=step;actual=stage.update()
        reference=copy.deepcopy(original);reference_params,_=freeze_for_code_stage(reference)
        reference.train()
        family=replay.families[step%len(replay.families)];occurrence=step//len(replay.families)
        field=replay_field(family)
        code_batches=[];replay_batches=[]
        for r in range(4):
            for group in EpochOrder(len(source),options['seed']).rank_groups(step,r,4,1,2):
                code_batches.append(code.batch([dict(human_program_index=i) for i in group],'response','cpu'))
            for group in EpochOrder(replay.size(family),options['seed']+1000+replay.families.index(family)).rank_groups(occurrence,r,4,1,1):
                replay_batches.append(replay_encoder.batch([replay.row(family,i) for i in group],field,'cpu'))
        means=[]
        for batches,coefficient in [(code_batches,1.),(replay_batches,.5)]:
            counts=[int((b['labels'][:,1:]!=-100).sum()) for b in batches];total=sum(counts);scalar=0.
            for batch,count in zip(batches,counts):
                loss=reference(**batch)['loss'];(loss*(coefficient*count/total)).backward()
                scalar+=float(loss.detach())*coefficient*count/total
            means.append(scalar)
        norm=torch.nn.utils.clip_grad_norm_(reference_params,1.)
        lr=.003*min(1.,(step+1)/max(1,stage.steps//10))*(.1+.9*.5*(1+math.cos(math.pi*step/stage.steps)))
        ref_optimizer=torch.optim.AdamW(reference_params,lr=lr,weight_decay=.01,foreach=False)
        ref_optimizer.step()
        assert actual['summed_objective']==pytest.approx(sum(means),rel=1e-6,abs=1e-7)
        assert actual['gradient_norm_before_clipping']==pytest.approx(float(norm),rel=1e-5,abs=1e-6)
        for (name,observed),(_,expected) in zip(raw.named_parameters(),reference.named_parameters()):
            torch.testing.assert_close(observed,expected,rtol=5e-5,atol=5e-7,msg=name)
        for observed,expected in zip(optimizer.state.values(),ref_optimizer.state.values()):
            for key in ['exp_avg','exp_avg_sq']:
                torch.testing.assert_close(observed[key],expected[key],rtol=1e-5,atol=1e-7)
        dist.barrier()
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('step',[0,28])
def test_four_worker_whole_language_Adam_matches_independent_serial_global_scalar(tmp_path,step):
    mp.spawn(worker,args=(str(tmp_path),step),nprocs=4,join=True)
