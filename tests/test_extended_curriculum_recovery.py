from copy import deepcopy
import pytest
from slm_lab.extended_curriculum_recovery import command, resume_matches, preserve_interrupted_output


def fixture():
    spec = {'name':'language-pretraining','stage':'pretrain','steps':17746,'data_sha256':'data',
            'trainable':'language','batch':1,'accumulation':16,'lr':.0001,'image_size':448,'data':'/data'}
    metadata = {'stage':'pretrain','steps':12000,'planned_steps':17746,'data_sha256':'data',
        'trainable':'language','tokenizer_sha256':'token','world_size':4,'optimizer':{},'torch_rng_state':[],
        'cuda_rng_state':[],'truncated_sequences':0,'training_options':{'batch_size':1,'gradient_accumulation':16,
            'learning_rate':.0001,'image_size':448,'gradient_checkpointing':True,'seed':42,'beta':.1,
            'kl_coef':.05,'ppo_epochs':2,'max_new_tokens':16,'contrastive_temperature':.07}}
    return spec, metadata


def test_resume_uses_actual_snapshot_step_and_exact_original_options():
    spec, metadata = fixture(); assert resume_matches(metadata, spec, 'token') == 12000
    call = command(spec, '/checkpoints/step-00012000.pt', '/output', True)
    assert call[-1] == '--resume' and '--nproc_per_node=4' in call
    assert call[call.index('--steps') + 1] == '17746'
    assert call[call.index('--gradient-accumulation') + 1] == '16'


@pytest.mark.parametrize('field,value', [('planned_steps',17747),('data_sha256','changed'),('world_size',2),
                                      ('steps',17746),('trainable','all')])
def test_changed_original_plan_or_worker_budget_rejected(field,value):
    spec, metadata = fixture(); metadata[field] = value
    with pytest.raises(ValueError): resume_matches(metadata, spec, 'token')


def test_no_weights_only_snapshot_or_changed_optimizer_schedule_resume():
    spec, metadata = fixture(); del metadata['optimizer']
    with pytest.raises(ValueError): resume_matches(metadata,spec,'token')
    spec, metadata = fixture(); metadata['training_options']['learning_rate'] = .0002
    with pytest.raises(ValueError): resume_matches(metadata,spec,'token')


def test_interrupted_snapshot_is_preserved_while_resume_output_becomes_fresh(tmp_path):
    directory = tmp_path / 'stage'; directory.mkdir(); source = directory / 'step-00012000.pt'
    source.write_bytes(b'exact saved optimizer and model bytes'); inode = source.stat().st_ino
    retained = preserve_interrupted_output(directory, source)
    assert not directory.exists() and retained.read_bytes() == b'exact saved optimizer and model bytes'
    assert retained.stat().st_ino == inode


def test_unrelated_source_is_never_moved(tmp_path):
    directory = tmp_path / 'stage'; directory.mkdir(); other = tmp_path / 'other.pt'; other.write_bytes(b'data')
    with pytest.raises(ValueError): preserve_interrupted_output(directory,other)
    assert directory.exists() and other.read_bytes() == b'data'
