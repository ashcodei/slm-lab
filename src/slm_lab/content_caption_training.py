"""Isolated content-pooled contrastive objective using the existing own trainer.

The experimental output wrapper is rebound only in this process. Shared trainer
files, active workers and default objectives remain unchanged.
"""
import argparse
from argparse import Namespace
from contextlib import contextmanager
import os
from pathlib import Path

import torch
from . import training
from .corpus import file_hash
from .diverse_images import atomic_json


def content_mean(hidden, attention_mask):
    lengths = attention_mask.sum(-1)
    if hidden.shape[:2] != attention_mask.shape or (lengths < 3).any():
        raise ValueError('Content pooling requires BOS, nonempty content and EOS')
    positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
    valid = attention_mask.bool() & (positions > 0) & (positions < lengths[:, None] - 1)
    return (hidden.float() * valid[:, :, None]).sum(1) / valid.sum(1)[:, None]


class ContentTrainingOutputs(training.TrainingOutputs):
    def forward(self, *args, **kwargs):
        if self.stage != 'contrastive' or args:
            raise ValueError('Content-caption wrapper supports keyword-only contrastive inputs')
        with torch.no_grad():
            hidden = self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask'],
                                return_hidden=True, return_logits=False)['hidden']
            text = content_mean(hidden, kwargs['attention_mask'])
        images = self.model.vision(kwargs['pixel_values'], self.model.gradient_checkpointing).mean(1)
        return {'image_features': images, 'text_features': text}


@contextmanager
def isolated_outputs():
    original = training.TrainingOutputs
    training.TrainingOutputs = ContentTrainingOutputs
    try:
        yield
    finally:
        training.TrainingOutputs = original


def train(args):
    if args.stage != 'contrastive' or args.trainable != 'vision' or int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This ablation requires single-GPU, vision-only contrastive training')
    implementation = file_hash(__file__)
    with isolated_outputs():
        training.train(args)
    if file_hash(__file__) != implementation:
        raise ValueError('Content-pooled objective implementation changed during training')
    atomic_json(Path(args.output) / 'caption-target.json',
                {'caption_feature_pooling': 'mean of content-token hidden states excluding BOS/EOS/padding',
                 'implementation_sha256': implementation, 'shared_trainer_sha256': file_hash(training.__file__),
                 'shared_trainer_modified': False, 'goal_achieved': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'tokenizer', 'data', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--steps', type=int, required=True); parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=5e-5); parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42); parser.add_argument('--device', default='cuda')
    parser.add_argument('--checkpoint-every', type=int, default=200)
    args = parser.parse_args()
    train(Namespace(**vars(args), stage='contrastive', trainable='vision', config='configs/model.json',
                    gradient_accumulation=1, threads=1, beta=.1, kl_coef=.05, ppo_epochs=2, max_new_tokens=16,
                    log_every=10, gradient_checkpointing=True, keep_checkpoints=1, resume=False,
                    contrastive_temperature=.07, reward_checkpoint=None))


if __name__ == '__main__':
    main()
