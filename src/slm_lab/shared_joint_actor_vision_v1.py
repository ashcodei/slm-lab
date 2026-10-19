"""One under1B own policy with explicit text/image/word optimizer selections.

Reuse the actual encoder and all counted native heads. Native captions can
differentiate both the language adapter and shared vision; word CTC omits the
unused vision-to-language projection. Complete owned training, durable joint
state, four-GPU capacity and final quality remain separate requirements.
Earlier actor-only snapshots cannot restore the newly mutable visual values.
"""
from torch import nn

from .own_actor_vision_prefix_reuse_v1 import OwnActorVisionPrefixReuse
from .own_dense_actor_v1 import OwnDenseActor

FORMAT = 'own-shared-joint-policy-vision-word-gradient-view-v1'


class SharedJointActorVisionV1(nn.Module):
    def __init__(self, actor):
        super().__init__()
        if type(actor) is not OwnDenseActor: raise ValueError('Exactly one own complete dense policy is required')
        self.actor = actor
        self.alignment = OwnActorVisionPrefixReuse(actor)
        self.assert_shared()

    def assert_shared(self):
        self.actor.assert_shared(); self.alignment.assert_shared()
        if ({id(p) for p in self.parameters()} != {id(p) for p in self.actor.parameters()}
                or sum(p.numel() for p in self.parameters()) != self.actor.counts()['actual_unique']):
            raise ValueError('Every joint path must share the same counted under1B parameters')

    def named_selected(self):
        groups = [('actor', self.actor.named_selected(('actor',))),
            ('vision', self.actor.base.network.vision.named_parameters()),
            ('pointing', self.actor.base.heads['pointing'].named_parameters()),
            # The scalar height bias is a uniform shift before a softmax
            # across height. Its mathematical derivative is zero. Keep its
            # existing value fixed so Adam cannot amplify FP32 cancellation
            # noise differently across full and sharded word batches.
            ('word', ((n, p) for n, p in self.actor.base.heads['word'].named_parameters() if n != 'height.bias'))]
        result = [(prefix + '.' + n, p) for prefix, values in groups for n, p in values]
        if len({id(p) for _, p in result}) != len(result): raise ValueError('Optimizer selections cannot duplicate a shared parameter')
        return result

    def task_names(self, task):
        if task not in ['text', 'native_image', 'image_matching', 'word_CTC']:
            raise ValueError('An explicit complete native objective role is required')
        prefixes = dict(text=('actor.',), native_image=('actor.', 'vision.'),
            image_matching=('actor.', 'vision.', 'pointing.'), word_CTC=('vision.', 'word.'))[task]
        return [n for n, _ in self.named_selected() if n.startswith(prefixes)
            and not (task == 'word_CTC' and n.startswith('vision.project.'))]

    def text(self, **batch):
        if batch.get('pixel_values') is not None: raise ValueError('A text-only primary anchor cannot acquire image inputs')
        return self.actor(**batch)

    def native_image(self, **batch):
        if batch.get('pixel_values') is None: raise ValueError('A native image task requires the complete original pixels')
        return self.alignment.native(**batch)

    def image_matching(self, caption_batch, text_ids, text_mask, positive, **options):
        return self.alignment.objective(caption_batch, text_ids, text_mask, positive, **options)

    def word_logits(self, pixels): return self.actor.base.word_logits(pixels)
