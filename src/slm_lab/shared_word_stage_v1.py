"""Train shared vision factors and a byte reader on complete original words.

CTC uses native UTF-8 bytes0..255/blank256, equal word weights after dividing
each whole-sequence CTC loss by its byte count. Native image-conditioned CE
uses the global response/EOS token mean. Backward requests only vision/word
leaves; it retains derivatives through the fixed language computation without
changing any parameter's trainability metadata or copying another backbone.
"""
from collections import Counter
from contextlib import nullcontext
import math

import torch
from torch.nn import functional as F

from .codenet_stage_v1 import EpochOrder
from .own_aligned_assistant_v3 import OwnAlignedAssistantV3
from .own_alignment_checkpoint_v1 import make_state, parameters, restore_state
from .shared_word_data_v1 import SharedWordData, native_bytes

FORMAT = 'own-shared-vision-whole-word-CTC-native-CE-Adam-v1'
PREFIXES = ('vision_linears.', 'vision_norms.', 'vision_projection.', 'word.')
GROUPS = ('vision', 'word')


def ctc_word_sum(logits, texts):
    """Return the sum of byte-normalized complete word losses, never clipped.

    CPU CTC avoids an unsupported/nondeterministic CUDA reduction. The copy
    keeps its autograd connection to the actual GPU vision/word parameters.
    """
    if logits.ndim != 3 or logits.shape[-1] != 257 or len(texts) != logits.shape[0] or not len(texts):
        raise ValueError('One complete byte transcript per actual 257-class word sequence required')
    targets = [native_bytes(text) for text in texts]
    if any(required > logits.shape[1] for _, required in targets):
        raise ValueError('Every complete transcript and repeated byte must fit the actual CTC frames')
    lengths = torch.tensor([len(values) for values, _ in targets], dtype=torch.long)
    packed = torch.tensor([v for values, _ in targets for v in values], dtype=torch.long)
    logp = logits.float().cpu().log_softmax(-1).transpose(0, 1)
    raw = F.ctc_loss(logp, packed, torch.full_like(lengths, logits.shape[1]), lengths,
                     blank=256, reduction='none', zero_infinity=False)
    loss = (raw / lengths).sum()
    if not bool(torch.isfinite(loss)):
        raise ValueError('Complete CTC targets must have a finite unreplaced loss')
    return loss, int(lengths.sum())


def accumulate_gradients(model, batches, *, ctc_weight, native_CE_weight, amp=None):
    """Accumulate exactly the separately normalized CTC/CE joint objective."""
    if (type(model) is not OwnAlignedAssistantV3 or not batches
            or any(not b.get('texts') for b in batches)
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                   for v in (ctc_weight, native_CE_weight))):
        raise ValueError('Actual shared v3 model, whole word batches and positive finite objective weights required')
    selected = model.extension_parameters(*GROUPS)
    word_count = sum(len(batch['texts']) for batch in batches)
    token_count = sum(int((batch['labels'][:, 1:] != -100).sum()) for batch in batches)
    if not token_count:
        raise ValueError('Every native whole word needs its response and real EOS targets')
    amp = amp or nullcontext
    ctc_total = ce_total = 0.; byte_count = context_count = 0
    for batch in batches:
        if (batch['input_ids'].shape != batch['labels'].shape
                or batch['attention_mask'].shape != batch['labels'].shape
                or len(batch['texts']) != len(batch['input_ids'])):
            raise ValueError('Complete corresponding image, transcript, native token and response masks required')
        with amp():
            word_logits = model.word_logits(batch['word_pixels'])
        ctc, byte_targets = ctc_word_sum(word_logits, batch['texts'])
        # Explicit inputs is essential: the fixed new language leaves retain
        # requires_grad metadata, but must receive no gradients. This also
        # works with non-reentrant activation-checkpoint recomputation.
        (ctc * (ctc_weight / word_count)).backward(inputs=selected)
        ctc_total += float(ctc.detach()); byte_count += byte_targets
        del word_logits, ctc
        tokens = int((batch['labels'][:, 1:] != -100).sum())
        inputs = {k: batch[k] for k in ('input_ids', 'attention_mask', 'pixel_values', 'labels')}
        with amp():
            mean_ce = model.aligned(**inputs)['loss']
        if not bool(torch.isfinite(mean_ce)):
            raise ValueError('Every complete native word CE must be finite before an optimizer update')
        (mean_ce * (native_CE_weight * tokens / token_count)).backward(inputs=selected)
        ce_total += float(mean_ce.detach()) * tokens
        context_count += int(batch['attention_mask'].sum())
        del mean_ce
    result = dict(words=word_count, UTF8_byte_targets=byte_count,
                  complete_context_tokens=context_count, response_EOS_targets=token_count,
                  word_mean_byte_normalized_CTC=ctc_total / word_count,
                  token_mean_native_CE=ce_total / token_count)
    result['summed_objective'] = ctc_weight * result['word_mean_byte_normalized_CTC'] + native_CE_weight * result['token_mean_native_CE']
    return result


class _VisionWordRestoreView:
    """Validate the full export; copy only trained vision/word tensors."""
    def __init__(self, model): self.model = model

    def __getattr__(self, name): return getattr(self.model, name)

    def validate_export(self, payload, *, expected_sources):
        self.model.validate_export(payload, expected_sources=expected_sources)
        if any(not torch.equal(t.detach().cpu(), payload['extensions'][name].detach().cpu())
               for name, t in self.model.parts.state_dict().items() if not name.startswith(PREFIXES)):
            raise ValueError('Every unused language, reward and critic tensor must remain fixed before resume')

    def import_export(self, payload, *, expected_sources):
        self.validate_export(payload, expected_sources=expected_sources)
        with torch.no_grad():
            for name, p in self.model.parts.named_parameters():
                if name.startswith(PREFIXES): p.copy_(payload['extensions'][name])


class SharedWordStage:
    def __init__(self, model, optimizer, data, options, sources, device='cpu', amp=None):
        if (type(model) is not OwnAlignedAssistantV3 or type(data) is not SharedWordData
                or (data.role, data.purpose) != ('train', 'optimization') or not len(data)):
            raise ValueError('Exact shared v3 graph and independently admitted train-only word source required')
        keys = {'seed', 'epochs', 'steps', 'microbatch', 'accumulation', 'learning_rate',
                'betas', 'weight_decay', 'gradient_clip', 'ctc_weight', 'native_CE_weight'}
        if (set(options) != keys or any(type(options[k]) is not int or options[k] < 1
                                       for k in ('epochs', 'steps', 'microbatch', 'accumulation'))
                or type(options['seed']) is not int
                or any(type(options[k]) not in (int, float) or not math.isfinite(options[k]) or options[k] <= 0
                       for k in ('learning_rate', 'weight_decay', 'gradient_clip', 'ctc_weight', 'native_CE_weight'))
                or len(options['betas']) != 2 or any(not 0 <= v < 1 for v in options['betas'])):
            raise ValueError('Explicit complete-pass exposure and finite fixed word objective/Adam options required')
        self.width = options['microbatch'] * options['accumulation']
        self.steps_per_epoch = math.ceil(len(data) / self.width)
        if options['steps'] != options['epochs'] * self.steps_per_epoch:
            raise ValueError('Every complete original source pass including its unrepeated final batch required')
        self.model, self.optimizer, self.data = model, optimizer, data
        self.options, self.sources, self.device = dict(options), dict(sources), torch.device(device)
        self.amp = amp or nullcontext
        self.parameters = parameters(model, optimizer, GROUPS)
        group = optimizer.param_groups[0]
        if (group['lr'] != options['learning_rate'] or group['weight_decay'] != options['weight_decay']
                or tuple(group['betas']) != tuple(options['betas'])):
            raise ValueError('Exact ordinary vision/word-only AdamW hyperparameters required')
        selected = {id(p) for p in self.parameters}
        self.fixed = [p for p in model.parameters() if id(p) not in selected]
        self.order = EpochOrder(len(data), options['seed']); self.completed = 0; self.metrics = []

    def positions(self, step):
        if type(step) is not int or not 0 <= step < self.options['steps']:
            raise ValueError('Exact bounded whole-word update required')
        epoch, within = divmod(step, self.steps_per_epoch)
        start, stop = within * self.width, min((within + 1) * self.width, len(self.data))
        positions = [self.order.at(epoch * len(self.data) + p) for p in range(start, stop)]
        micro = self.options['microbatch']
        return [positions[i:i + micro] for i in range(0, len(positions), micro)]

    def cursor(self, completed):
        epochs, updates = divmod(completed, self.steps_per_epoch)
        draws = epochs * len(self.data) + min(updates * self.width, len(self.data))
        return dict(completed_updates=completed, complete_passes=epochs,
                    next_pass_draws=draws - epochs * len(self.data), actual_draws=draws,
                    source_population=len(self.data), seed=self.options['seed'],
                    microbatch=self.options['microbatch'], accumulation=self.options['accumulation'],
                    repeated_rounding_sources=0, world_size=1)

    def source_counts(self, positions):
        rows = [self.data.rows[p] for p in positions]
        return dict(words=len(rows), UTF8_byte_targets=sum(len(r['response'].encode('utf-8')) for r in rows),
                    complete_context_tokens=sum(r['native']['length'] for r in rows),
                    response_EOS_targets=sum(r['native']['length'] - r['native']['target_start'] for r in rows),
                    by_family=dict(Counter(r['byte_family'] for r in rows)))

    def update(self):
        groups = self.positions(self.completed); positions = [p for group in groups for p in group]
        expected = self.source_counts(positions)
        self.model.assert_word_contract(); self.model.aligned.train(); self.model.original.eval()
        self.optimizer.zero_grad(set_to_none=True)
        try:
            if any(p.grad is not None for p in self.fixed):
                raise ValueError('No fixed language/original/reward/critic gradient may enter word learning')
            batches = [self.data.batch(group, self.device) for group in groups]
            record = accumulate_gradients(self.model, batches, ctc_weight=self.options['ctc_weight'],
                                          native_CE_weight=self.options['native_CE_weight'], amp=self.amp)
            if (any(record[k] != expected[k] for k in expected if k != 'by_family')
                    or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters)
                    or any(p.grad is not None for p in self.fixed)):
                raise ValueError('Every complete word/target and every and only vision/word gradient required')
            norm = float(torch.nn.utils.clip_grad_norm_(self.parameters, self.options['gradient_clip'], error_if_nonfinite=True))
            self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True); raise
        self.completed += 1
        record.update(step=self.completed, source_positions=positions, by_family=expected['by_family'],
                      learning_rate=self.options['learning_rate'], gradient_norm_before_clip=norm)
        self.metrics.append(record); return record

    def state(self):
        core = make_state(self.model, self.optimizer, groups=GROUPS, sources=self.sources,
                          completed_steps=self.completed, planned_steps=self.options['steps'],
                          sampler_state=self.cursor(self.completed), metrics=self.metrics, device=self.device)
        return dict(format=FORMAT, core=core, options=self.options, source_population=len(self.data),
                    candidate_promoted=False, goal_achieved=False)

    def resume(self, state, *, allow_completed=False):
        core = state.get('core', {}); steps = core.get('steps')
        if (state.get('format') != FORMAT or state.get('options') != self.options
                or state.get('source_population') != len(self.data) or type(steps) is not int
                or not 0 < steps <= self.options['steps'] or len(core.get('metrics', [])) != steps
                or state.get('candidate_promoted') is not False or state.get('goal_achieved') is not False):
            raise ValueError('Exact original whole-word options, population and complete update budget required')
        for step, metric in enumerate(core['metrics']):
            positions = [p for group in self.positions(step) for p in group]
            counts = self.source_counts(positions)
            if (metric.get('step') != step + 1 or metric.get('source_positions') != positions
                    or any(metric.get(k) != v for k, v in counts.items())
                    or metric.get('learning_rate') != self.options['learning_rate']
                    or any(type(metric.get(k)) not in (int, float) or not math.isfinite(metric[k]) or metric[k] < 0
                           for k in ('word_mean_byte_normalized_CTC', 'token_mean_native_CE', 'gradient_norm_before_clip'))
                    or metric.get('summed_objective') != self.options['ctc_weight'] * metric['word_mean_byte_normalized_CTC']
                        + self.options['native_CE_weight'] * metric['token_mean_native_CE']):
                raise ValueError('Every complete word, family, target normalizer and joint objective must reconstruct')
        self.completed, self.metrics = restore_state(_VisionWordRestoreView(self.model), self.optimizer, core,
            groups=GROUPS, expected_sources=self.sources, planned_steps=self.options['steps'],
            expected_sampler_state=self.cursor(steps), device=self.device, allow_completed=allow_completed)
