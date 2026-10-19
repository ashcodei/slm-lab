"""Complete admitted public causal windows plus exact previous replay exposures.

All new raw train windows occur once. Every selected presentation from the
existing code/state/replay continuation is retained, including any original
replay repetitions. Rebatching changes update grouping, not source exposure.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random

from .corpus import file_hash
from .public_causal_windows_v1 import CausalPublicPack

RAW = 'raw:public_code_math_web'


class PublicPretrainingData:
    role = 'train'
    purpose = 'optimization'

    def __init__(self, root, config):
        from .focused_code_continuation_data_v3 import FocusedCodeContinuationData, FocusedContinuationSchedule
        self.root = Path(root).resolve()
        if config.max_text_length != 4096:
            raise ValueError('The actual complete native4096 context configuration is required')
        folder = self.root/'data/public-billion-causal-windows-v1'
        self.raw = CausalPublicPack(folder, 'train', 'optimization')
        if self.raw.manifest['roles']['train']['target_tokens'] < 10**9:
            raise ValueError('No partial public candidate may replace the complete billion-target corpus')
        self.retention = FocusedCodeContinuationData(self.root, config, purpose='optimization')
        prior = self.root/'reports/code-first-continuation-v3/plan.json'
        plan = json.loads(prior.read_text())
        original = FocusedContinuationSchedule(self.retention, seed=plan['seed'], width=plan['global_batch_width'])
        if (original.identity_sha256 != plan['source_schedule_sha256']
                or original.selected_budget(self.retention) != plan['selected_source_budget']):
            raise ValueError('Every exact previous code/state/12family replay presentation must reconstruct')
        self.retention_entries = [(n, list(p)) for n,p in original.entries]
        self.retention_identity_sha256 = original.identity_sha256
        self.lengths = dict(self.retention.lengths)
        self.targets = dict(self.retention.targets)
        self.kinds = dict(self.retention.kinds)
        self.lengths[RAW] = [int(v) for v in self.raw.index['length']]
        self.targets[RAW] = [int(v) for v in self.raw.index['targets']]
        self.kinds[RAW] = 'text'
        self.populations = {n:len(v) for n,v in self.lengths.items()}
        self.code = self.retention.code
        self.source_files_sha256 = dict(self.retention.source_files_sha256)
        self.source_files_sha256.update(self.raw.manifest['source_files_sha256'])
        for path in [prior, folder/'manifest.json', folder/'admission.json']:
            self.source_files_sha256[str(path)] = file_hash(path)
        admission = json.loads((folder/'admission.json').read_text())
        path = Path(admission['independent_readback'])
        self.source_files_sha256[str(path)] = admission['independent_readback_sha256']
        tokenizer = self.root/'data/tokenizer-corpus/tokenizer.json'
        if (file_hash(tokenizer) != self.code.provenance['tokenizer_sha256']
                or self.source_files_sha256.get(str(tokenizer)) != file_hash(tokenizer)):
            raise ValueError('New raw and all retained sources must use the identical own BPE tokenizer')

    def batch(self, name, positions, device='cpu'):
        if name not in self.populations or any(type(p) is not int or not 0 <= p < self.populations[name] for p in positions):
            raise ValueError('Explicit complete original train-source positions required')
        if not positions:return None
        result = self.raw.batch(positions, device) if name == RAW else self.retention.batch(name, positions, device)
        if (int(result['attention_mask'].sum()) != sum(self.lengths[name][p] for p in positions)
                or int((result['labels'][:,1:] != -100).sum()) != sum(self.targets[name][p] for p in positions)):
            raise ValueError('Every original context and text/EOS target must reach the actual batch')
        return result

    def budget(self):
        return {n:dict(examples=len(v), complete_context_tokens=sum(v), response_EOS_targets=sum(self.targets[n]), kind=self.kinds[n])
                for n,v in self.lengths.items()}


class TokenBudgetSchedule:
    """Bound padded compute cost while preserving all selected source occurrences."""
    def __init__(self, data, *, image_prefix=784, context_budget=262144, maximum_rows=1024, seed=976319):
        if (data.role != 'train' or data.purpose != 'optimization' or RAW not in data.lengths
                or type(image_prefix) is not int or image_prefix < 0
                or any(type(v) is not int or v < 1 for v in [context_budget, maximum_rows]) or type(seed) is not int):
            raise ValueError('Admitted train sources and explicit whole-example compute bounds required')
        exposures = defaultdict(list)
        for name, positions in data.retention_entries:
            if name == RAW:raise ValueError('Previous retention cannot silently contain the new public raw source')
            exposures[name].extend(positions)
        exposures[RAW] = list(range(len(data.lengths[RAW])))
        if any(not p for p in exposures.values()):raise ValueError('Every declared retained/raw source must remain nonempty')
        self.entries = []
        ordering = []
        for number, name in enumerate(sorted(exposures)):
            prefix = image_prefix if data.kinds[name] in {'image','word'} else 0
            original = exposures[name]
            if (name not in data.targets or len(data.lengths[name]) != len(data.targets[name])
                    or any(type(p) is not int or not 0 <= p < len(data.lengths[name]) for p in original)
                    or any(data.targets[name][p] < 1 or data.targets[name][p] >= data.lengths[name][p]
                           or data.lengths[name][p]+prefix > context_budget for p in original)):
                raise ValueError('Every original whole context/target must fit; no source clipping or dropping')
            rng = random.Random(seed+number*1009)
            occurrences = list(range(len(original)));rng.shuffle(occurrences)
            groups = []
            bucket_size = max(32, min(32768, maximum_rows*32))
            for begin in range(0, len(occurrences), bucket_size):
                bucket = sorted(occurrences[begin:begin+bucket_size], key=lambda i:(data.lengths[name][original[i]], i))
                group = [];maximum = 0
                for i in bucket:
                    position = original[i];length = data.lengths[name][position]+prefix
                    if group and (len(group) == maximum_rows or max(maximum,length)*(len(group)+1) > context_budget):
                        groups.append(group);group = [];maximum = 0
                    group.append(position);maximum = max(maximum,length)
                if group:groups.append(group)
            rng.shuffle(groups)
            actual = [p for group in groups for p in group]
            if Counter(actual) != Counter(original):raise ValueError('Rebatching must preserve every source occurrence, including original repeats')
            for i, group in enumerate(groups):
                ordering.append((Fraction(2*i+1,2*len(groups)), number, name, group))
        self.entries = [(n,list(p)) for _,_,n,p in sorted(ordering)]
        self.settings = dict(image_prefix=image_prefix, context_budget=context_budget, maximum_rows=maximum_rows, seed=seed)
        self.identity_sha256 = hashlib.sha256(json.dumps(dict(settings=self.settings, entries=self.entries,
            retention_identity=data.retention_identity_sha256), separators=(',',':'), sort_keys=True).encode()).hexdigest()
        self.source_exposures = {n:len(p) for n,p in exposures.items()}

    def __len__(self):return len(self.entries)

    def at(self, step):
        if type(step) is not int or not 0 <= step < len(self):raise IndexError('Complete public-pretraining source schedule exhausted')
        name, positions = self.entries[step]
        return name, list(positions)

    def selected_budget(self, data):
        totals = {}
        for name, positions in self.entries:
            row = totals.setdefault(name, dict(presentations=0, complete_context_tokens=0, response_EOS_targets=0, updates=0))
            row['presentations'] += len(positions)
            row['complete_context_tokens'] += sum(data.lengths[name][p] for p in positions)
            row['response_EOS_targets'] += sum(data.targets[name][p] for p in positions)
            row['updates'] += 1
        return totals
