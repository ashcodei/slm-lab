"""All eight earlier visual families from the finalized full source bundle.

Paired questions are supplied separately by FinalizedVisualPairs. Text-only
families stay in the bound original bundle; a vision-only update cannot change
their text-only outputs. This reader never finalizes or substitutes sources.
"""
import json
from pathlib import Path

from .code_residual_collective_checkpoint_v2 import fingerprint
from .code_residual_source_contract_v4 import open_final_source_replay
from .codenet_checkpoint_v1 import FileBindings
from .codenet_image_replay_v1 import FAMILIES
from .codenet_replay_v1 import training_role
from .corpus import file_hash
from .model import ModelConfig


VISUAL_FAMILIES = ('image_reading', 'generic_visual') + FAMILIES


class FinalizedVisualReplay:
    def __init__(self, root, tokenizer, config, image_size=448):
        root = Path(root).resolve()
        folder = root / 'data/codenet-final-source-contract-v4'
        manifest = folder / 'manifest.json'
        if not manifest.exists():
            raise ValueError('Finalized complete30-family source bundle required before visual replay')
        m = json.loads(manifest.read_text())
        if (m.get('format') != 'own-complete-code-final-source-bundle-v4'
                or m.get('all30_actual_prepared_families_opened') is not True
                or m.get('full_original_paired_admission_completed') is not True
                or m.get('every_prepared_logical_record_reached') is not True):
            raise ValueError('Complete original sources and original paired admission required')
        token = root / 'data/tokenizer-corpus/tokenizer.json'
        from tokenizers import Tokenizer
        if (file_hash(token) != m['tokenizer_sha256']
                or json.loads(tokenizer.to_str()) != json.loads(Tokenizer.from_file(str(token)).to_str())
                or config != ModelConfig.load(root / 'configs/model.json') or type(image_size) is not int or image_size != 448):
            raise ValueError('Exact own BPE/config and full448px visual replay required')
        bindings = dict(m['source_files_sha256'])
        bindings.update({str(folder / n): h for n, h in m['artifact_files_sha256'].items()})
        bindings[str(manifest)] = file_hash(manifest)
        bindings[str(token)] = file_hash(token)
        bindings[str(root / 'configs/model.json')] = file_hash(root / 'configs/model.json')
        bindings[str(Path(__file__).resolve())] = file_hash(__file__)
        self.guard = FileBindings(bindings)
        self.replay, full_contract = open_final_source_replay(m['replay_folders'], m['tokenizer_sha256'])
        recorded = json.loads((folder / 'replay-contract.json').read_text())
        if full_contract != recorded or fingerprint(full_contract) != m['replay_contract_fingerprint']:
            raise ValueError('Exact complete original replay contract required')
        self.families = tuple(n for n in self.replay.families if n in VISUAL_FAMILIES)
        if set(self.families) != set(VISUAL_FAMILIES) or len(self.families) != 8:
            raise ValueError('All eight earlier original visual families required')
        self.contract = dict(format='own-finalized-earlier-visual-replay-v1',
            source_bundle_manifest_sha256=file_hash(manifest), full_source_contract=full_contract,
            tokenizer_sha256=file_hash(token), image_size=image_size,
            families=[dict(name=n, source_records=self.replay.size(n)) for n in self.families],
            implementation_sha256=file_hash(__file__), text_only_families_remain_bound=True,
            paired_questions_supplied_separately=True)
        self.contract_sha256 = fingerprint(self.contract)

    def size(self, name):
        if name not in self.families:
            raise ValueError('Original visual family required')
        return self.replay.size(name)

    def row(self, name, position):
        self.guard.check()
        if fingerprint(self.contract) != self.contract_sha256:
            raise ValueError('Bound original visual replay contract changed')
        self.size(name)
        row = self.replay.row(name, position)
        training_role(row)
        if not row.get('image') or not row.get('response'):
            raise ValueError('Whole original training image/response required')
        return row
