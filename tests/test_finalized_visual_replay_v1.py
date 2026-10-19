"""Final visual reader interface fixture; full source opener is explicitly mocked."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest
from tokenizers import Tokenizer

from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
from slm_lab.code_residual_source_contract_v4 import REQUIRED_FAMILIES, COMPONENT_NAMES, build_required_contract
from slm_lab.codenet_runner_v1 import VerifiedReplayEncoder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import pixel_hash
from slm_lab.finalized_visual_replay_v1 import FinalizedVisualReplay, VISUAL_FAMILIES
import slm_lab.finalized_visual_replay_v1 as module
from test_codenet_runner_v1 import small_config

ROOT = Path(__file__).resolve().parents[1]


def test_bound_full_source_visual_reader_preserves_all_eight_families_and_complete_pixels_masks(tmp_path, monkeypatch):
    token = tmp_path / 'data/tokenizer-corpus/tokenizer.json'
    token.parent.mkdir(parents=True)
    shutil.copy2(ROOT / 'data/tokenizer-corpus/tokenizer.json', token)
    config = replace(small_config(), vocab_size=32768, max_text_length=4096,
        image_size=224, patch_size=16, pool_size=2)
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/model.json').write_text(json.dumps(asdict(config)))
    image = tmp_path / 'image.png'
    Image.new('RGB', (110, 90), (20, 170, 70)).save(image)
    with Image.open(image) as opened:
        golden = pixel_hash(opened, 448)

    class AdmittedDependencyFixture:
        families = REQUIRED_FAMILIES
        split = 'train'

        def size(self, name):
            assert name in self.families
            return 2

        def row(self, name, position):
            assert name in self.families and position in {0, 1}
            return dict(id=f'fixture:{name}:{position}', split=self.split, prompt='Describe this image.',
                response='Original complete fixture response.', image=str(image), model_pixel_sha256_448=golden)

    fixture = AdmittedDependencyFixture()
    contract = build_required_contract(fixture, file_hash(token), {n: 'd' * 64 for n in COMPONENT_NAMES})
    folder = tmp_path / 'data/codenet-final-source-contract-v4'
    folder.mkdir()
    (folder / 'replay-contract.json').write_text(json.dumps(contract))
    manifest = dict(format='own-complete-code-final-source-bundle-v4',
        fixture=True, real_source_admission=False, all30_actual_prepared_families_opened=True,
        full_original_paired_admission_completed=True, every_prepared_logical_record_reached=True,
        tokenizer_sha256=file_hash(token), replay_folders={n: str(tmp_path / n) for n in COMPONENT_NAMES},
        source_files_sha256={str(image): file_hash(image)},
        artifact_files_sha256={'replay-contract.json': file_hash(folder / 'replay-contract.json')},
        replay_contract_fingerprint=fingerprint(contract))
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(module, 'open_final_source_replay', lambda folders, digest: (fixture, contract))
    reader = FinalizedVisualReplay(tmp_path, Tokenizer.from_file(str(token)), config)
    assert set(reader.families) == set(VISUAL_FAMILIES) and len(reader.families) == 8
    encoder = VerifiedReplayEncoder(Tokenizer.from_file(str(token)), config, tmp_path, 448)
    for name in reader.families:
        assert reader.size(name) == 2
        for position in range(2):
            row = reader.row(name, position)
            ids, labels = encoder.sequence(row, 'response')
            batch = encoder.batch([row], 'response', 'cpu')
            assert batch['input_ids'][0].tolist() == ids and batch['labels'][0].tolist() == labels
            assert ids[-1] == labels[-1] == 2 and tuple(batch['pixel_values'].shape) == (1, 3, 448, 448)
    assert encoder.truncated == 0
    fixture.split = 'valid'
    with pytest.raises(ValueError, match='held/source role'):
        reader.row(reader.families[0], 0)
    fixture.split = 'train'
    reader.contract['families'][0]['source_records'] = 1
    with pytest.raises(ValueError, match='contract changed'):
        reader.row(reader.families[0], 0)
