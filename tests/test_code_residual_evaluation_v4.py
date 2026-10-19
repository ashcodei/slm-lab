"""Real own-BPE/four-worker export through unchanged text/code/image evaluators."""
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from slm_lab import (knowledge_evaluation, math_evaluation, ocr_evaluation,
    public_code_benchmark)
import slm_lab.code_residual_evaluation_v4 as bridge
from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
from slm_lab.code_residual_export_v2 import export_inference, load_inference
from slm_lab.code_residual_source_contract_v4 import (COMPONENT_NAMES, SourceContractResidualKernel,
    build_required_contract)
from slm_lab.code_residual_stage_v3 import ExpandedReplayEncoder
from slm_lab.codenet_token_pack_v1 import packed_encoder
from slm_lab.corpus import file_hash
from slm_lab.model import TinyMultimodalLM
from slm_lab.recurrent_ablation import state_digest
from slm_lab.training import load_checkpoint

from test_code_residual_source_contract_v4 import Code, Replay
from test_codenet_runner_v1 import options, small_config


def training_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4,
        timeout=timedelta(seconds=90))
    try:
        folder = Path(directory)
        token = Tokenizer.from_file(str(folder / 'tokenizer.json'))
        base, _ = load_checkpoint(folder / 'base.pt', 'cpu', training_state=False)
        torch.manual_seed(7011)
        model = RoutedCodeResidual(base, width=12)
        opt = {**options(), 'passes': 1, 'gradient_accumulation': 1, 'replay_accumulation': 1, 'image_size': 16}
        optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=opt['learning_rate'],
            weight_decay=opt['weight_decay'], foreach=False)
        replay, code = Replay(folder), Code(folder)
        digest = file_hash(folder / 'tokenizer.json')
        contract = build_required_contract(replay, digest,
            {name: hashlib.sha256(('fixture:' + name).encode()).hexdigest() for name in COMPONENT_NAMES})
        sources = dict(base_checkpoint_sha256=file_hash(folder / 'base.pt'), tokenizer_sha256=digest,
            run_plan_sha256=file_hash(folder / 'plan.json'))
        kernel = SourceContractResidualKernel(model, optimizer, code, packed_encoder(code, token, model.config),
            replay, ExpandedReplayEncoder(token, model.config, folder, 16), opt, sources, 'cpu', nullcontext,
            replay_contract=contract, expected_contract_fingerprint=fingerprint(contract))
        original = state_digest(model.base.state_dict())
        assert kernel.steps == 30
        for _ in range(kernel.steps):
            kernel.update()
        assert len({m['replay_family'] for m in kernel.metrics}) == 30
        assert state_digest(model.base.state_dict()) == original
        receipt = kernel.snapshot(folder / 'complete.pt', maximum_bytes=2_000_000, minimum_free_bytes=0)
        (folder / f'worker-{rank}.json').write_text(json.dumps(dict(actual_CPU_updates=30,
            actual_CPU_world_size=4, all30_fixture_families_exercised=True,
            own_real_BPE_tokenizer_used=True, model_counts=model.parameter_counts(),
            frozen_base_unchanged=True, complete_checkpoint_sha256=receipt['checkpoint_sha256'],
            actual857M_training_or_capability_result=False)))
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope='module')
def completed_subject(tmp_path_factory):
    folder = tmp_path_factory.mktemp('own-bpe-evaluation-subject')
    token = Tokenizer(models.BPE(unk_token='<unk>'))
    token.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    token.decoder = decoders.ByteLevel()
    token.train_from_iterator(['User: Read the text. Assistant: red blue abc numbers code',
        'def identity(value): return value', 'The following is a multiple-choice question about physics.',
        'Answer with one letter only. Answer: A. B. C. D. 0123456789',
        'Write a Python function. Use built-in types and public math imports. ####'],
        trainers.BpeTrainer(vocab_size=320, special_tokens=['<pad>', '<bos>', '<eos>', '<unk>'],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    token.save(str(folder / 'tokenizer.json'))
    config = replace(small_config(), vocab_size=token.get_vocab_size(), max_text_length=512)
    torch.manual_seed(9441)
    torch.save(dict(model=TinyMultimodalLM(config).state_dict(), config=asdict(config), fixture_auxiliary_tensor=torch.tensor([9]),
        tokenizer_sha256=file_hash(folder / 'tokenizer.json'), stage='own CPU bridge fixture'), folder / 'base.pt')
    (folder / 'manifest.json').write_text(json.dumps(dict(fixture=True, production_source_admission=False)))
    (folder / 'plan.json').write_text(json.dumps(dict(fixture=True, production_training_ready=False,
        base_checkpoint_sha256=file_hash(folder / 'base.pt'), tokenizer_sha256=file_hash(folder / 'tokenizer.json'),
        code_manifest_sha256=file_hash(folder / 'manifest.json'))))
    Image.new('RGB', (8, 4), (21, 55, 90)).save(folder / 'image.png')
    Image.new('RGB', (8, 4), (220, 85, 10)).save(folder / 'other.png')
    mp.spawn(training_worker, args=(str(folder / 'rendezvous'), str(folder)), nprocs=4, join=True)
    export_inference(folder / 'complete.pt', folder / 'base.pt', folder / 'tokenizer.json',
        folder / 'plan.json', folder / 'manifest.json', folder / 'inference.pt',
        expected_training_checkpoint_sha256=file_hash(folder / 'complete.pt'), maximum_bytes=1_000_000, minimum_free_bytes=0)
    subject = dict(format=bridge.FORMAT,
        expected_model_counts=json.loads((folder / 'worker-0.json').read_text())['model_counts'])
    for key, name in [('inference_checkpoint', 'inference.pt'), ('base_checkpoint', 'base.pt'),
            ('tokenizer', 'tokenizer.json'), ('run_plan', 'plan.json'),
            ('training_checkpoint', 'complete.pt'), ('source_manifest', 'manifest.json')]:
        subject[key] = str(folder / name)
        subject[key + '_sha256'] = file_hash(folder / name)
    (folder / 'subject.json').write_text(json.dumps(subject))
    return subject


def direct_code(subject):
    return load_inference(subject['inference_checkpoint'], subject['base_checkpoint'], subject['tokenizer'],
        expected_checkpoint_sha256=subject['inference_checkpoint_sha256'],
        expected_base_sha256=subject['base_checkpoint_sha256'], expected_plan_sha256=subject['run_plan_sha256'],
        expected_training_checkpoint_sha256=subject['training_checkpoint_sha256'],
        expected_data_sha256=subject['source_manifest_sha256'])[0]


def test_exact_linked_views_keep_general_and_code_text_image_logits_and_cached_generation(completed_subject, tmp_path):
    subject = completed_subject
    direct = {'general': load_checkpoint(subject['base_checkpoint'], 'cpu', training_state=False)[0],
        'code': direct_code(subject)}
    tok = Tokenizer.from_file(subject['tokenizer'])
    ids = torch.tensor([[tok.token_to_id('<bos>')] + tok.encode('User: abc\nAssistant:').ids])
    pixels = torch.linspace(-1, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
    logits = {}
    for route in ['general', 'code']:
        view = tmp_path / route
        checkpoint = bridge.prepare_view(subject, view, route)
        key = 'inference_checkpoint' if route == 'code' else 'base_checkpoint'
        assert checkpoint.stat().st_ino == Path(subject[key]).stat().st_ino
        loader = bridge.BoundLoader(subject, view, 'cpu')
        actual, metadata = loader(checkpoint.resolve(), device='cpu', training_state=False)
        direct[route].eval()
        before = state_digest(actual.state_dict())
        with torch.no_grad():
            for image in [None, pixels]:
                expected = direct[route](ids, pixel_values=image)['logits']
                assert torch.equal(actual(ids, pixel_values=image)['logits'], expected)
                assert torch.equal(actual.generate(ids, image, max_new_tokens=3, use_cache=True),
                    direct[route].generate(ids, image, max_new_tokens=3, use_cache=True))
            logits[route] = actual(ids)['logits']
        assert state_digest(actual.state_dict()) == before
        assert not any(p.requires_grad for p in actual.parameters())
        assert metadata['tokenizer_sha256'] == subject['tokenizer_sha256']
    assert not torch.equal(logits['general'], logits['code'])


def test_unchanged_native_knowledge_and_math_evaluators_run_both_routes(completed_subject, tmp_path):
    subject = completed_subject
    questions = tmp_path / 'questions.jsonl'
    questions.write_text(json.dumps(dict(id='fixture-knowledge', domain='physics', question='Which is a number?',
        options=['5', 'blue'], answer_index=0)) + '\n')
    questions.with_suffix('.audit.json').write_text(json.dumps(dict(clean_questions_sha256=file_hash(questions), fixture=True)))
    data = tmp_path / 'math'
    data.mkdir()
    math_data = data / 'test.jsonl'
    math_data.write_text(json.dumps(dict(id='fixture-math', question='What is 2 plus 3?', answer_numerator=5, answer_denominator=1)) + '\n')
    (data / 'manifest.json').write_text(json.dumps(dict(files_sha256={'test.jsonl': file_hash(math_data)},
        official_test_examples=1, limitations=['Test fixture; not official GSM8K data.'])))
    originals = [knowledge_evaluation.load_checkpoint, math_evaluation.load_checkpoint]
    for route in ['general', 'code']:
        view = tmp_path / (route + '-view')
        bridge.prepare_view(subject, view, route)
        output = tmp_path / (route + '-knowledge.json')
        receipt = tmp_path / (route + '-knowledge-loader.json')
        bridge.evaluate_bound(subject, view, knowledge_evaluation, output, receipt,
            arguments=dict(data=questions, max_examples=1, max_new_tokens=2))
        report = json.loads(output.read_text())
        proof = json.loads(receipt.read_text())
        assert report['overall']['questions'] == 1 and not report['skipped']
        assert report['generation_budget'] == 2 and report['checkpoint_sha256'] == file_hash(view / 'model.pt')
        assert proof['report_files_sha256'][str(output)] == file_hash(output)
        assert proof['route'] == route and proof['complete_capability_suite_certified'] is False
        assert 'fixture_auxiliary_tensor' not in proof['loaded_metadata'][0]
        assert proof['evaluator_arguments']['data'] == str(questions)
        bridge.evaluate_bound(subject, view, math_evaluation, tmp_path / (route + '-math.json'),
            tmp_path / (route + '-math-loader.json'), arguments=dict(data=str(math_data), max_new_tokens=2))
        math_report = json.loads((tmp_path / (route + '-math.json')).read_text())
        assert math_report['evaluated_examples'] == 1 and math_report['generation_budget'] == 2
        assert math_report['results'][0]['id'] == 'fixture-math'
    assert [knowledge_evaluation.load_checkpoint, math_evaluation.load_checkpoint] == originals


def test_unchanged_python_execution_and_real_image_ocr_scoring_use_the_code_export(completed_subject, tmp_path):
    subject = completed_subject
    view = tmp_path / 'code-view'
    bridge.prepare_view(subject, view, 'code')
    data = tmp_path / 'public-code'
    data.mkdir()
    questions = data / 'questions.jsonl'
    questions.write_text(json.dumps(dict(id='fixture-code', task_id=1, signature='def identity(value):',
        specification='Return value.', function='identity', cases=[{'args': [1], 'expected': 1}, {'args': [2], 'expected': 2}])) + '\n')
    (data / 'manifest.json').write_text(json.dumps(dict(questions_sha256=file_hash(questions),
        official_sanitized_test_problems=1, excluded_problems=0, limitations=['Fixture, not actual MBPP.'])))
    output = tmp_path / 'public-code.json'
    bridge.evaluate_bound(subject, view, public_code_benchmark, output, tmp_path / 'code-loader.json',
        arguments=dict(data=str(questions), max_new_tokens=2))
    report = json.loads(output.read_text())
    assert report['supported_problems'] == 1 and report['generation_budget'] == 2
    assert len(report['results']) == 1 and 'execution' in report['results'][0]
    assert report['results'][0]['execution']['total_cases'] == 2
    assert 'worker_exit_code' not in report['results'][0]['execution']
    reference = public_code_benchmark.execute(dict(language='Python', code='def identity(value):\n    return value\n',
        function='identity', cases=[{'args': [1], 'expected': 1}, {'args': [2], 'expected': 2}], stress=[]))
    assert reference['functional_passed'] and reference['passed_cases'] == 2
    fixture = Path(subject['base_checkpoint']).parent
    ocr = tmp_path / 'ocr.jsonl'
    rows = [dict(id=f'ocr-{i}', group_id=f'photo-{i}', task='word_ocr', prompt='Read the text.',
        response=value, image=str(fixture / image)) for i, (value, image) in enumerate([('red', 'image.png'), ('blue', 'other.png')])]
    ocr.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    bridge.evaluate_bound(subject, view, ocr_evaluation, tmp_path / 'ocr-report.json', tmp_path / 'ocr-loader.json',
        arguments=dict(data=str(ocr), max_examples=2, max_new_tokens=2, image_size=16))
    result = json.loads((tmp_path / 'ocr-report.json').read_text())
    assert result['image_size'] == 16 and result['overall']['original_images'] == 2 and result['truncated_sequences'] == 0
    for row in result['results']:
        score = ocr_evaluation.transcription_scores(row['expected'], row['generated'])
        assert all(row[k] == v for k, v in score.items())


def test_loader_rejects_training_wrong_device_paths_and_replaced_view_before_loading(completed_subject, tmp_path):
    subject = completed_subject
    checkpoint = bridge.prepare_view(subject, tmp_path / 'view', 'code')
    loader = bridge.BoundLoader(subject, checkpoint.parent, 'cpu')
    for path, device, training in [(checkpoint, 'cpu', True), (checkpoint, 'cuda:0', False),
            (Path(subject['inference_checkpoint']), 'cpu', False)]:
        with pytest.raises(ValueError, match='exact bound view/device'):
            loader(path, device=device, training_state=training)
    checkpoint.unlink()
    checkpoint.write_bytes(b'changed-view')
    with pytest.raises(ValueError, match='Bound runtime/data changed'):
        loader(checkpoint, training_state=False)
    assert file_hash(Path(subject['inference_checkpoint'])) == subject['inference_checkpoint_sha256']


def test_original_loader_is_restored_after_exception_and_nested_replacement_rejects(completed_subject, tmp_path):
    view = tmp_path / 'view'
    bridge.prepare_view(completed_subject, view, 'code')
    loader = bridge.BoundLoader(completed_subject, view, 'cpu')
    original = knowledge_evaluation.load_checkpoint
    with pytest.raises(RuntimeError, match='fixture generation stopped'):
        with bridge.evaluator_loader(knowledge_evaluation, loader):
            assert knowledge_evaluation.load_checkpoint is loader
            with pytest.raises(ValueError, match='Only one process-local'):
                with bridge.evaluator_loader(math_evaluation, loader):
                    pytest.fail('Nested replacement acquired a second global evaluator lease')
            raise RuntimeError('fixture generation stopped')
    assert knowledge_evaluation.load_checkpoint is original


def test_existing_results_are_preserved_before_any_new_evaluator_load(completed_subject, tmp_path, monkeypatch):
    output = tmp_path / 'existing.json'
    output.write_text('preserve')
    monkeypatch.setattr(bridge, 'BoundLoader', lambda *a, **k: pytest.fail('Existing output reached model loading'))
    with pytest.raises(ValueError, match='New distinct evaluation output'):
        bridge.evaluate_bound(completed_subject, tmp_path / 'unused', knowledge_evaluation, output, tmp_path / 'receipt.json')
    assert output.read_text() == 'preserve'
