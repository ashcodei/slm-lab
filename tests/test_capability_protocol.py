from slm_lab.alignment_selection import REPORT_FILES
from slm_lab.capability_protocol import specifications
from slm_lab import code_quality


def test_handoff_schedules_every_required_artifact_including_both_code_modes(tmp_path):
    records = specifications(tmp_path)
    assert len(records) == 9
    assert {filename for _, _, filename, _ in records} == set(REPORT_FILES.values())
    assert len({filename for _, _, filename, _ in records}) == len(records)
    code = {family: (module, arguments) for family, module, _, arguments in records if family.startswith('code')}
    assert code['code'] == ('slm_lab.code_quality', ['--prompt-mode', 'completion'])
    assert code['code_instruction'] == ('slm_lab.code_quality', ['--prompt-mode', 'instruction'])


def test_registry_code_commands_emit_each_requested_artifact_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(code_quality, 'evaluate', lambda checkpoint, output, device, budget, prompt_mode:
                        calls.append((output, prompt_mode)))
    expected = []
    for family, module, filename, arguments in specifications(tmp_path):
        if family not in {'code', 'code_instruction'}: continue
        output = str(tmp_path / filename)
        monkeypatch.setattr('sys.argv', [module, '--checkpoint', 'own-model.pt', '--output', output,
                                        '--device', 'cpu', *arguments])
        code_quality.main()
        expected.append((output, 'completion' if family == 'code' else 'instruction'))
    assert calls == expected
