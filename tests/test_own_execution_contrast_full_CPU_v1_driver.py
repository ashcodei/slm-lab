import importlib.util
from pathlib import Path


def driver():
    path = Path(__file__).resolve().parents[1] / 'scripts/train_own_execution_contrast_full_CPU_v1.py'
    spec = importlib.util.spec_from_file_location('own_contrast_driver_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_every_original_pair_once_in_each_fixed_pass_and_missing_process_not_live():
    module = driver(); order = module.pair_order()
    assert order == module.pair_order() and len(order) == 40
    assert sorted(order[:20]) == sorted(order[20:]) == list(range(20))
    assert order[:20] != order[20:]
    assert module.same(dict(pid=999999999, ticks='0')) is None
