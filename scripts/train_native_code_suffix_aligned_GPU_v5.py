"""Select the recovered source recipe for the unchanged full indexed driver.

This parent adapter changes no neural worker, objective, source, budget or
assessment. Original version4 worker processes still execute the experiment.
"""
import argparse


def main():
    import train_native_code_suffix_aligned_GPU_v4 as driver
    from slm_lab import native_suffix_owned_runtime_contract_v4 as original
    from slm_lab.native_suffix_owned_runtime_contract_v5 import require_complete_preparation
    from slm_lab.metadata_observer_recovery_v1 import INDEXED_OUT
    previous = original.require_complete_preparation
    previous_preparation = driver.PREP
    original.require_complete_preparation = require_complete_preparation
    driver.PREP = INDEXED_OUT
    try:
        driver.configure()
        driver.parent()
    finally:
        original.require_complete_preparation = previous
        driver.PREP = previous_preparation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        parser.error('Use explicit --run only after complete recovered preparation and all original owned GPUgates')
    main()
