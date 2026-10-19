"""Versioned delivery bootstrap that also pins the separately invoked CLI.

Keep the closed V1 controller/exporter bytes unchanged. Its subprocess uses a
module name, so explicitly include that query module in this bootstrap's source
closure. Dispatch future owned children through this same pinned bootstrap.
"""
import argparse
import importlib.util
from pathlib import Path


def source_dependencies():
    # Source-closure declarations only; never execute neural imports while
    # the exact original PPO owner is live.
    import deliver_native_policy_reward_CPU_v1
    from slm_lab.own_policy_reward_query_v1 import main
    return deliver_native_policy_reward_CPU_v1,main


spec=importlib.util.spec_from_file_location('preserved_delivery_controller_v1',
    Path(__file__).with_name('deliver_native_policy_reward_CPU_v1.py'))
implementation=importlib.util.module_from_spec(spec);spec.loader.exec_module(implementation)
implementation.__file__=__file__
parent=implementation.parent
worker=implementation.worker


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path('/tmp/slm-native-policy-reward-delivery-CPU-v2'))
    parser.add_argument('--PPO',type=Path,default=Path('/tmp/slm-native-parallel-PPO-owned-GPU-v2/reports'))
    parser.add_argument('--run',action='store_true');parser.add_argument('--export',action='store_true');args=parser.parse_args(argv)
    if args.run:return parent(args)
    if args.export:return worker(args.workspace)
    parser.error('Use explicit source-waiting --run or registered owned CPU --export mode')


if __name__=='__main__':main()
