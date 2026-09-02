"""Fast, non-writing smoke tests for the three paper applications."""

import argparse

from Simulator.runners.main_agg import main as run_aggregation
from Simulator.runners.main_drcc import main as run_drcc
from Simulator.runners.main_ds import main as run_distribution


RUNNERS = {
    'aggregation': lambda: run_aggregation(['--smoke', '--device', 'cpu']),
    'td': lambda: run_distribution(['--smoke', '--device', 'cpu']),
    'drcc': lambda: run_drcc(['--smoke', '--device', 'cpu']),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--case', choices=(*RUNNERS, 'all'), default='all',
        help='Application smoke test to run.',
    )
    args = parser.parse_args(argv)
    selected = list(RUNNERS) if args.case == 'all' else [args.case]
    for name in selected:
        print(f'=== smoke: {name} ===')
        RUNNERS[name]()
    print(f"SMOKE_TESTS_OK cases={selected}")


if __name__ == '__main__':
    main()
