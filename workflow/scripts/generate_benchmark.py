import sys
import os
import argparse

import EMB


def gather_args(args):
    parser = argparse.ArgumentParser(
        prog='LFR binary - Python wrapper',
        description='Synthesize an LFR benchmark.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='Have a great day!... or else.'
    )
    parser.add_argument()

    return args


def main():
    D, _, _, _ = EMB.netsci.models.benchmarks.generate_duplex_LFR(
            int(N),
            2.1, 1.0, 0.1, 6, 100, 1,
            ROOT=os.path.join("..", "..", "")
        )
        D = dict(enumerate(EMB.netsci.models.preprocessing.make_layers_disjoint(*D.values())))
    

    

    return


if __name__ == "__main__":
    main()
