import argparse

parser = argparse.ArgumentParser(description="an example program")

parser.add_argument(
    "--flist", required=True, nargs="*", type=float, help="a list of float variables"
)
args = parser.parse_args()

print(args.flist)
