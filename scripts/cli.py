import argparse, sys
from scripts import etl

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["import"], help="import raw → parquet")
    args = parser.parse_args()
    if args.cmd == "import":
        etl.run_etl()

if __name__ == "__main__":
    main() 