import pickle
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_files", nargs="+", help="Evaluation result_r05.pkl files")
    args = parser.parse_args()

    for p in args.result_files:
        with open(p, "rb") as f:
            res = pickle.load(f)

        eer = res["eer"]
        arg = np.argmin(np.abs(res["precisions"] - eer))
        print(p, res["thresholds"][arg], res["precisions"][arg], res["recalls"][arg])


if __name__ == "__main__":
    main()
