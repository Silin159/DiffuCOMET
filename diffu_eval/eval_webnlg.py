import json
from diffu_eval.misc.utils import compute_scores_simple
from extract_tools import get_triple_from_fact_webnlg
from tqdm import tqdm
import argparse


def main(args):

    with open(args.generation, "r") as f:
        gen_results = json.load(f)
    hyp = []
    ref = []
    for dp in tqdm(gen_results):
        hyp.append([get_triple_from_fact_webnlg(x) for x in list(dp["generations"].keys())])
        ref.append([get_triple_from_fact_webnlg(x) for x in list(dp["golds"].keys())])
    compute_scores_simple(hyp, ref, args.eval_result_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments")

    parser.add_argument("--generation", type=str)
    parser.add_argument("--eval_result_dir", type=str)

    args = parser.parse_args()
    main(args)
