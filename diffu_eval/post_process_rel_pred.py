import json
import argparse
from diffu_eval.extract_tools import extract_rel
from tqdm import tqdm


def main(args):

    with open(args.rel_pred_ids, "r") as f:
        rel_pred_ids = json.load(f)
    with open(args.rel_pred_results, "r") as f:
        rel_pred_results = json.load(f)

    pred_rel_map = {}
    for pid, pred in zip(rel_pred_ids, rel_pred_results):
        cid = pid.split("|")[0]
        head = pid.split("|")[1]
        if args.test_data == "webnlg":
            rel = pred["text"][0]
        else:
            rel = extract_rel(pred["text"][0])
        tail = pid.split("|")[2]
        if cid not in pred_rel_map:
            pred_rel_map[cid] = []
        if args.test_data == "webnlg":
            pred_rel_map[cid].append(head+" <rel_bos> "+rel+" <rel_eos> "+tail)
        else:
            pred_rel_map[cid].append(" ".join([head, rel, tail]))

    with open(args.processed_dir, "r") as f:
        processed_results = json.load(f)

    for cid, _ in tqdm(enumerate(processed_results)):
        processed_results[cid]["generations"] = list(set(pred_rel_map[str(cid)]))

    with open(args.pipeline_result_dir+"/gen_processed.json", "w") as f:
        json.dump(processed_results, f, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments")

    parser.add_argument("--test_data", type=str)
    parser.add_argument("--rel_pred_ids", type=str)
    parser.add_argument("--rel_pred_results", type=str)
    parser.add_argument("--pipeline_result_dir", type=str)

    args = parser.parse_args()
    main(args)
