import json
import argparse


def main(args):

    with open(args.comfact_output, "r") as f:
        comfact_outputs = json.load(f)

    with open(args.generation, "r") as f:
        generations = json.load(f)

    preds = {}
    for pred in comfact_outputs:
        idx_info = "|".join([str(pred["context_id"]), str(pred["fact_id"])])
        relevance = pred["score"][1]
        preds[idx_info] = relevance

    for cid, gen in enumerate(generations):

        gen_eval = {}
        for fid, fact in enumerate(gen["generations"]):
            idx = "|".join(["gen", str(cid), str(fid)])
            gen_eval[fact] = preds[idx]
            generations[cid]["generations"] = gen_eval

        gen_eval_gold = {}
        for fid, fact in enumerate(gen["golds"]):
            idx = "|".join(["gold", str(cid), str(fid)])
            gen_eval_gold[fact] = preds[idx]
            generations[cid]["golds"] = gen_eval_gold

    with open(args.generation, "w") as f:
        json.dump(generations, f, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments")

    parser.add_argument("--comfact_output", type=str)
    parser.add_argument("--generation", type=str)

    args = parser.parse_args()
    main(args)
