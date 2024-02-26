import json
import argparse
from extract_tools import filter_special
from tqdm import tqdm
import os


def main(args):

    with open(args.gold_dir+".contexts", "r") as f:
        gold_contexts = f.readlines()
    with open(args.gold_dir+".facts.json", "r") as f:
        gold_facts = json.load(f)

    processed_results = []
    for cid, ctx in enumerate(gold_contexts):
        processed_results.append({"context": ctx.strip(), "golds": gold_facts[cid], "gen_no_rel": []})

    with open(args.tail_gen_input_dir+".contexts", "r") as f:
        tail_gen_contexts = f.readlines()
    with open(args.tail_gen_input_dir+".ids", "r") as f:
        tail_gen_ids = f.readlines()
    with open(args.tail_gen_result_dir+"/generations.json", "r") as f:
        tail_gen_results = json.load(f)

    rel_pred_inputs = []
    rel_pred_ids = []
    shift = 0
    skip_cid = -1

    for pid, prediction in tqdm(enumerate(tail_gen_results)):
        tail_gen_ctx = tail_gen_contexts[pid].split(" <fact_sep> ")[0].strip()
        tail_gen_info = tail_gen_ids[pid].strip()
        cid = int(tail_gen_info.split("_")[0])
        head = tail_gen_info.split("_")[1]

        if cid != skip_cid:

            if (cid-shift) < len(processed_results) and processed_results[cid-shift]["context"] == tail_gen_ctx:
                for tail in prediction["gen"]:
                    tail = filter_special(tail)
                    if tail == "<eos_fact>":
                        break
                    elif tail and "<eos_fact>" not in tail:
                        gen_fact = " ".join([head, "<fact_sep>", tail])
                        processed_results[cid-shift]["gen_no_rel"].append(gen_fact)
                        rel_pred_inputs.append(" <fact_sep> ".join([tail_gen_ctx, head, tail]))
                        rel_pred_ids.append("|".join([str(cid-shift), head, tail]))
            else:
                shift += 1
                skip_cid = cid

    for cid, output in enumerate(processed_results):
        processed_results[cid]["gen_no_rel"] = list(set(output["gen_no_rel"]))

    os.makedirs(args.pipeline_result_dir, exist_ok=True)

    with open(args.pipeline_result_dir+"/gen_processed.json", "w") as f:
        json.dump(processed_results, f, indent=2)

    os.makedirs(args.rel_pred_input_dir, exist_ok=True)

    with open(args.rel_pred_input_dir+"/logs.json", "w") as f:
        json.dump(rel_pred_inputs, f, indent=2)
    with open(args.rel_pred_input_dir+"/labels.json", "w") as f:
        json.dump(rel_pred_ids, f, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments")

    parser.add_argument("--gold_dir", type=str)
    parser.add_argument("--tail_gen_input_dir", type=str)
    parser.add_argument("--tail_gen_result_dir", type=str)
    parser.add_argument("--pipeline_result_dir", type=str)
    parser.add_argument("--rel_pred_input_dir", type=str)

    args = parser.parse_args()
    main(args)
