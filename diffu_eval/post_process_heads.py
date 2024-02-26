import json
import argparse
from diffu_eval.extract_tools import filter_special
from tqdm import tqdm


def main(args):

    with open(args.context, "r") as f:
        test_contexts = f.readlines()
    with open(args.result_dir+"/generations.json", "r") as f:
        gen_results = json.load(f)

    processed_results = []
    tail_gen_input_contexts = []
    tail_gen_input_labels = []
    tail_gen_input_ids = []

    for nid, narrative in tqdm(enumerate(test_contexts)):

        result_single = {"context": narrative.strip(), "generations": [], "golds": []}

        for head in gen_results[nid]["gen"]:
            head = filter_special(head)
            if head == "<eos_fact>":
                break
            elif head and "<eos_fact>" not in head:
                result_single["generations"].append(head)

        for head in gen_results[nid]["ground"]:
            head = filter_special(head)
            if head != "<eos_fact>":
                result_single["golds"].append(head)
            else:
                break

        result_single["generations"] = list(set(result_single["generations"]))
        processed_results.append(result_single)

        for head in result_single["generations"]:
            tail_gen_input_contexts.append(" ".join([result_single["context"], "<fact_sep>", head]))
            tail_gen_input_labels.append([])  # no gold labels for pipeline generation
            tail_gen_input_ids.append(str(nid)+"_"+head)

    with open(args.tail_gen_input_dir+".contexts", "w") as f:
        for line in tail_gen_input_contexts:
            f.write(line + "\n")

    with open(args.tail_gen_input_dir+".tails.json", "w") as f:
        json.dump(tail_gen_input_labels, f, indent=2)

    with open(args.tail_gen_input_dir+".ids", "w") as f:
        for line in tail_gen_input_ids:
            f.write(line + "\n")

    with open(args.result_dir+"/gen_processed.json", "w") as f:
        json.dump(processed_results, f, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments")

    parser.add_argument("--context", type=str)
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--tail_gen_input_dir", type=str)

    args = parser.parse_args()
    main(args)
