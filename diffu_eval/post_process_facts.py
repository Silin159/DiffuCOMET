import json
import argparse
from diffu_eval.extract_tools import filter_special
from tqdm import tqdm


def main(args):

    with open(args.context_dir, "r") as f:
        test_contexts = f.readlines()
    with open(args.result_dir+"/generations.json", "r") as f:
        gen_results = json.load(f)

    processed_results = []

    for nid, narrative in tqdm(enumerate(test_contexts)):

        result_single = {"context": narrative.strip(), "generations": [], "golds": []}

        for fact in gen_results[nid]["gen"]:
            fact = filter_special(fact)
            if fact == "<eos_fact>":
                break
            elif fact and "<eos_fact>" not in fact:
                result_single["generations"].append(fact)

        for fact in gen_results[nid]["ground"]:
            fact = filter_special(fact)
            if fact != "<eos_fact>":
                result_single["golds"].append(fact)
            else:
                break

        result_single["generations"] = list(set(result_single["generations"]))
        processed_results.append(result_single)

    with open(args.result_dir+"/gen_processed.json", "w") as f:
        json.dump(processed_results, f, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments")

    parser.add_argument("--context_dir", type=str)
    parser.add_argument("--result_dir", type=str)

    args = parser.parse_args()
    main(args)
