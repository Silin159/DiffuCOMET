import json
from tqdm import tqdm
from extract_tools import get_triple_from_fact, get_triple_from_fact_webnlg
from copy import deepcopy
import argparse
import os

relation_to_natural = {"<atlocation>": "located or found at/in/on",
                       "<capableof>": "is/are capable of",
                       "<causes>": "causes",
                       "<desires>": "desires",
                       "<hasproperty>": "can be characterized by being/having",
                       "<hassubevent>": "includes the event/action",
                       "<hinderedby>": "can be hindered by",
                       "<isafter>": "happens after",
                       "<isbefore>": "happens before",
                       "<madeupof>": "made (up) of",
                       "<notdesires>": "do(es) not desire",
                       "<objectuse>": "used for",
                       "<oeffect>": "as a result, persony or others will",
                       "<oreact>": "as a result, persony or others feels",
                       "<owant>": "as a result, persony or others wants",
                       "<xattr>": "he/she is seen as",
                       "<xeffect>": "as a result, he/she will",
                       "<xintent>": "because he/she wants",
                       "<xneed>": "but before, he/she needs",
                       "<xreact>": "as a result, he/she feels",
                       "<xreason>": "because",
                       "<xwant>": "as a result, he/she wants"}


def extract_context(context_str):
    context_ori = deepcopy(context_str)
    for delimiter in ["<past>", "<center>", "<future>"]:
        context_str = context_str.replace(delimiter, "|")
    context_split = [x.strip() for x in context_str.split("|") if x]

    if "<past>" in context_ori and "<future>" in context_ori:
        p_context = context_split[0].split(" <utter_sep> ")
        center = context_split[1].split(" <utter_sep> ")
        f_context = context_split[2].split(" <utter_sep> ")
    elif "<past>" in context_ori:
        p_context = context_split[0].split(" <utter_sep> ")
        center = context_split[1].split(" <utter_sep> ")
        f_context = []
    elif "<future>" in context_ori:
        p_context = []
        center = context_split[0].split(" <utter_sep> ")
        f_context = context_split[1].split(" <utter_sep> ")
    else:
        p_context = []
        center = context_split[0].split(" <utter_sep> ")
        f_context = []

    return {"p_context": p_context, "center": center, "f_context": f_context}


def main(args):

    logs = []
    labels = []

    with open(args.generation, "r") as f:
        generations = json.load(f)

    for gid, gen in tqdm(enumerate(generations)):

        context = extract_context(gen["context"])
        if args.test_data == "webnlg":
            gen_facts = [get_triple_from_fact_webnlg(x) for x in gen["generations"]]
            gold_facts = [get_triple_from_fact_webnlg(x) for x in gen["golds"]]
        else:
            gen_facts = [get_triple_from_fact(x) for x in gen["generations"]]
            gold_facts = [get_triple_from_fact(x) for x in gen["golds"]]
        facts = {"gen": gen_facts, "gold": gold_facts}

        for portion in ["gen", "gold"]:
            cid = "|".join([portion, str(gid)])
            data_point = {"cid": cid, "tid": -1, "text": []}

            for utter in context["p_context"]:
                data_point["text"].append({"type": "p_context", "utter": utter.lower()})
            for utter in context["f_context"]:
                data_point["text"].append({"type": "f_context", "utter": utter.lower()})
            data_point["text"].append({"type": "center", "utter": context["center"][0].lower()})

            for fid, triple in enumerate(facts[portion]):
                data_point_single = deepcopy(data_point)
                data_point_single["fid"] = fid
                data_point_single["text"].append({"type": "fact", "utter": triple[0]})
                if args.test_data == "webnlg":
                    data_point_single["text"].append({"type": "fact", "utter": triple[1]})
                else:
                    data_point_single["text"].append({"type": "fact", "utter": relation_to_natural.get(triple[1], ", ")})
                data_point_single["text"].append({"type": "fact", "utter": triple[2]})

                logs.append(data_point_single)
                labels.append({"target": True, "linking": None})  # fake labels

    os.makedirs(args.comfact_input_dir, exist_ok=True)
    with open(args.comfact_input_dir+"/logs.json", "w") as f:
        json.dump(logs, f, indent=2)
    with open(args.comfact_input_dir+"/labels.json", "w") as f:
        json.dump(labels, f, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments")

    parser.add_argument("--test_data", type=str)
    parser.add_argument("--generation", type=str)
    parser.add_argument("--comfact_input_dir", type=str)

    args = parser.parse_args()
    main(args)
