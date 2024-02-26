import argparse
import json
import os
from tqdm import tqdm
from diffu_eval.metrics import BLEU, METEOR, ROUGE, KggNGramDiversity

metrics = [BLEU(), METEOR(), ROUGE()]
diverse_gold = [KggNGramDiversity(1), KggNGramDiversity(2), KggNGramDiversity(3), KggNGramDiversity(4)]
diverse_gen = [KggNGramDiversity(1), KggNGramDiversity(2), KggNGramDiversity(3), KggNGramDiversity(4)]


def main(args):

    with open(args.generation, "r") as f:
        outputs = json.load(f)

    result = {"gold_count": 0, "gen_count": 0}

    for metric_set in [metrics, diverse_gold, diverse_gen]:
        for metric in metric_set:
            metric.reset()

    for sample in tqdm(outputs):

        if isinstance(sample["golds"], dict):
            gold_facts = list(sample["golds"].keys())
        else:
            gold_facts = sample["golds"]

        if isinstance(sample["generations"], dict):
            gen_facts = list(sample["generations"].keys())
        else:
            gen_facts = sample["generations"]

        result["gold_count"] += len(gold_facts)
        result["gen_count"] += len(gen_facts)

        for metric in diverse_gold:
            for gold in gold_facts:
                metric.update((gold, None))
            metric.sub_compute_reset()

        for metric in diverse_gen:
            for gen in gen_facts:
                metric.update((gen, None))
            metric.sub_compute_reset()

        for gen in gen_facts:
            for metric in metrics:
                metric.update((gen, gold_facts))

    for key in ["gold_count", "gen_count"]:
        result[key] = result[key] / len(outputs)
        print("%s = %s\n" % (key, str(result[key])))

    for metric in metrics:
        name = metric.name()
        score = metric.compute()
        result[name] = score
        print("%s = %s\n" % (name, str(score)))

    for m_type, metric_set in {"Gold": diverse_gold, "Gen": diverse_gen}.items():
        for metric in metric_set:
            name = metric.name() + "-" + m_type
            score_count, score_ratio = metric.compute()
            result[name] = (score_count, score_ratio)
            print("%s = %s\n" % (name, str(score_count) + "|" + str(score_ratio)))

    os.makedirs(args.eval_result_dir, exist_ok=True)

    with open(args.eval_result_dir+"/nlg_eval.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments")

    parser.add_argument("--generation", type=str)
    parser.add_argument("--eval_result_dir", type=str)

    args = parser.parse_args()
    main(args)
