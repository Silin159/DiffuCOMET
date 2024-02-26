import json
from extract_tools import get_pts, get_pts_webnlg, get_triple_from_fact, get_triple_from_fact_webnlg
from extract_tools import relation_to_natural, edit_distance
from extract_tools import group_facts_dbscn_edit, group_facts_dbscn_embed
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv
import argparse

sentence_bert = SentenceTransformer('all-mpnet-base-v2')


def cluster_edit_similarity(cluster_1, cluster_2, test="comfact"):
    min_distance = None
    for fact_pair in itertools.product(cluster_1, cluster_2):
        if test == "webnlg":
            distance = edit_distance(get_pts_webnlg(fact_pair[0]), get_pts_webnlg(fact_pair[1]))
        else:
            distance = edit_distance(get_pts(fact_pair[0]), get_pts(fact_pair[1]))
        if min_distance is None or distance < min_distance:
            min_distance = distance
    return 1.0 - min_distance


def cluster_emb_similarity(cluster_1, cluster_2, test="comfact"):

    def process_as_natural(fact, test_data):
        if test_data == "webnlg":
            triple = get_triple_from_fact_webnlg(fact)
            return ", ".join(triple)
        else:
            triple = get_triple_from_fact(fact)
            if triple[1] in relation_to_natural:
                return relation_to_natural[triple[1]].replace("$H", triple[0]).replace("$T", triple[2])
            else:
                return triple[0]+", "+triple[2]

    cluster_1_natural = [process_as_natural(fa, test) for fa in cluster_1]
    cluster_2_natural = [process_as_natural(fa, test) for fa in cluster_2]
    cluster_1_embeds = sentence_bert.encode(cluster_1_natural, convert_to_numpy=True)
    cluster_2_embeds = sentence_bert.encode(cluster_2_natural, convert_to_numpy=True)
    similarity = cosine_similarity(cluster_1_embeds, cluster_2_embeds)
    return max(np.max(similarity), 0.0)


def cluster_distance_recall(gen_facts, gold_clusters, eval_type="edit", test="comfact"):
    if len(gold_clusters) == 0:
        return 1.0
    if len(gen_facts) == 0:
        return 0.0
    recall_dist_sum = 0.0
    for cluster in gold_clusters:
        assert len(cluster) > 0
        if eval_type == "edit":
            max_sim = cluster_edit_similarity(gen_facts, cluster, test)
        elif eval_type == "emb":
            max_sim = cluster_emb_similarity(gen_facts, cluster, test)
        else:
            raise ValueError
        recall_dist_sum += max_sim
    return recall_dist_sum / len(gold_clusters)


def cluster_comfact_precision(comfact_eval, gen_clusters, comfact_invalid):
    if len(gen_clusters) == 0:
        return 0.0
    precision_sum = 0.0
    for cluster in gen_clusters:
        assert len(cluster) > 0
        score = 0.0
        for fact in cluster:
            score += comfact_eval[fact]
        precision_sum += score / len(cluster)
    return precision_sum / (len(gen_clusters) + len(comfact_invalid))


def clustering(fact_list, threshold, cluster_type="edit", test="comfact"):
    if len(fact_list) == 0:
        return []
    if threshold == 0.0:
        return [[x] for x in fact_list]

    if cluster_type == "edit":
        fact_pts = []
        for fact_single in fact_list:
            if test == "webnlg":
                fact_pts.append(get_pts_webnlg(fact_single))
            else:
                fact_pts.append(get_pts(fact_single))
        fact_pts_group, _ = group_facts_dbscn_edit(fact_list, fact_pts, eps=threshold, min_samples=2)
        return fact_pts_group

    else:  # cluster_type == "emb"
        fact_emb = []
        for fact_single in fact_list:
            if test == "webnlg":
                triple = get_triple_from_fact_webnlg(fact_single)
                natural = ", ".join(triple)
            else:
                triple = get_triple_from_fact(fact_single)
                if triple[1] in relation_to_natural:
                    natural = relation_to_natural[triple[1]].replace("$H", triple[0]).replace("$T", triple[2])
                else:
                    natural = triple[0] + ", " + triple[2]
            s_bert_emb = sentence_bert.encode(natural, convert_to_numpy=True)
            s_bert_emb = s_bert_emb / np.linalg.norm(s_bert_emb, ord=2)
            fact_emb.append(s_bert_emb)
        fact_emb_group, _ = group_facts_dbscn_embed(fact_list, fact_emb, eps=threshold, min_samples=2)
        return fact_emb_group


def main(args):

    eps_range_mapping = {"comfact_roc": {"edit": list(range(3, 16)), "emb": list(range(5, 22))},
                         "comfact_persona": {"edit": list(range(3, 17)), "emb": list(range(6, 27))},
                         "comfact_mutual": {"edit": list(range(3, 17)), "emb": list(range(5, 23))},
                         "comfact_movie": {"edit": list(range(3, 17)), "emb": list(range(7, 27))},
                         "webnlg": {"edit": list(range(3, 17)), "emb": list(range(8, 24))}}

    g_type_map = {"gold": "golds", "gen": "generations"}

    eval_results = {"gold_cluster_count_edit": [],
                    "gold_cluster_count_emb": [],
                    "gold_cluster_precision_edit": [],
                    "gold_cluster_precision_emb": [],
                    "gen_cluster_count_edit": [],
                    "gen_cluster_count_emb": [],
                    "gen_cluster_precision_edit": [],
                    "gen_cluster_precision_emb": [],
                    "cluster_recall_edit": [],
                    "cluster_recall_emb": []}

    with open(args.generation, "r") as f:
        gen_results = json.load(f)

    for c_type in ["edit", "emb"]:

        eps_range = eps_range_mapping[args.test_data][c_type]

        for eps_idx in tqdm(eps_range):
            eps = eps_idx / 20.0

            scores = {"gold_cluster_count": 0.0,
                      "gold_cluster_precision": 0.0,
                      "gen_cluster_count": 0.0,
                      "gen_cluster_precision": 0.0,
                      "cluster_recall": 0.0}

            for sample in tqdm(gen_results):

                group = {"gold": [], "gen": []}
                for g_type in ["gold", "gen"]:

                    f_comfact = sample[g_type_map[g_type]]
                    f_list = list(f_comfact.keys())

                    group[g_type] = clustering(f_list, eps, cluster_type=c_type, test=args.test_data)

                    if g_type == "gen":
                        scores[g_type+"_cluster_count"] += len(group[g_type])
                        if sample.get("invalid_gens"):
                            scores[g_type+"_cluster_precision"] += \
                                cluster_comfact_precision(f_comfact, group[g_type], sample["invalid_gens"])
                        else:
                            scores[g_type + "_cluster_precision"] += \
                                cluster_comfact_precision(f_comfact, group[g_type], [])
                    else:
                        scores[g_type+"_cluster_count"] += len(group[g_type])
                        scores[g_type+"_cluster_precision"] += \
                            cluster_comfact_precision(f_comfact, group[g_type], [])

                scores["cluster_recall"] += cluster_distance_recall(sample["generations"], group["gold"],
                                                                    eval_type=c_type, test=args.test_data)

            for key, value in scores.items():
                eval_results[key+"_"+c_type].append(value / len(gen_results))
                print(key+": "+str(value / len(gen_results)))

    with open(args.eval_result_dir+"/cluster_eval.csv", "w") as f:
        writer = csv.writer(f)
        for key, value in eval_results.items():
            writer.writerow([key])
            writer.writerow(value)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments")

    parser.add_argument("--test_data", type=str)
    parser.add_argument("--generation", type=str)
    parser.add_argument("--eval_result_dir", type=str)

    args = parser.parse_args()
    main(args)
