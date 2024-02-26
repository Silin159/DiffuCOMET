import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.manifold import TSNE
from diffu_eval.parse_toolkits import tag_lemmatize, extract_patterns

ent_pts = {}

o_list = ["<oeffect>", "<oreact>", "<owant>"]

total_list = ["<atlocation>", "<capableof>", "<causes>", "<desires>", "<hasproperty>",
              "<hassubevent>", "<hinderedby>", "<isafter>", "<isbefore>", "<madeupof>",
              "<notdesires>", "<objectuse>", "<oeffect>", "<oreact>", "<owant>", "<xattr>",
              "<xeffect>", "<xintent>", "<xneed>", "<xreact>", "<xreason>", "<xwant>"]
special_tokens_webnlg = ["<fact_sep>", "<eos_fact>", "<rel_bos>", "<rel_eos>"]

relation_to_natural = {"<atlocation>": "$H is located or found at $T",
                       "<capableof>": "$H is capable of $T",
                       "<causes>": "$H causes $T",
                       "<desires>": "$H desires $T",
                       "<hasproperty>": "$H can be characterized by having or being $T",
                       "<hassubevent>": "$H includes that $T",
                       "<hinderedby>": "$H can be hindered by $T",
                       "<isafter>": "$H happens after $T",
                       "<isbefore>": "$H happens before $T",
                       "<madeupof>": "$H is made up of $T",
                       "<notdesires>": "$H does not desire $T",
                       "<objectuse>": "$H is used for $T",
                       "<oeffect>": "$H, as a result, persony or others will $T",
                       "<oreact>": "$H, as a result, persony or others feels $T",
                       "<owant>": "$H, as a result, persony or others wants $T",
                       "<xattr>": "$H, so he is seen as $T",
                       "<xeffect>": "$H, as a result, he will $T",
                       "<xintent>": "$H because he wanted $T",
                       "<xneed>": "$H, but before, he needed $T",
                       "<xreact>": "$H, as a result, he feels $T",
                       "<xreason>": "$H because $T",
                       "<xwant>": "$H, as a result, he wants $T"}


def extract_new_patterns(entity):
    entity_tokens = entity.split(" ")
    entity_lemma = tag_lemmatize(entity_tokens)
    patterns = extract_patterns(entity_lemma, filt_general="atomic")
    return patterns


def process(ent):
    if ent != "none" and "___" not in ent and "———" not in ent:
        ent = ent.lower()
        if ent.startswith("personx ") or ent.startswith("personx, "):
            ent = " ".join(ent.split(" ")[1:])
        if ent.startswith("person x "):
            ent = " ".join(ent.split(" ")[2:])

        ent = ent.replace("personx's ", "his ")
        ent = ent.replace("person x's ", "his ")
        ent = ent.replace("personx/s ", "his ")
        ent = ent.replace("personx\"s ", "his ")
        ent = ent.replace("personx;s ", "his ")
        ent = ent.replace("personx' ", "his ")
        ent = ent.replace("personxs ", "his ")
        ent = ent.replace("personxs' ", "his ")

        ent = ent.replace(" personx", " him")
        ent = ent.replace(" person x", " him")

        ent = ent.replace("´", "'")
        ent = ent.replace("ê", "e")
        ent = ent.replace("ø", "o")
        ent = ent.replace("ü", "u")
        ent = ent.replace("é", "e")
        ent = ent.replace("á", "a")
        ent = ent.replace("î", "i")
        ent = ent.replace("â", "a")
        ent = ent.replace("ñ", "n")
        ent = ent.strip("\u200e")

        if "personx" not in ent and "person x" not in ent:
            return ent
        else:
            return ""
    else:
        return ""


def filter_patterns(pts):
    new_pts = []
    for pt in pts:
        if "personx" not in pt and "person x" not in pt:
            new_pts.append(pt)
    return new_pts


def edit_distance(sl1, sl2):
    m, n = len(sl1), len(sl2)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif sl1[i-1] == sl2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n] / max(m, n)


def check_distance(part1, part2):
    list_dis = edit_distance(part1, part2)
    str_dis = edit_distance(" ".join(part1), " ".join(part2))
    return list_dis, str_dis


def extract_rel(rel_str):
    rel_str_clean = "<" + rel_str.split("<")[1].split(">")[0] + ">"
    if rel_str_clean in total_list:
        return rel_str_clean
    else:
        return "<fact_sep>"


def get_fact_from_triple(triple):
    new_head = process(triple[0])
    rel = "<" + triple[1].lower() + ">"
    new_tail = process(triple[2])
    if new_head and new_tail:
        return " ".join([new_head, rel, new_tail])
    else:
        return ""


def get_fact_from_triple_webnlg(triple):
    if triple[0] and triple[1] and triple[2]:
        return triple[0] + " <rel_bos> " + triple[1].lower() + " <rel_eos> " + triple[2]
    else:
        return ""


def get_triple_from_fact(fact):
    fact = fact.replace("<unk>", "#unk#")
    head = fact.split(" <")[0]
    rel = "<" + fact.split("<")[1].split(">")[0] + ">"
    tail = fact.rsplit("> ")[-1]
    head = head.replace("#unk#", "<unk>")
    tail = tail.replace("#unk#", "<unk>")
    return [head, rel, tail]


def get_triple_from_fact_webnlg(fact):
    head = fact.split(" <rel_bos> ")[0]
    rel = fact.split(" <rel_bos> ")[-1].split(" <rel_eos> ")[0]
    tail = fact.split(" <rel_eos> ")[-1]
    return [head, rel, tail]


def check_valid(fact):
    for rel in total_list:
        if rel in fact:
            head_tail = fact.split(rel)
            if len(head_tail) == 2 and head_tail[0].strip() and head_tail[1].strip():
                return True
            else:
                return False
    return False


def check_valid_webnlg(fact_str):
    if " <rel_bos> " in fact_str and " <rel_eos> " in fact_str:
        head = fact_str.split(" <rel_bos> ")[0]
        rel = fact_str.split(" <rel_bos> ")[-1].split(" <rel_eos> ")[0]
        tail = fact_str.split(" <rel_eos> ")[-1]
        if all(x for x in [head, rel, tail]) and all(y not in " ".join([head, rel, tail]) for y in special_tokens_webnlg):
            return True
    return False


def get_pts(fact_str):
    head_str = fact_str.split(" <")[0]
    rel_str = "<" + fact_str.split("<")[1].split(">")[0] + ">"
    rel_pt = "<o>" if rel_str in o_list else "<x>"
    tail_str = fact_str.rsplit("> ")[-1]

    if head_str not in ent_pts:
        head_pts = extract_new_patterns(head_str)
        ent_pts[head_str] = head_pts
    else:
        head_pts = ent_pts[head_str]

    if tail_str not in ent_pts:
        tail_pts = extract_new_patterns(tail_str)
        ent_pts[tail_str] = tail_pts
    else:
        tail_pts = ent_pts[tail_str]

    return head_pts + [rel_pt] + tail_pts


def get_pts_webnlg(fact_str):
    head_str = fact_str.split(" <rel_bos> ")[0]
    rel_str = fact_str.split(" <rel_bos> ")[-1].split(" <rel_eos> ")[0]
    rel_pt = rel_str
    tail_str = fact_str.split(" <rel_eos> ")[-1]

    if head_str not in ent_pts:
        head_pts = extract_new_patterns(head_str)
        ent_pts[head_str] = head_pts
    else:
        head_pts = ent_pts[head_str]

    if tail_str not in ent_pts:
        tail_pts = extract_new_patterns(tail_str)
        ent_pts[tail_str] = tail_pts
    else:
        tail_pts = ent_pts[tail_str]

    return head_pts + [rel_pt] + tail_pts


def get_mean_distances(add_fact, groups):
    distances = []
    for group in groups:
        g_dis = []
        for g_fact in group:
            w_dis, c_dis = check_distance(get_pts(add_fact), get_pts(g_fact))
            g_dis.append(min(w_dis, c_dis))
        distances.append(sum(g_dis) / len(g_dis))
    return distances


def get_mean_distances_webnlg(add_fact, groups):
    distances = []
    for group in groups:
        g_dis = []
        for g_fact in group:
            w_dis, c_dis = check_distance(get_pts_webnlg(add_fact), get_pts_webnlg(g_fact))
            g_dis.append(min(w_dis, c_dis))
        distances.append(sum(g_dis) / len(g_dis))
    return distances


def filter_special(fact_str):
    for token in ["<s>", "</s>", "<pad>"]:
        fact_str = fact_str.replace(token, "").strip()
    return fact_str


def group_facts(fact_list, min_edit=0.45):
    fact_groups = []
    for fact_str in fact_list:
        if fact_str != "<eos_fact>" and any([rel in fact_str for rel in total_list]):
            fact_str = filter_special(fact_str)
            mean_dis = get_mean_distances(fact_str, fact_groups)
            if mean_dis and min(mean_dis) < min_edit:
                fact_groups[mean_dis.index(min(mean_dis))].append(fact_str)
            else:
                fact_groups.append([fact_str])
    return fact_groups


def group_facts_webnlg(fact_list, min_edit=0.45):
    fact_groups = []
    for fact_str in fact_list:
        if fact_str != "<eos_fact>" and any([rel in fact_str for rel in total_list]):
            fact_str = filter_special(fact_str)
            mean_dis = get_mean_distances_webnlg(fact_str, fact_groups)
            if mean_dis and min(mean_dis) < min_edit:
                fact_groups[mean_dis.index(min(mean_dis))].append(fact_str)
            else:
                fact_groups.append([fact_str])
    return fact_groups


def group_facts_dbscn_edit(fact_list, fact_pts, eps=0.45, min_samples=2,
                           min_cluster_size=15, max_cluster_size=None, hierarchical=False):

    def edit_distance_metric(x, y):
        i, j = int(x[0]), int(y[0])
        return edit_distance(fact_pts[i], fact_pts[j])

    fact_ids = np.arange(len(fact_list)).reshape(-1, 1)
    if hierarchical:
        cluster = HDBSCAN(metric=edit_distance_metric, min_cluster_size=min_cluster_size, min_samples=min_samples,
                          max_cluster_size=max_cluster_size, algorithm='brute')
    else:
        cluster = DBSCAN(metric=edit_distance_metric, eps=eps, min_samples=min_samples, algorithm='brute')
    cluster.fit(fact_ids)
    labels = cluster.labels_.tolist()

    grouping = {}
    for fid, label in enumerate(labels):
        if label not in grouping:
            grouping[label] = []
        grouping[label].append(fact_list[fid])

    fact_groups = []
    for gid, group in grouping.items():
        if gid != -1:
            fact_groups.append(group)
        else:
            for fact in group:
                fact_groups.append([fact])

    return fact_groups, labels


def group_facts_dbscn_embed(fact_list, fact_embeds, eps=0.45, min_samples=2, min_cluster_size=15,
                            max_cluster_size=None, hierarchical=False, tsne=False):

    fact_embeds_np = np.array(fact_embeds)
    if tsne:
        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        fact_embeds_np = tsne.fit_transform(fact_embeds_np)

    if hierarchical:
        cluster = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                          max_cluster_size=max_cluster_size)
    else:
        cluster = DBSCAN(eps=eps, min_samples=min_samples)
    cluster.fit(fact_embeds_np)
    labels = cluster.labels_.tolist()

    grouping = {}
    for fid, label in enumerate(labels):
        if label not in grouping:
            grouping[label] = []
        grouping[label].append(fact_list[fid])

    fact_groups = []
    for gid, group in grouping.items():
        if gid != -1:
            fact_groups.append(group)
        else:
            for fact in group:
                fact_groups.append([fact])

    return fact_groups, labels
