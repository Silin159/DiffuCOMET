from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import json
import string

wnl = WordNetLemmatizer()
skip_lemma = ["as"]
idioms_file_path = "./diffu_eval/idioms_eng_dic.json"
with open(idioms_file_path, 'r') as f:
    idioms = json.load(f)


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def tag_lemmatize(tokens, include_tags=True):
    tokens_tagged = pos_tag(tokens)
    lemma = []
    for word_tagged in tokens_tagged:
        wordnet_pos = get_wordnet_pos(word_tagged[1])
        if word_tagged[0] in skip_lemma:
            lemma.append(word_tagged)
        elif wordnet_pos is not None:
            lemma.append([wnl.lemmatize(word_tagged[0], pos=wordnet_pos), word_tagged[1]])
        else:
            lemma.append([wnl.lemmatize(word_tagged[0], pos=wordnet.NOUN), word_tagged[1]])
    if include_tags:
        return lemma
    else:
        return [x[0] for x in lemma]


def process_suffix(phrase):
    phrase = phrase.replace("'s", " 's")
    phrase = phrase.replace("can't", "can not")
    phrase = phrase.replace("n't", " not")
    return phrase


def extract_patterns(tokens_tagged, filt_general=None):
    if filt_general == "atomic":
        general_terms = ["personx", "personx's", "persony", "persony's", "personz", "personz's", "itemx"]
    elif filt_general == "persona":
        general_terms = [punc for punc in string.punctuation]
    else:
        general_terms = []
    word_num = len(tokens_tagged)
    word_tokens = [x[0] for x in tokens_tagged]
    selectable = [True] * word_num
    p_left = 0
    p_right = 0
    patterns = []
    while p_left < word_num and p_right < word_num:
        if p_right < p_left:
            p_right = p_left
        if word_tokens[p_left] in general_terms:
            selectable[p_left] = False
            p_left += 1
        else:
            first_word = word_tokens[p_left].split(" ")[0]
            idioms_subset_lemma = [x["lemma"] for x in idioms.get(first_word, [])]
            if len(idioms_subset_lemma) == 0:
                p_left += 1
            else:
                phrase = " ".join(word_tokens[p_left:(p_right+1)])
                p_right_prev = p_right
                p_right += 1
                # phrase = word_tokens[p_left]
                #  p_right = p_left + 1
                while p_right < word_num:
                    '''
                    if tokens_tagged[p_right][1].startswith("W"):
                        # p_right -= 1
                        p_right = p_right_prev  #
                        break
                    '''
                    phrase = phrase + " " + word_tokens[p_right]
                    if phrase in idioms_subset_lemma:
                        # selectable[p_right] = False
                        break
                    elif any(phrase in idiom for idiom in idioms_subset_lemma):
                        # selectable[p_right] = False
                        p_right += 1
                    else:
                        # phrase = " ".join(phrase.split(" ")[:-1])
                        # p_right -= 1
                        p_right = p_right_prev  #
                        break
                if p_right_prev < p_right < word_num:
                    for idx in range(p_left, p_right+1):
                        selectable[idx] = False
                    patterns.append(process_suffix(phrase))
                p_left += 1
                # p_left = p_right + 1

    filtered_type = ["CC", "PDT", "POS", "PRP", "PRP$", "SYM", "TO"]
    filtered_dt = ["a", "an", "the", "one", "some", "this", "that", "these", "those"]
    filtered_ps = ["one 's", "someone 's", "person 's", "people 's", "'s"]
    for idx in range(word_num):
        if not selectable[idx] or tokens_tagged[idx][1] in filtered_type or \
                (tokens_tagged[idx][1] == "DT" and tokens_tagged[idx][0] in filtered_dt):
            pass
        else:
            phrase = process_suffix(tokens_tagged[idx][0])
            for filt_p in filtered_ps:
                if phrase.endswith(filt_p):
                    phrase = phrase.replace(filt_p, "").strip()
                    break
            if phrase:
                patterns.append(phrase)
    return patterns


def combine_phrase(tokens_tagged, skip_general=None):
    if skip_general == "atomic":
        general_terms = ["PersonX", "PersonX's", "PersonY", "PersonY's", "PersonZ", "PersonZ's", "ItemX"]
    elif skip_general == "persona":
        general_terms = [punc for punc in string.punctuation]
    else:
        general_terms = []
    target_type = ["V"]  # ["V", "N"]
    idx = 0
    while idx < len(tokens_tagged):
        if tokens_tagged[idx][1][0] in target_type and tokens_tagged[idx][0] not in general_terms:
            left, right = idx, idx
            cb_type = tokens_tagged[idx][1][0]
            if cb_type == "V":
                sub_type = ["RB", "RBR", "RBS", "RP"]
            elif cb_type == "N":
                sub_type = ["VBG"]
            else:
                sub_type = []
            for idx_t in range(left + 1, len(tokens_tagged)):
                if tokens_tagged[idx_t][1][0] == cb_type and tokens_tagged[idx_t][0] not in general_terms:
                    right = idx_t
                elif tokens_tagged[idx_t][1] in sub_type and tokens_tagged[idx_t][0] not in general_terms:
                    pass
                else:
                    break
            if left < right:
                phrase = []
                for pos in range(right, left - 1, -1):
                    phrase.insert(0, tokens_tagged.pop(pos))
                phrase_string = " ".join(word[0] for word in phrase)
                tokens_tagged.insert(left, [phrase_string, cb_type + "PH"])
            idx = left + 1
        else:
            idx += 1
    return tokens_tagged
