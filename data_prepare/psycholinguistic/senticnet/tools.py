import re


def clean(string, TREC=False):
    """
    Tokenization/string cleaning for extracting emotion、senticnet and so on
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " is ", string)
    string = re.sub(r"\'ve", " have ", string)
    string = re.sub(r"n\'t", " not ", string)
    string = re.sub(r"\'re", " are ", string)
    string = re.sub(r"\'d" , " would ", string)
    string = re.sub(r"\'ll", " will ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
#    string = re.sub(r"[a-zA-Z]{4,}", "", string)
    string = re.sub(r"\s{2,}", " ", string) # 空白字符
    return string.strip() if TREC else string.strip().lower()


def analysis_in_tree(tree, words):
    # words[0] must be in tree.keys
    kept_value = tree[words[0]]['concept-value']
    sub_tree = tree[words[0]]['sub-concepts']
    words.pop(0)
    if len(words) > 1 and words[0] in sub_tree.keys():
        explored_value = analysis_in_tree(sub_tree, words)
        if explored_value:
            kept_value = explored_value
    return kept_value


def analysis_by_word(sentic_tree, words, raw_vectors):
    while words:
        if words[0] not in sentic_tree.keys():
            words.pop(0)
            continue
        kept_words = words.copy()
        sentic_vector = analysis_in_tree(sentic_tree, words)
        if sentic_vector:
            raw_vectors['_'.join(kept_words[:len(kept_words)-len(words)])] = sentic_vector


def analysis_text(text, sentic_tree):
    words = re.findall(r'\w+', text, re.UNICODE)
    sentic_dict = {}
    analysis_by_word(sentic_tree, words, sentic_dict)
    return sentic_dict

