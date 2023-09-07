import numpy as np
import re
import csv
import math
import os
from psycholinguistic.senticnet.tools import analysis_text
import random
from psycholinguistic.mairesse import Mairesse
import pandas as pd
import html
from urllib import parse
from concurrent.futures import ProcessPoolExecutor
from gpt2_bpe import get_encoder


def clean_text(text, remove_url=True, email=True, weibo_at=True, stop_terms=("转发",),
                 emoji=True, weibo_topic=False, deduplicate_space=True,
                 norm_url=False, norm_html=False, to_url=False,
                 remove_puncts=False, remove_tags=True, t2s=False,
                 expression_len=(1,6), linesep2space=False):
    '''
    进行各种文本清洗操作，特殊格式，网址，email，html代码，等等

    :param text: 输入文本
    :param remove_url: （默认使用）是否去除网址
    :param email: （默认使用）是否去除email
    :param weibo_at: （默认使用）是否去除\@相关文本
    :param stop_terms: 去除文本中的一些特定词语，默认参数为("转发",)
    :param emoji: （默认使用）去除\[\]包围的文本，一般是表情符号
    :param weibo_topic: （默认不使用）去除##包围的文本，一般是话题
    :param deduplicate_space: （默认使用）合并文本中间的多个空格为一个
    :param norm_url: （默认不使用）还原URL中的特殊字符为普通格式，如(%20转为空格)
    :param norm_html: （默认不使用）还原HTML中的特殊字符为普通格式，如(\&nbsp;转为空格)
    :param to_url: （默认不使用）将普通格式的字符转为还原URL中的特殊字符，用于请求，如(空格转为%20)
    :param remove_puncts: （默认不使用）移除所有标点符号
    :param remove_tags: （默认使用）移除所有html块
    :param t2s: （默认不使用）繁体字转中文
    :param expression_len: 假设表情的表情长度范围，不在范围内的文本认为不是表情，不加以清洗，如[加上特别番外荞麦花开时共五册]。设置为None则没有限制
    :param linesep2space: （默认不使用）把换行符转换成空格
    :return: 清洗后的文本
    '''
    # unicode不可见字符
    # 未转义
    text = re.sub(r"[\u200b-\u200d]", "", text)
    # 已转义
    text = re.sub(r"(\\u200b|\\u200c|\\u200d)", "", text)
    # 反向的矛盾设置
    if norm_url and to_url:
        raise Exception("norm_url和to_url是矛盾的设置")
    if norm_html:
        text = html.unescape(text)
    #if to_url:
    #    text = urllib.parse.quote(text)
    # if remove_tags:
    #     text = w3lib.html.remove_tags(text)
    if remove_url:
        try:
            sina_url = re.compile(r'<sina.*?>')
            text = re.sub(sina_url, "", text)
            URL_REGEX = re.compile(
                r'(?i)http[s]?://(?:[a-zA-Z]|[0-9]|[#$%*\-/;=?&@~.&+_]|[!*,])+',
                re.IGNORECASE)
            text = re.sub(URL_REGEX, "", text)

        except:
            # sometimes lead to "catastrophic backtracking"
            zh_puncts1 = "，；、。！？（）《》【】"
            URL_REGEX = re.compile(
                r'(?i)((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>' + zh_puncts1 + ']+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’' + zh_puncts1 + ']))',
                re.IGNORECASE)
            text = re.sub(URL_REGEX, "", text)

    if norm_url:
        text = parse.unquote(text)
    if email:
        EMAIL_REGEX = re.compile(r"[-a-z0-9_.]+@(?:[-a-z0-9]+\.)+[a-z]{2,6}", re.IGNORECASE)
        text = re.sub(EMAIL_REGEX, "", text)
    if weibo_at:
        text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:|：| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
    if emoji:
        # 去除括号包围的表情符号
        # ? lazy match避免把两个表情中间的部分去除掉
        if type(expression_len) in {tuple, list} and len(expression_len) == 2:
            # 设置长度范围避免误伤人用的中括号内容，如[加上特别番外荞麦花开时共五册]
            lb, rb = expression_len
            text = re.sub(r"\[\S{"+str(lb)+r","+str(rb)+r"}?\]", "", text)
        else:
            text = re.sub(r"\[\S+?\]", "", text)
        # text = re.sub(r"\[\S+\]", "", text)
        # 去除真,图标式emoji
        emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002702-\U000027B0"
                        u"\u2600-\u2B55" u"\U00010000-\U0010ffff"
                        "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
    if weibo_topic:
        # text = re.sub(r"#\S+#", "", text)  # 去除话题内容
        # re.sub(r"#\S+\s?\S+#", "", "#ds e哈 oa#分发")
        text = re.sub(r"#[^#]*?#","",text)
    if linesep2space:
        text = text.replace("\n", " ")   # 不需要换行的时候变成1行
    if deduplicate_space:
        text = re.sub(r"(\s)+", r" ", text)   # 合并正文中过多的空格
        # text = re.sub(r"(\s)+", r"\1", text)   # 合并正文中过多的空格

    # if t2s:
    #     cc = OpenCC('t2s')
    #     text = cc.convert(text)
    assert hasattr(stop_terms, "__iter__"), Exception("去除的词语必须是一个可迭代对象")
    if type(stop_terms) == str:
        text = text.replace(stop_terms, "")
    else:
        for x in stop_terms:
            text = text.replace(x, "")
    if remove_puncts:
        allpuncs = re.compile(
            r"[～，\_《。》、？；：‘’＂“”【「】」·！@￥…（）—\,\<\.\>\/\?\;\:\'\"\[\]\{\}\~\`\!\@\#\$\%\^\&\*\(\)\-\=\+]")
        text = re.sub(allpuncs, "", text)

    return text.strip()


def clean_basic(string):
    """
    Tokenization/string cleaning for basic need
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def get_mairesse_dict(file_name="mairesse.csv"):
    mairesse_dict = {}
    with open(file_name, 'r', encoding='utf-8-sig') as csvf:
        csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
        for line in csvreader:
            mairesse_dict[str(line[0]).strip()] = [float(f) for f in line[1:]]
    return mairesse_dict


def get_emotion_dict(file_name="psycholinguistic/Emotion_Lexicon.csv"):
    emotion_dict = {}
    with open(file_name, 'r', encoding='utf-8-sig') as csvf:
        csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
        next(csvreader)
        for line in csvreader:
            emotion_dict[str(line[0]).strip()] = [int(emo) for emo in line[1:]]
    return emotion_dict


def get_senticnet_tree(file_name="psycholinguistic/senticnet/sentic_tree.npy"):
    senticnet_tree = np.load(file_name, allow_pickle=True).item()
    return senticnet_tree


def get_sentic_emotion_vocab(file_name="psycholinguistic/senticnet/Emotion_Type.total"):
    sentic_emotion_vocab = []
    with open(file_name, 'r') as f:
        for type_word in f.readlines():
            sentic_emotion_vocab.append(str(type_word).strip())
    return sentic_emotion_vocab


def get_mairesse_feats(text_id, mairesse_dict):
    return np.round(np.array(mairesse_dict[text_id]), decimals=3)


def get_emotion_feats(text, scale, emotion_dict):
    emotion_feats = np.array([0.0]*11, dtype=np.float16)
    words = re.findall(r'\w+', text, re.UNICODE)
    for word in words:
        if word in emotion_dict.keys():
            emotion_feats += np.array(emotion_dict[str(word)])
    emotion_feats = (100.0 * emotion_feats * scale)
    return emotion_feats


def get_sentic_feats_tuple(text, scale, senticnet_tree, sentic_emotion_vocab):
    sentic_emotion_dict = dict.fromkeys(sentic_emotion_vocab, 0)
    sentic_dict = analysis_text(text, senticnet_tree)
    sentic_feats = np.array([0.0]*5, dtype=np.float16)
    for value_vector in sentic_dict.values():
        temp = value_vector[0:4] + [value_vector[7]]
        sentic_feats += np.array(temp, dtype=np.float16)
        sentic_emotion_dict[str(value_vector[4]).strip()] += 1

    def get_top_emotion(emotion_dict, num=5):
        sorted_items = sorted(emotion_dict.items(), key=lambda item: item[1], reverse=True)
        return np.array(sorted_items)[0: num, 0]

    top_emotion = get_top_emotion(sentic_emotion_dict)
    emotion_tokens = list(map(sentic_emotion_vocab.index, top_emotion))
    return np.round(100.0 * sentic_feats * scale, 3), np.array(emotion_tokens, dtype=int)


class DocEncoder:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.num_seg_big = 2
        self.num_seg_sma = 4

        self.pattern_standard = re.compile(r'!(?=[^!]+)|\.(?=[^\.]+)|\?(?=[^\?]+)')    # . ! ?
        self.pattern_secondary = re.compile(r',(?=[^,]+)')    # ,
        self.pattern_candidate = re.compile(
            r' (I|You|We|He|She|It|They|But) '
            r'| ((I|You|We|He|She|It|They|i|you|we|he|she|it|they)(\'s|\'ve|\'re|\'d|\'ll)) | I\'m | i\'m ')
        self.pattern_possible = re.compile(r' (i|you|we|he|she|it|they|and|but|:|;) ')

    def text2segment(self, text):

        def word_count(txt):
            words = re.findall(r'\w+', txt, re.UNICODE)
            return len(words)

        num_words = word_count(text)

        if num_words < 300:
            sentences = text.split('.')
            num_sentence = len(sentences)
            i = 0
            while num_words < 400:
                text = text + sentences[i]
                i = (i + 1) % num_sentence
                num_words = word_count(text)

        num_words_of_seg_big = num_words/self.num_seg_big
        num_words_of_seg_sma = num_words/self.num_seg_sma

        num_words_of_seg_big = 200 if num_words_of_seg_big < 150 else num_words_of_seg_big
        num_words_of_seg_sma = 100 if num_words_of_seg_sma < 75 else num_words_of_seg_sma

        cut_dict = {}
        for i in re.finditer(self.pattern_standard, text):
            cut_dict[i.end()] = 4
        for i in re.finditer(self.pattern_secondary, text):
            cut_dict[i.end()] = 3
        for i in re.finditer(self.pattern_candidate, text):
            cut_dict[i.start()] = 2
        for i in re.finditer(self.pattern_possible, text):
            cut_dict[i.start()] = 1
        cut_dict[len(text)] = 4
        cut_set_copy = sorted(cut_dict.items())  # tuple [(index, priority),...]

        def in_boundary(current, target):
            if math.fabs(current-target) / target < 0.17:
                return True
            else:
                return False

        def before_boundary(current, target):
            if math.fabs(current-target) / target > 0.17 and current < target:
                return True
            else:
                return False

        def after_boundary(current, target):
            if math.fabs(current-target) / target > 0.17 and target < current:
                return True
            else:
                return False

        def cut_out(s, segments):
            s = s.strip()
            segments.append(s)
            return segments

        def cut_in(s, segments):
            if len(segments) > 0:
                s = s.strip()
                segments[-1] += s
            else:
                segments = cut_out(s, segments)
            return segments

        def process(cut_set, num_seg, num_words_of_seg):
            segments = []

            pre_index = 0
            while len(cut_set) > 0:
                # end condition
                if pre_index == cut_set[-1][0]:
                    break

                tmp_set = cut_set.copy()
                search_record = None
                for cut in cut_set:   # (1, 1)
                    cur_text = text[pre_index: cut[0]]

                    if cut == cut_set[-1]:
                        if len(segments) < num_seg:
                            cut_out(cur_text, segments)
                        else:
                            cut_in(cur_text, segments)
                        pre_index = cut[0]
                        break

                    if before_boundary(word_count(cur_text), num_words_of_seg):
                        continue

                    if after_boundary(word_count(cur_text), num_words_of_seg):
                        if search_record is None:
                            cut_out(cur_text, segments)
                            pre_index = cut[0]
                        else:
                            # cut segment according to search_record
                            for p in (3, 2, 1):
                                if p in search_record.keys():
                                    cut_index = search_record[p]
                                    cut_out(text[pre_index: cut_index], segments)
                                    pre_index = cut_index
                                    for tmp_cut in cut_set:
                                        if tmp_cut[0] == cut_index:
                                            tmp_set.remove(tmp_cut)
                                            cut_set = tmp_set
                                            break
                                        tmp_set.remove(tmp_cut)
                                    break
                            break

                        continue

                    if in_boundary(word_count(cur_text), num_words_of_seg):
                        priority = cut[1]
                        if priority == 4:
                            cut_out(cur_text, segments)
                            search_record = None
                            pre_index = cut[0]
                            continue

                        if search_record is None:
                            search_record = {priority: cut[0]}
                            continue

                        if priority not in search_record.keys():
                            search_record[priority] = cut[0]
            return segments

        big_segments = process(cut_set_copy.copy(), self.num_seg_big, num_words_of_seg_big)
        assert len(big_segments) == self.num_seg_big
        sma_segments = process(cut_set_copy.copy(), self.num_seg_sma, num_words_of_seg_sma)
        assert len(sma_segments) == self.num_seg_sma

        segments_res = big_segments + sma_segments

        enc_segments_res = []
        for seg in segments_res:
            enc_segments_res.append(self.tokenizer.encode(seg)[:510])

        return segments_res, enc_segments_res


class Encoder(object):

    tokenizer = get_encoder("gpt2_bpe/encoder.json", "gpt2_bpe/vocab.bpe")
    encoder = DocEncoder(tokenizer)
    mairesser = Mairesse()
    mairesse_dict = get_mairesse_dict()
    emotion_dict = get_emotion_dict()
    senticnet_tree = get_senticnet_tree()
    sentic_emotion_vocab = get_sentic_emotion_vocab()
    seg_columns = ['user', 'big5_str', 'enc_tokens_str'] + ['mairesse'] * 96 + ['senticnet'] * 5 + ['emotion'] * 11
    doc_columns = ['user', 'big5_str', 'enc_tokens_str'] + ['mairesse'] * 84 + ['senticnet'] * 5 + ['emotion'] * 11 + ['senticnet_dis'] * 5

    @classmethod
    def sample_process(cls, line):

        def text_scale(txt):
            words = re.findall(r'\w+', txt, re.UNICODE)
            return 1 / len(words)

        def strip(t):
            return str(t).strip()

        user = str(line[0].strip())
        text = clean_basic(str(line[1]))
        big5 = list(map(strip, line[2:7]))
        segments, enc_segments = cls.encoder.text2segment(text)

        # ext, neu, agr, con, opn = big5
        big5_str = ' '.join(big5)

        essay_scale = text_scale(text)

        doc_enc_tokens = []

        df_seg_items = pd.DataFrame(columns=[cls.seg_columns, np.arange(1, 116)])

        seg_index = 0
        for seg, enc_seg in zip(segments, enc_segments):
            seg_index = seg_index + 1

            try:
                scale = text_scale(seg)
            except ZeroDivisionError:
                print(seg)
                continue

            mairesse = cls.mairesser.extractor(seg)
            senticnet, _ = get_sentic_feats_tuple(seg, scale, cls.senticnet_tree, cls.sentic_emotion_vocab)
            emotion = get_emotion_feats(seg, scale, cls.emotion_dict)
            enc_seg_str = ' '.join(map(str, enc_seg))

            row = [user, big5_str, enc_seg_str] + mairesse.tolist() + senticnet.tolist() + emotion.tolist()
            df_seg_item = pd.DataFrame([row], columns=[cls.seg_columns, np.arange(1, 116)])

            df_seg_items = pd.concat([df_seg_items, df_seg_item], ignore_index=True)
            doc_enc_tokens.extend(enc_seg)
            doc_enc_tokens.append(-100)

        doc_enc_tokens.pop()
        enc_tokens_str = ' '.join(map(str, doc_enc_tokens))
        mairesse = get_mairesse_feats(user, cls.mairesse_dict)
        senticnet, senticnet_dis = get_sentic_feats_tuple(text, essay_scale, cls.senticnet_tree, cls.sentic_emotion_vocab)
        emotion = get_emotion_feats(text, essay_scale, cls.emotion_dict)

        row = [user, big5_str,
               enc_tokens_str] + mairesse.tolist() + senticnet.tolist() + emotion.tolist() + senticnet_dis.tolist()
        df_doc_item = pd.DataFrame([row], columns=[cls.doc_columns, np.arange(1, 109)])

        return df_seg_items, df_doc_item


def build_data_set_pack(csvfile):
    """
    Loads data and split into 4 folds.
    """

    df_seg = pd.DataFrame(columns=[Encoder.seg_columns, np.arange(1, 116)])
    df_doc = pd.DataFrame(columns=[Encoder.doc_columns, np.arange(1, 109)])

    pool = ProcessPoolExecutor(max_workers=os.cpu_count())

    csvf = open(csvfile, 'r', encoding='utf-8')
    csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
    next(csvreader)

    ind = 0
    for df_seg_items, df_doc_item in pool.map(Encoder.sample_process, csvreader):
        print('process {} row'.format(ind))
        ind = ind + 1
        df_seg = pd.concat([df_seg, df_seg_items], ignore_index=True)
        df_doc = pd.concat([df_doc, df_doc_item], ignore_index=True)

    normalize_func = lambda x: (x - x.mean()) / x.std()

    df_seg['mairesse'] = df_seg['mairesse'].apply(normalize_func)
    df_seg['senticnet'] = df_seg['senticnet'].apply(normalize_func)
    df_seg['emotion'] = df_seg['emotion'].apply(normalize_func)
    df_seg.dropna(axis=1, inplace=True)
    #
    df_doc['senticnet'] = df_doc['senticnet'].apply(normalize_func)
    df_doc['emotion'] = df_doc['emotion'].apply(normalize_func)
    df_doc.dropna(axis=1, inplace=True)

    seg_dataset_pack = []
    doc_dataset_pack = []
    for index, row in df_doc.iterrows():
        user = row['user'].values[0]
        doc_dataset = {
            'big5_str': row['big5_str'].values[0],
            'enc_tokens_str': row['enc_tokens_str'].values[0],
            'mairesse': row['mairesse'].values,
            'senticnet': row['senticnet'].values,
            'emotion': row['emotion'].values,
            'senticnet_dis': row['senticnet_dis'].values,
        }

        df = df_seg.loc[df_seg['user'][1] == user]
        seg_tmp_pack = []
        seg_mairesse = []
        seg_senticnet = []
        seg_emotion = []

        for i, r in df.iterrows():
            mai = r['mairesse'].values
            sen = r['senticnet'].values
            emo = r['emotion'].values

            if i == 1:
                print('mairesse length: ' + str(len(mai)))

            seg_tmp_pack.append({
                'big5_str': r['big5_str'].values[0],
                'enc_tokens_str': r['enc_tokens_str'].values[0],
                'mairesse': mai,
                'senticnet': sen,
                'emotion': emo,
            })
            seg_mairesse.append(mai)
            seg_senticnet.append(sen)
            seg_emotion.append(emo)

        seg_dataset_pack.append(seg_tmp_pack)

        doc_dataset.update({
            'seg_mairesse': seg_mairesse,
            'seg_senticnet': seg_senticnet,
            'seg_emotion': seg_emotion,
        })
        doc_dataset_pack.append(doc_dataset)

    # 随机排序
    combined = list(zip(seg_dataset_pack, doc_dataset_pack))
    random.seed(85)
    random.shuffle(combined)
    seg_dataset_pack, doc_dataset_pack = zip(*combined)

    seg_dataset_pack = np.array(seg_dataset_pack, dtype=object)
    doc_dataset_pack = np.array(doc_dataset_pack, dtype=object)

    return seg_dataset_pack, doc_dataset_pack


def split_pack(dataset_pack, k_fold=4, seed=0, data_type='seg_dataset'):
    data_folds = np.array_split(dataset_pack, k_fold)
    dataset_valid = data_folds.pop(seed)
    dataset_train = [item for fold in data_folds for item in fold]

    def flatten(dataset):
        tmp_dataset = []
        for pack in dataset:
            for data_item in pack:
                tmp_dataset.append(data_item)
        return tmp_dataset

    if data_type == 'seg_dataset':
        #  unzip
        dataset_valid = flatten(dataset_valid)
        dataset_train = flatten(dataset_train)

    return dataset_valid, dataset_train


def pack2dataset(dataset_pack, split, data_type='seg_dataset', output_dir=''):
    def get_data_path(file_name, prefix=None):
        if prefix is not None:
            out_dir = os.path.join(output_dir, data_type, prefix)
        else:
            out_dir = os.path.join(output_dir, data_type)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        return os.path.join(out_dir, file_name)

    def save_to_np(prefix, dataset):
        name = '{}.npy'.format(split)
        np.save(get_data_path(name, prefix=prefix), dataset)

    input_name = '{}.input'.format(split)
    input_file = open(get_data_path(input_name), 'w', encoding='utf-8')

    big5_name = '{}.label'.format(split)
    big5_file = open(get_data_path(big5_name), 'w', encoding='utf-8')

    mairesse = []
    emotion = []
    senticnet = []

    seg_mairesse = []
    seg_emotion = []
    seg_senticnet = []

    senticnet_dis = []

    for data_item in dataset_pack:
        print(data_item['enc_tokens_str'], file=input_file)
        print(data_item['big5_str'], file=big5_file)

        mairesse.append(data_item['mairesse'])
        emotion.append(data_item['emotion'])
        senticnet.append(data_item['senticnet'])

        if data_type == 'doc_dataset':
            seg_mairesse.append(data_item['seg_mairesse'])
            seg_emotion.append(data_item['seg_emotion'])
            seg_senticnet.append(data_item['seg_senticnet'])
            senticnet_dis.append(data_item['senticnet_dis'])

    if data_type == 'seg_dataset':
        save_to_np('feats/mairesse', np.array(mairesse))
        save_to_np('feats/senticnet', np.array(emotion))
        save_to_np('feats/emotion', np.array(senticnet))

    else:
        save_to_np('feats/seg/mairesse', np.array(seg_mairesse, dtype=object))
        save_to_np('feats/seg/senticnet', np.array(seg_emotion, dtype=object))
        save_to_np('feats/seg/emotion', np.array(seg_senticnet, dtype=object))

        save_to_np('feats/doc/mairesse', np.array(mairesse))
        save_to_np('feats/doc/senticnet', np.array(senticnet))
        save_to_np('feats/doc/emotion', np.array(emotion))
        save_to_np('feats/doc/senticnet_dis', np.array(senticnet_dis))


def process_for_pd():
    csvfile = "essays.csv"
    print('building big5 dataset···')
    seg_dataset_np, doc_dataset_np = build_data_set_pack(csvfile)
    print('finish!')

    k_fold = 10
    dir_name = '{}-{}'.format(2, 4)
    print('=====ready to pack big5 dataset to k-folds=====')

    for data_type, dataset_np in (('seg_dataset', seg_dataset_np), ('doc_dataset', doc_dataset_np)):
        print('working on building {}···'.format(data_type))
        for i in range(k_fold):
            print('working on {}-fold···'.format(i))
            dataset_valid, dataset_train = split_pack(dataset_np, k_fold=k_fold, seed=i, data_type=data_type)
            output_dir = '{}/fold-{}'.format(dir_name, str(i))
            print('building valid dataset···')
            pack2dataset(dataset_valid, split='valid', data_type=data_type, output_dir=output_dir)
            print('finish!')
            print('building train dataset···')
            pack2dataset(dataset_train, split='train', data_type=data_type, output_dir=output_dir)
            print('finish!')
    print('succeed in process dataset for big5 personality detection!')


if __name__ == "__main__":
    process_for_pd()


# essay preprocess in word
# 抦	'm
# 抯	's
# 抰	't
# 抎	'd
# 抳	've
# 抣	'l
