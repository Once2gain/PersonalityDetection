import numpy as np
import re
import csv
import os
from gpt2_bpe import get_encoder
import random
from psycholinguistic.mairesse import Mairesse
import pandas as pd
from process_essay import clean_text, clean_basic, get_emotion_dict, get_senticnet_tree, get_sentic_emotion_vocab, \
    get_emotion_feats, get_sentic_feats_tuple
from concurrent.futures import ProcessPoolExecutor


class DocEncoder:

    def __init__(self, tokenizer, boundary):
        self.tokenizer = tokenizer
        self.num_seg_big = boundary[0]
        self.num_seg_sma = boundary[1]

    @staticmethod
    def clean_posts(posts):
        new_posts = []
        for post in posts:
            post = clean_text(post)
            if post and len(re.findall(r'[A-Za-z]', post)):
                new_posts.append(post)
        return new_posts

    def posts2segment(self, posts):
        posts = DocEncoder.clean_posts(posts)
        num_post = len(posts)
        while num_post < self.num_seg_sma:
            copy_index = random.randint(0, num_post-1)
            posts.append(posts[copy_index])
            num_post += 1

        segments = []
        enc_segments = []

        for ary in np.array_split(posts, self.num_seg_big):
            segments.append(' '.join(ary.tolist()))
        random.shuffle(posts)
        for ary in np.array_split(posts, self.num_seg_sma):
            segments.append(' '.join(ary.tolist()))

        for segs in segments:
            enc_segments.append(self.tokenizer.encode(segs)[:510])

        return segments, enc_segments


class Encoder:
    tokenizer = get_encoder("gpt2_bpe/encoder.json", "gpt2_bpe/vocab.bpe")
    mairesser = Mairesse()
    emotion_dict = get_emotion_dict()
    senticnet_tree = get_senticnet_tree()
    sentic_emotion_vocab = get_sentic_emotion_vocab()
    seg_columns = ['user', 'mbti_str', 'enc_tokens_str'] + ['mairesse'] * 96 + ['senticnet'] * 5 + ['emotion'] * 11
    doc_columns = ['user', 'mbti_str', 'enc_tokens_str'] + ['mairesse'] * 96 + ['senticnet'] * 5 + [
        'emotion'] * 11 + ['senticnet_dis'] * 5
    encoder = None

    @classmethod
    def build_encoder(cls, boundary):
        cls.encoder = DocEncoder(cls.tokenizer, boundary)

    @classmethod
    def sample_process(cls, row):

        def text_scale(txt):
            words = re.findall(r'\w+', txt, re.UNICODE)
            return 1 / len(words)

        def strip(t):
            return str(t).strip()

        user = ''

        posts = row[1].split('|||')
        posts = list(map(clean_basic, posts))

        # if len(posts) < 100:
        #     continue

        segments, enc_segments = cls.encoder.posts2segment(posts)

        EI, SN, TF, JP = strip(row[0])
        EI = '1' if EI == 'E' else '0'
        SN = '1' if SN == 'S' else '0'
        TF = '1' if TF == 'T' else '0'
        JP = '1' if JP == 'J' else '0'

        mbti_str = ' '.join([EI, SN, TF, JP])

        raw_text = ' '.join(posts)
        doc_scale = text_scale(raw_text)

        doc_enc_tokens = []

        df_seg_items = pd.DataFrame(columns=[cls.seg_columns, np.arange(1, 116)])

        for seg, enc_seg in zip(segments, enc_segments):
            try:
                scale = text_scale(seg)
            except ZeroDivisionError:
                print(seg)
                continue

            mairesse = cls.mairesser.extractor(seg)
            senticnet, _ = get_sentic_feats_tuple(seg, scale, cls.senticnet_tree, cls.sentic_emotion_vocab)
            emotion = get_emotion_feats(seg, scale, cls.emotion_dict)
            enc_seg_str = ' '.join(map(str, enc_seg))

            row = [user, mbti_str, enc_seg_str] + mairesse.tolist() + senticnet.tolist() + emotion.tolist()
            df_seg_item = pd.DataFrame([row], columns=[cls.seg_columns, np.arange(1, 116)])

            df_seg_items = pd.concat([df_seg_items, df_seg_item], ignore_index=True)
            doc_enc_tokens.extend(enc_seg)
            doc_enc_tokens.append(-100)

        doc_enc_tokens.pop()
        enc_tokens_str = ' '.join(map(str, doc_enc_tokens))
        mairesse = cls.mairesser.extractor(raw_text)
        senticnet, senticnet_dis = get_sentic_feats_tuple(raw_text, doc_scale, cls.senticnet_tree, cls.sentic_emotion_vocab)
        emotion = get_emotion_feats(raw_text, doc_scale, cls.emotion_dict)

        row = [user, mbti_str,
               enc_tokens_str] + mairesse.tolist() + senticnet.tolist() + emotion.tolist() + senticnet_dis.tolist()
        df_doc_item = pd.DataFrame([row], columns=[cls.doc_columns, np.arange(1, 121)])

        return df_seg_items, df_doc_item


def build_data_set_pack(csvfile, boundary):
    """
    Loads data and split into 4 folds.
    """
    df_seg = pd.DataFrame(columns=[Encoder.seg_columns, np.arange(1, 116)])
    df_doc = pd.DataFrame(columns=[Encoder.doc_columns, np.arange(1, 121)])

    pool = ProcessPoolExecutor(max_workers=os.cpu_count())

    csvf = open(csvfile, 'r', encoding='utf-8')
    csvreader = csv.reader(csvf, delimiter=',', quotechar='"')
    next(csvreader)
    ind = 0

    Encoder.build_encoder(boundary)
    for df_seg_items, df_doc_item in pool.map(Encoder.sample_process, csvreader):
        print('process {} row'.format(ind))
        df_seg_items['user'] = 'mbti-{}'.format(ind)
        df_doc_item['user'] = 'mbti-{}'.format(ind)
        ind = ind + 1
        df_seg = pd.concat([df_seg, df_seg_items], ignore_index=True)
        df_doc = pd.concat([df_doc, df_doc_item], ignore_index=True)

    dir_name = 'mbti-{}-{}/tmp'.format(boundary[0], boundary[1])
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    df_seg.to_csv(dir_name + "/df_seg.csv")
    df_doc.to_csv(dir_name + "/df_doc.csv")

    normalize = lambda x: (x - x.mean()) / x.std()

    df_seg['mairesse'] = df_seg['mairesse'].apply(normalize)
    df_seg['senticnet'] = df_seg['senticnet'].apply(normalize)
    df_seg['emotion'] = df_seg['emotion'].apply(normalize)
    df_seg.dropna(axis=1, inplace=True)
    #
    df_doc['mairesse'] = df_doc['mairesse'].apply(normalize)
    df_doc['senticnet'] = df_doc['senticnet'].apply(normalize)
    df_doc['emotion'] = df_doc['emotion'].apply(normalize)
    df_doc.dropna(axis=1, inplace=True)

    try:
        print(len(df_seg['mairesse'].values))
        print(len(df_seg['senticnet'].values))
        print(len(df_seg['emotion'].values))
        print(len(df_doc['mairesse'].values))
        print(len(df_doc['senticnet'].values))
        print(len(df_doc['emotion'].values))
    except AttributeError:
        print('error')

    seg_dataset_pack = []
    doc_dataset_pack = []
    for index, row in df_doc.iterrows():
        user = row['user'].values[0]
        doc_dataset = {
            'mbti_str': row['mbti_str'].values[0],
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

            seg_tmp_pack.append({
                'mbti_str': r['mbti_str'].values[0],
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
    random.seed(811)
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

    mbti_name = '{}.label'.format(split)
    mbti_file = open(get_data_path(mbti_name), 'w', encoding='utf-8')

    mairesse = []
    emotion = []
    senticnet = []

    seg_mairesse = []
    seg_emotion = []
    seg_senticnet = []

    senticnet_dis = []

    for data_item in dataset_pack:
        print(data_item['enc_tokens_str'], file=input_file)
        print(data_item['mbti_str'], file=mbti_file)

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


def process_for_pd(boundary):
    csvfile = "mbti_1.csv"
    print('building mbti dataset···')
    seg_dataset_np, doc_dataset_np = build_data_set_pack(csvfile, boundary)
    print('finish!')

    k_fold = 10
    dir_name = '../DATA-bin/kaggle-mbti/mbti-{}-{}'.format(boundary[0], boundary[1])
    print('=====ready to pack mbti dataset to k(4)-folds=====')

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
    print('succeed in process dataset for mbti personality detection!')


if __name__ == "__main__":
    process_for_pd(boundary=(3, 5))


# essay preprocess in word
# 抦	'm
# 抯	's
# 抰	't
# 抎	'd
# 抳	've
# 抣	'l
