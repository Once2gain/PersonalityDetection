import re
import liwc
import numpy as np
import spacy


class Mairesse:
    def __init__(self):
        self.LIWC_parse, self.LIWC_category = liwc.load_token_parser('psycholinguistic/Mairesse_material/LIWC2007_English100131.dic')
        self.MRC_category = [
            'nlet',
            'nphon',
            'nsyl',
            'kf_freq',
            'kf_ncats',
            'kf_nsamp',
            'tl_freq',
            'brown_freq',
            'fam',
            'conc',
            'imag',
            'meanc',
            'meanp',
            'aoa',
        ]

        self.MRC_parse = {}
        with open('psycholinguistic/Mairesse_material/mrc2.dct') as f:
            for line in f:
                fields = line.strip().split()
                word = fields[-1].split("|", 1)[0].lower()
                pos = line[44]
                item = {
                            'pos': pos,
                            'value': {
                                'nlet': int(line[0:2]),
                                'nphon': int(line[2:4]),
                                'nsyl': int(line[4]),
                                'kf_freq': int(line[5:10]),
                                'kf_ncats': int(line[10:12]),
                                'kf_nsamp': int(line[12:15]),
                                'tl_freq': int(line[15:21]),
                                'brown_freq': int(line[21:25]),
                                'fam': int(line[25:28]),
                                'conc': int(line[28:31]),
                                'imag': int(line[31:34]),
                                'meanc': int(line[34:37]),
                                'meanp': int(line[37:40]),
                                'aoa': int(line[40:43]),
                            }
                        }
                if word not in self.MRC_parse:
                    self.MRC_parse[word] = [
                        item,
                    ]
                else:
                    self.MRC_parse[word].append(
                        item
                    )

        self.blank_LIWC_dict = {}
        for name in self.LIWC_category:
            self.blank_LIWC_dict[name] = 0

        self.blank_MRC_value = []
        for _ in self.MRC_category:
            self.blank_MRC_value.append(0.0)

        self.pos_map = {
            'NOUN': ['N'],
            'PROPN': ['N'],
            'ADJ': ['J'],
            'VERB': ['V', 'P'],
            'ADV': ['A'],
            'ADP': ['A', 'R', 'C'],
            'CCONJ': ['C'],
            'PRON': ['U'],
            'DET': ['U'],
            'INTJ': ['I']
        }

        self.nlp = spacy.load("psycholinguistic/Mairesse_material/en_core_web_sm/en_core_web_sm-3.6.0")


    @staticmethod
    def tokenize(str_text):
        # you may want to use a smarter tokenizer
        for match in re.finditer(r'\w+', str_text, re.UNICODE):
            yield match.group(0)

    def extra_feature(self, text, tokens, tokens_count):
        sentences = re.split(r'\s*[.!?]+\s+', text)

        WPS = tokens_count / len(sentences)

        six_letters = 0
        for token in tokens:
            if len(token) > 6:
                six_letters += 1
        # numbers = len(re.findall(r"\d+", text))

        UNIQUE = 100.0 * len(set(tokens)) / tokens_count

        SIXLTR = 100.0 * six_letters / tokens_count

        abbrev = len(re.findall(r"\w\.(\w\.)+", text))
        ABBREVIATIONS = 100.0 * abbrev / tokens_count

        emoticons = 100.0 * len(re.findall(r"[:;8%]-[)(@\[\]|]+", text))
        EMOTICONS = emoticons / tokens_count

        qmarks = len(re.findall(r"\w\s*\?", text))
        QMARKS = 100.0 * qmarks / len(sentences)

        period = len(re.findall(r"\.", text))
        PERIOD = 100.0 * period / tokens_count

        comma = len(re.findall(r",", text))
        COMMA = 100.0 * comma / tokens_count

        colon = len(re.findall(r":", text))
        COLON = 100.0 * colon / tokens_count

        semicolon = len(re.findall(r";", text))
        SEMIC = 100.0 * semicolon / tokens_count

        qmark = len(re.findall(r"\?", text))
        QMARK = 100.0 * qmark / tokens_count

        exclam = len(re.findall(r"!", text))
        EXCLAM = 100.0 * exclam / tokens_count

        dash = len(re.findall(r"-", text))
        DASH = 100.0 * dash / tokens_count

        quote = len(re.findall(r"\"", text))
        QUOTE = 100.0 * quote / tokens_count

        apostr = len(re.findall(r"'", text))
        APOSTRO = 100.0 * apostr / tokens_count

        parent = len(re.findall(r"[(\[{]", text))
        PARENTH = 100.0 * parent / tokens_count

        otherp = len(re.findall(r"[^\w\d\s.:;?!\"'({\[,-]", text))
        OTHERP = 100.0 * otherp / tokens_count

        allp = period + comma + colon + semicolon + qmark + exclam + dash + quote + apostr + parent + otherp
        ALLPCT = 100.0 * allp / tokens_count

        return [WPS, UNIQUE, SIXLTR, ABBREVIATIONS, EMOTICONS, QMARKS, PERIOD, COMMA,
                COLON, SEMIC, QMARK, EXCLAM, DASH, QUOTE, APOSTRO, PARENTH, OTHERP, ALLPCT]

    def extractor(self, text):
        LIWC_dict = self.blank_LIWC_dict.copy()
        MRC_value = np.array(self.blank_MRC_value.copy(), dtype=np.float32)

        tokens_yield = self.tokenize(text.lower())
        tokens = []
        for token in tokens_yield:
            tokens.append(token)
            for category in self.LIWC_parse(token):
                LIWC_dict[category] += 1

            pos = self.nlp(token)[0].pos_
            # creating POS tag list
            if pos in self.pos_map:
                tag_list = self.pos_map[pos].copy()
            else:
                tag_list = ['O']

            if token in self.MRC_parse.keys():
                parse_res = self.MRC_parse[token]
                for item in parse_res:
                    if item['pos'] in tag_list:
                        MRC_value = MRC_value + np.array(list(item['value'].values()), dtype=np.float32)

                        tag_list.remove(item['pos'])
                        if not tag_list:
                            break

        tokens_count = len(tokens)
        others = self.extra_feature(text, tokens, tokens_count)
        LIWC = 100.0 * np.array(list(LIWC_dict.values()), dtype=np.float32) / tokens_count
        MRC = MRC_value / tokens_count

        features = np.concatenate([LIWC, MRC, others])

        return features




