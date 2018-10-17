import nltk
import re
import os
import json

from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet


class IO:

    @staticmethod
    def read_from_file(dataFile):
        fileFormat = ".json"
        if not os.path.exists(dataFile + fileFormat):
            return None
        with open(dataFile + fileFormat, "r") as file:
            return json.load(file)

    @staticmethod
    def store_to_file(dataFile, data):
        fileFormat = ".json"
        with open(dataFile + fileFormat, "w") as file:
            json.dump(data, file)


class Utils:

    # convert all symbols(except '-') to space
    @staticmethod
    def remove_symbols(s):
        return re.sub(r"[^a-zA-Z\-]", " ", s)

    # string to list
    # @staticmethod
    # def to_list(s):
    #     return nltk.word_tokenize(Utils.remove_symbols(s))

    @staticmethod
    def to_list(s):
        return Utils.remove_symbols(s).split()

    # list to string
    @staticmethod
    def to_string(l, c=' '):
        return c.join(l)

    # lower word list
    @staticmethod
    def to_lower(words):
        return list(map(lambda x: x.lower(), words))

    # rebuild list from string list with actual types
    @staticmethod
    def convert_types(strs, types):
        return list(map(lambda str, type: type(str), strs, types))

    # get word sentiment from nltk.corpus - sentiwordnet
    # first init time: 2s, query time: 0.001s
    # return (positive-negative) in range [-1.0, 1.0]
    @staticmethod
    def get_sentiment(word):
        m, p, f = 0, 0, 0
        for t in swn.senti_synsets(word):
            n = abs(t.pos_score() - t.neg_score())
            if n > m:
                m, p, f = n, t.pos_score(), t.neg_score()
        return (p - f)

    # get antonym from synsets
    @staticmethod
    def get_antonym(word):
        synset = wordnet.synsets(word)
        for syn in synset:
            for l in syn.lemmas():
                if l.antonyms():
                    return l.antonyms()[0].name()

    # generate range string list
    # ['0.5', '1.0', ..., '5.0']
    @staticmethod
    def get_string_range(base, offset):
        base = float(base)
        if base == 0 or offset == 0:
            return [str(base)]
        base = int(base * 10)
        offset = int(offset * 10)
        low = max(base - offset, 5)
        up = min(base + offset, 50) + 1
        return [str(i / 10) for i in range(low, up, 5)]

    # get precision and recall
    # parameters are two set
    @staticmethod
    def get_indices(reco, real):
        len_reco = len(reco)
        len_real = len(real)
        len_inte = len(reco & real)
        if len_inte == 0:
            return [0, 0, 0]
        pre = len_inte / len_reco
        rec = len_inte / len_real
        f1 = (2 * pre * rec) / (pre + rec)
        return [pre, rec, f1]

    # get total number of elements in list
    @staticmethod
    def get_num(l):
        lens = 0
        for e in l:
            if not e:
                continue
            elif not isinstance(e, list):
                lens += 1
            else:
                lens += Utils.get_num(e)
        return lens

    # flatten nested list to 1-d list
    @staticmethod
    def flatten(l):
        new = []
        for e in l:
            if not e:
                continue
            elif not isinstance(e, list):
                new.append(e)
            else:
                new.extend(Utils.flatten(e))
        return new

    @staticmethod
    def extend_list(a, b, lens=3):
        for i in range(lens):
            a[i].extend(b[i])

    @staticmethod
    def most_common(c, n):
        return [t[0] for t in c.most_common(n)]

    @staticmethod
    def update_counters(a, b, lens=3):
        for i in range(lens):
            a[i].update(b[i])

