import pandas as pd
import os
import numpy as np
import json
from modules.data.DataManagement import DataManager

class NodeMerge():
    def __init__(self):
        self.name = "NodeMerge"

    def merge(self, dm: DataManager) -> DataManager:
        raise NotImplementedError


class NodeMergeByTokenize(NodeMerge):
    def __init__(self):
        super(NodeMerge, self).__init__()
        self.name = "NodeMergeByTokenize"
        self.stopword_path = "modules/data/utils/stopwords_en.txt"

    # tokenize filter
    def lenFilter(self, word):
        return len(word) >= 2

    # tokenize
    def tokenize(self, text, stopwords=[]):
        original_text = str(text).lower()
        tok = original_text.split(' ')
        text = u''
        for x in tok:
            if len(x) == 0:
                # remove empty word
                continue
            elif x[0:4] == 'http' or x[0:5] == 'https':
                # remove url
                continue
            elif x[0] == '@':
                # remove mention
                continue
            elif x[0] == '#':
                # remove hashtag
                continue
            elif x in stopwords:
                # remove stopwords
                continue
            text = text + ' ' + x
        translate_to = u' '
        word_sep = u" ,.?:;'\"/<>`!$%^&*()-=+~[]\\|{}()\n\t" \
                   + u"©℗®℠™،、⟨⟩‒–—―…„“”–――»«›‹‘’：（）！？=【】　・" \
                   + u"⁄·† ‡°″¡¿÷№ºª‰¶′″‴§|‖¦⁂❧☞‽⸮◊※⁀「」﹁﹂『』﹃﹄《》―—" \
                   + u"“”‘’、，一。►…¿«「」ー⋘▕▕▔▏┈⋙一ー।;!؟"
        word_sep = u'#' + word_sep
        translate_table = dict((ord(char), translate_to) for char in word_sep)
        tokens = text.translate(translate_table).split(' ')
        return ' '.join(sorted(list(filter(self.lenFilter, tokens))))

    # from rawTweet to clean keyword text
    def textProcess(self, dataframe, ithreshold=None, uthreshold=None):
        stopwords = []
        # get stopwords
        with open(self.stopword_path, 'r') as infile:
            for word in infile.readlines():
                stopwords.append(word[:-1])
        dataframe['item.tokenized'] = dataframe.item.apply(
            lambda x: self.tokenize(x, stopwords=stopwords))

        if ithreshold is not None and uthreshold is not None:
            # filter
            tweet_counts = dataframe['item.tokenized'].value_counts()
            user_counts = dataframe['user'].value_counts()

            dataframe = dataframe[
                dataframe['item.tokenized'].isin(tweet_counts[tweet_counts > ithreshold].index)
                & dataframe['user'].isin(user_counts[user_counts > uthreshold].index)
            ]

        # obtain id of users and items
        dataframe['user.merge.id'] = dataframe.groupby(["user"]).ngroup()
        dataframe['item.merge.id'] = dataframe.groupby(["item.tokenized"]).ngroup()
        dataframe['edge.merge.id'] = dataframe.groupby(["user", "item.tokenized"]).ngroup()

        dataframe.reset_index(drop=True, inplace=True)

        print("[{}]: #Lines: {} #Users: {} #Items: {} #Edges: {}".format(
            self.name, len(dataframe), len(dataframe["user.merge.id"].unique()),
            len(dataframe["item.merge.id"].unique()), len(dataframe["edge.merge.id"].unique())
        ))

        return dataframe

    def merge(self, dm: DataManager, ithreshold=None, uthreshold=None) -> DataManager:
        dm.data = self.textProcess(dm.data, ithreshold, uthreshold)
        dm.mappings["user.merge.id"] = pd.Series(
            dm.data["user"].values, index=dm.data["user.merge.id"]
        ).to_dict()
        dm.mappings["item.merge.id"] = pd.Series(
            dm.data["item"].values, index=dm.data["item.merge.id"]
        ).to_dict()
        with open("mappings.json", "w") as fout:
            json.dump(dm.mappings, fout, indent=2)
        return dm

if __name__ == "__main__":
    from DataManagement import PandasDataManager
    from loader.russophobia import attach
    PDM = PandasDataManager()
    PDM = attach(PDM)
    print(PDM.data)
    node_merge = NodeMergeByTokenize()
    node_merge.merge(PDM, ithreshold=2, uthreshold=2)
    print(PDM.data)
    PDM.save_debug()