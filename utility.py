import json
import csv
import re
from collections import Counter
from functools import cached_property

import MeCab
import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import word2vec, KeyedVectors, LdaModel
from tqdm import tqdm
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from lexrank import LexRank

tagger = MeCab.Tagger()
# tagger = MeCab.Tagger('-Owakati')#タグはMeCab.Tagger（neologd辞書）を使用
# tagger.parse('')

nlp = spacy.load('ja_ginza')

p = re.compile(r"<[^>]*?>")


class News:
  def __init__(self):
    self.df = None
    self.load_items()
    # self.load_dataset()

  def load_items(self):
    self.df = pd.read_csv('news/nikkei.csv')
    self.df = self.df.drop(self.df.columns[[0]], axis=1)
    self.df = self.df.drop(self.df.columns[[0]], axis=1)
    self.df = self.df.rename(columns={'artical_id':'id','artical_title':'title'})
    self.df = self.df.dropna(how='any').dropna(how='any', axis=1)
    self.df["words"] = self.df.body.apply(wakati, to_lower=True, to_zero=True, ignore_symbols=True)
    self.save_datasets()

  def save_datasets(self):
    self.df.to_csv("news/nikkei_datasets.csv")

  def load_dataset(self):
    self.df = pd.read_csv("news/nikkei_datasets.csv")

  @cached_property
  def create_corpus(self):
    documents = []
    for i, item in tqdm(self.df.iterrows()):
      # documents.append(TaggedDocument(item["words"].split(), [item["id"]]))
      try:
        documents.append(TaggedDocument(item["words"].split(), [item["id"]]))
      except:
        print(item["id"])
        continue
    np.savetxt("news/nikkei_corpus.txt", documents, fmt = '%s', delimiter = ',', encoding='utf8')
    return documents

  @cached_property
  def create_doc(self):
    documents = []
    for i, item in tqdm(self.df.iterrows()):
      documents.append(item['body'])
    np.savetxt("news/nikkei_documents.txt", documents, fmt='%s', delimiter=',', encoding='utf8')
    return documents

  @cached_property
  def create_doc2vec_model(self):
    documents = self.create_corpus
    model = Doc2Vec(documents, vector_size=200, min_count=0, epochs=10)
    model.save("news/nikkei_doc2vec_model.model")
    return model

  @cached_property
  def create_lda_model(self):
    corpus = self.create_corpus
    # LdaModelモデルをトレーニングする
    model = LdaModel(corpus, num_topics=5)
    model.save("news/nikkei_lda_mode.model")
    return model

  @cached_property
  def create_lexrank_model(self):
    documents = self.create_doc
    model = LexRank(documents)
    return model

  def sim_page(self, ID):
    # wv = self.create_model
    model = self.create_doc2vec_model
    return pd.concat([self.df[self.df.id == k] for k, v in model.dv.most_similar(ID)])

  def sim_word(self, word):
    # wv = self.create_model
    model = self.create_doc2vec_model
    results = model.wv.most_similar(positive=[word])
    for result in results:
        print(result)

  def lex(self, sentences):
    lxr = self.create_lexrank_model
    return lxr.get_summary(sentences, summary_size=2, threshold=.1)

  def word2id_001(self, word):
    # gensimを使用して、コーパス内の記事をベクトル化します
    # ここでは、Doc2Vecモデルを使用しています
    model = self.create_doc2vec_model

    # 入力した単語をモデルにかけます
    vector = model.infer_vector([word])

    # 入力した単語に類似した記事を探すために、
    # コーパス内の各記事との入力単語のベクトルとの類似度を計算します
    similarities = model.docvecs.most_similar([vector])

    # 類似度が高い順に記事を出力します
    return pd.concat([self.df[self.df.id == k] for k, v in similarities])

  def categories(self, id):
    # model = self.create_model

    # news = self.df[self.df.id == id].body
    # category = model.predict_output_word(news)

    # max_prob = 0
    # max_label = None
    # for label, prob in category:
    #   if prob > max_prob:
    #     max_prob = prob
    #     max_label = label
    model = self.create_doc2vec_model
    tags = model.infer_vector(self.df.words)
    return TaggedDocument(words=self.df[self.df.id == id].body, tags=tags)


def cos_sim(v1, v2):
  return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def delete_brackets(s):
    """
    括弧と括弧内文字列を削除
    """
    """ brackets to zenkaku """
    table = {
        "(": "（",
        ")": "）",
        "<": "＜",
        ">": "＞",
        "{": "｛",
        "}": "｝",
        "[": "［",
        "]": "］"
    }
    for key in table.keys():
        s = s.replace(key, table[key])
    """ delete zenkaku_brackets """
    l = ['（[^（|^）]*）', '【[^【|^】]*】', '＜[^＜|^＞]*＞', '［[^［|^］]*］',
         '「[^「|^」]*」', '｛[^｛|^｝]*｝', '〔[^〔|^〕]*〕', '〈[^〈|^〉]*〉']
    for l_ in l:
        s = re.sub(l_, "", s)
    """ recursive processing """
    return delete_brackets(s) if sum([1 if re.search(l_, s) else 0 for l_ in l]) > 0 else s

def wakati(text, to_lower=False, to_zero=False, noun_only=False, ignore_symbols=False):
    if to_lower:
        text = text.lower()
    if to_zero:
        text = re.sub(r'[0-9０-９]+', "0", text)
    devided_text = text.splitlines()
    text = ' '.join(devided_text)
    text = re.sub(r"\s", " ", text)
    text = delete_brackets(text)
    result = tagger.parse(text).split("\n")
    wakati = []
    for r in result:
        if "\t" not in r:
            continue
        if "\t" in r:
            content, *ents = r.split("\t")
        if noun_only:
            if ents[0] not in ["名詞"]:
                continue
        if ignore_symbols:
            if ents[0] in ["記号"]:
                continue
            if ents[0] in ["名詞"]:
                if ents[1] in ["サ変接続"]:
                    if ents[6] in ["*"]:
                        continue
        wakati.append(content)
    return " ".join(wakati)

def unwiki_wakati(txt):
  soup = BeautifulSoup(txt)
  text = soup.get_text()
  lines = [line.strip() for line in text.splitlines()]
  # タグ削除
  s =  "\n".join(line for line in lines if line)
  # s1 = s[:(len(s)//7)*3]
  # s2 = s[(len(s)//7)*3:]
  # return s1
  return s
