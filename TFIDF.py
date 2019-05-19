import os
import pickle as pkl
import math
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import gc

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

from matplotlib import pyplot

article_dir = "./articles"
path_list = [os.path.join(article_dir, x) for x in os.listdir(article_dir)]

corpus = []
article_ids = []
for p in path_list:
    with open(p, 'rb') as f:
        txt = pkl.load(f)
        if len(txt)>5000:
            corpus.append(txt)
            article_ids.append(p.split('/')[2].split('.')[0])

num_docs = len(corpus)
v = CountVectorizer(ngram_range=(1,2), binary=True, stop_words='english',min_df=1)
vf = v.fit_transform(corpus)
terms = v.get_feature_names()
freqs = np.asarray(vf.sum(axis=0).ravel())[0]
idfs = [math.log10(num_docs/f) for f in freqs]
custom_weight1 = np.array([1+math.log10(f) for f in freqs])

keywords1 = []
interval = 50
cur_index = 0
while cur_index<len(corpus):
    v = CountVectorizer(ngram_range=(1,2), stop_words='english', vocabulary=terms, min_df=1)
    vf = v.fit_transform(corpus[cur_index:cur_index+interval])
    a = vf.toarray()

    for article in a:
        x = np.copy((np.log10(article+1)*idfs*custom_weight1).argsort()[-20:][::-1])
        keywords1.append(x)

    del v
    del vf
    del a
    gc.collect()
    cur_index += interval

x1 = np.array(keywords1).flatten()
unique_x1 = set(x1)

dummy = x1.copy().tolist()
for i in unique_x1:
    dummy.remove(i)

occ = Counter(dummy)
for wid in list(occ.keys()):
    if occ[wid] < 15:
        del occ[wid]

survived = {}
for word in list(occ.keys()):
    for i, k in enumerate(keywords1):
        if word in k:
            if article_ids[i] not in survived:
                survived[article_ids[i]] = [word]
            else:
                survived[article_ids[i]].append(word)

survived_terms = {}
print(len(occ))
for wid in list(occ.keys()):
    survived_terms[wid] = terms[wid]

with open("keyword_index.pkl", "rb") as f:
    last_st = pkl.load(f)

keyword_space = {}

x = max(survived_terms.keys())
y = max(last_st.keys())
max_index = max([x,y])
inv_curkeyword = {v:k for k,v in survived_terms.items()}

for k,v in last_st.items():
    keyword_space[k] = v
    if k in survived_terms:
        move_count += 1
        moving = survived_terms[k]
        inv_curkeyword[moving] = max_index+1
        max_index += 1
    oidx = inv_curkeyword[v]
    for ak,av in survived.items():
        if k in av:
            survived[ak] = [k if x==oidx else x for x in survived[ak]]
    del inv_curkeyword[v]

for k,v in inv_curkeyword.items():
    keyword_space[v] = k

with open("keyword_index.pkl", "wb") as f:
    pkl.dump(keyword_space,f)
with open("article_keyword_matrix.pkl",'wb') as f:
    pkl.dump(survived,f)
