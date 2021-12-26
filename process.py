import re
from collections import defaultdict
import jieba.posseg
import numpy as np
import jieba.posseg
import os
import pickle
from gensim import corpora, models, similarities

def load_embedding(filename):
    #返回一个list和一个dict{list}，其中list代表corpus.word2vec中所有的词语，dict代表word_id->vec的映射关系
    embeddings = []
    word2idx = defaultdict(list)
    with open(filename, mode="r", encoding="utf-8") as rf:
        for line in rf:
            arr = line.split(" ")
            embedding = [float(val) for val in arr[1: -1]]
            word2idx[arr[0]] = len(word2idx)
            embeddings.append(embedding)

    return embeddings, word2idx


def count_id(words_list, word2idx, max_len):
    #实现vec到num的映射
    unknown = word2idx.get("UNKNOWN", 0)
    num = word2idx.get("NUM", len(word2idx))
    index = [unknown] * max_len
    i = 0
    for word in words_list:
        if word in word2idx:
            index[i] = word2idx[word]
        else:
            if re.match("\d+", word):
                index[i] = num
            else:
                index[i] = unknown
        if i >= max_len - 1:
            break
        i += 1
    return index


def load_data(knowledge_file, filename, dt, stop_words, sim_ixs, max_len):
    # 每6行数据为1轮的(每个数据对于问题)
    knowledge_texts = rmv_infreq_words(knowledge_file, stop_words)
    train_texts = rmv_infreq_words(filename, stop_words)
    NN = 0
    tmp = []
    questions, answers, labels = [], [], []
    with open(filename, mode="r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 6 == 0:
                NN += 1
                for j in sim_ixs[i//6]:
                    tmp.extend(knowledge_texts[j])
                tmp.extend(train_texts[i])
            elif i % 6 == 1:
                tmp.extend(train_texts[i])
                t = count_id(tmp, dt, max_len)
            else:
                if line[0] == 'R':
                    questions.append(t)
                    answers.append(count_id(train_texts[i], dt, max_len))
                    labels.append(1)
                elif line[0] == 'W':
                    questions.append(t)
                    answers.append(count_id(train_texts[i], dt, max_len))
                    labels.append(0)
                if i % 6 == 5:
                    tmp.clear()
    return questions, answers, labels, NN


def prepare_data(bg, file_name, stop_words):
    #对数据进行预处理，主函数中表现为对于训练集的处理
    bg = rmv_infreq_words(bg, stop_words)
    train_texts = rmv_infreq_words(file_name, stop_words)
    codt = corpora.Dictionary(bg + train_texts)
    codt.save(os.path.join('tmp/dictionary.dict'))# 存入知识和训练数据
    corpus = [codt.doc2bow(text) for text in bg]
    corpora.MmCorpus.serialize('tmp/knowledge_corpus.mm', corpus)# 存入知识库的语料


def cal_scores(file_name, stop_words, k):
    #计算相似性最高的几个知识，作为候选知识参与最终决策
    sim_path = "tmp/" + file_name[5:-4]
    if os.path.exists(sim_path):
        with open(sim_path, "rb") as f:
            sim_ixs = pickle.load(f)
        return sim_ixs
    codt = corpora.Dictionary.load("tmp/dictionary.dict")
    corpus = corpora.MmCorpus("tmp/knowledge_corpus.mm")
    lsi = models.LsiModel(corpus, id2word=codt, num_topics=10)  # initialize an LSI transformation
    #计算相似性矩阵
    index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it
    sim_ixs = [] #用于存放最后相似性最高的知识
    with open(file_name,encoding="utf-8") as f:
        tmp = []
        for i, line in enumerate(f): #对于问题以6个句子为单位进行处理
            if i % 6 == 0:
                tmp.extend([token for token, _ in jieba.posseg.cut(line.rstrip()) if token not in stop_words])
            if i % 6 == 1:
                tmp.extend([token for token, _ in jieba.posseg.cut(line.rstrip()) if token not in stop_words])
                vec_lsi = lsi[codt.doc2bow(tmp)]  # convert the query to LSI space
                sim_ix = index[vec_lsi]  # perform a similarity query against the corpus
                sim_ix = [i for i, j in sorted(enumerate(sim_ix), key=lambda item: -item[1])[:k]]  # topk index
                sim_ixs.append(sim_ix)
                tmp.clear()
    with open(sim_path, "wb") as f:
        pickle.dump(sim_ixs, f)
    return sim_ixs

def rmv_infreq_words(filename, stop_words):
    #对于出现stop_words的问题进行出去，同时对于问题中不同的词语进行计数，如果出现频率为1，也除去
    texts = []
    with open(filename, 'r',encoding="utf-8") as f:
        for line in f.readlines():
            texts.append([token for token, _ in jieba.posseg.cut(line.rstrip())
                          if token not in stop_words])
    frequency = defaultdict(int)#对出现词语频率进行记数，结果存入名为frequency的字典
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1] for text in texts]#如果频率为1，则忽略这个数
    return texts


def pre_epoch(Q, T, category, N, length):
    #对问题进行预处理，每隔4句为一个小轮进行处理数据
    batch_num = int(N / length) + 1
    for batch in range(batch_num):
        ret_questions, true_answers, false_answers = [], [], []
        for i in range(batch * length, min((batch + 1) * length, N)):
            ix = i * 4
            ret_questions.extend([Q[ix]] * 3)
            for j in range(ix, ix + 4):
                if category[j]:
                    true_answers.extend([T[j]] * 3)
                else:
                    false_answers.append(T[j])
        yield np.array(ret_questions), np.array(true_answers), np.array(false_answers)


def lat_epoch(Q, T, N, batch_size):
    #对预测结果进行打标
    batch_num = int(N / batch_size) + 1
    Q, T = np.array(Q), np.array(T)
    for batch in range(batch_num):
        start_ix = batch * batch_size * 4
        end_ix = min((batch + 1) * batch_size * 4, len(Q))
        yield Q[start_ix:end_ix], T[start_ix:end_ix]
