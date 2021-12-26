import os
import codecs
import time
import numpy as np
import tensorflow as tf
import process
from model import Cell
from settings import *

embedding, word2idx = process.load_embedding("models/corpus.word2vec")#加载embeddings
stop_words = codecs.open("models/unused.txt", 'r', encoding='utf8').readlines()
for i in range(0,len(stop_words)):
    stop_words[i]=stop_words[i].strip()
process.prepare_data("models/database.txt", "data/train.txt", stop_words)
train_sim_ixs = process.cal_scores("data/train.txt", stop_words, sim_num)#计算最高相似性的几个知识
test_sim_ixs = process.cal_scores("data/test.txt", stop_words, sim_num)
train_questions, train_answers, train_labels, train_question_num = process.load_data("models/database.txt", "data/train.txt", word2idx, stop_words, train_sim_ixs, max_len)

test_questions, test_answers, test_labels, test_question_num = process.load_data("models/database.txt", "data/test.txt", word2idx, stop_words, test_sim_ixs, max_len)


questions, true_answers, false_answers = [], [], []#将数据按照epoch进行划分
for q, ta, fa in process.pre_epoch(
        train_questions, train_answers, train_labels, train_question_num, length):questions.append(q), true_answers.append(ta), false_answers.append(fa)

def cal_params():
    print("evaluating..")
    scores = []
    for test_q, test_a in process.lat_epoch(test_questions, test_answers, test_question_num, length):
        test_feed_dict = {
            lstm.Test_Q: test_q,
            lstm.Test_A: test_a,
            lstm.dropout_keep_prob: 1.0
        }
        _, score = sess.run([globalStep, lstm.result], test_feed_dict)
        scores.extend(score)
    cnt = 0
    scores = np.absolute(scores)
    for test_id in range(test_question_num):
        offset = test_id * 4
        predict_true_ix = np.argmax(scores[offset:offset + 4])
        if test_labels[offset + predict_true_ix] == 1:
            cnt += 1
    print("evaluation acc: ", cnt / test_question_num)

    scores = []
    for train_q, train_a in process.lat_epoch(train_questions, train_answers, train_question_num, length):
        test_feed_dict = {
            lstm.Test_Q: train_q,
            lstm.Test_A: train_a,
            lstm.dropout_keep_prob: 1.0
        }
        _, score = sess.run([globalStep, lstm.result], test_feed_dict)
        scores.extend(score)
    cnt = 0
    scores = np.absolute(scores)
    for train_id in range(train_question_num):
        offset = train_id * 4
        predict_true_ix = np.argmax(scores[offset:offset + 4])
        if train_labels[offset + predict_true_ix] == 1:
            cnt += 1
    print("evaluation acc(train): ", cnt / train_question_num)

def update_params():
    sess.run(tf.compat.v1.global_variables_initializer())
    lr = learning_rate
    for i in range(shr_rate):
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(lr)
        optimizer.apply_gradients(zip(grads, tvars))
        trainOp = optimizer.apply_gradients(zip(grads, tvars), global_step=globalStep)
        for epoch in range(num_epochs):
            for question, trueAnswer, falseAnswer in zip(questions, true_answers, false_answers):
                feed_dict = {
                    lstm.Train_Q: question,
                    lstm.Train_A: trueAnswer,
                    lstm.Train_F: falseAnswer,
                    lstm.dropout_keep_prob: sigma,
                }
                _, step, _, _, loss, summary = sess.run([trainOp, globalStep, lstm.trueCosSim, lstm.falseCosSim, lstm.loss, summary_op], feed_dict)
                print("step:", step, "loss:", loss)
                summary_writer.add_summary(summary, step)
                if step % train_period == 0:
                    cal_params()
            saver.save(sess, "res/savedModel")
        lr *= reg_factor

with tf.Graph().as_default(), tf.device("/gpu:0"):
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
    session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)
    with tf.compat.v1.Session(config=session_conf).as_default() as sess:
        globalStep = tf.compat.v1.Variable(0, name="global_step", trainable=False)
        lstm = Cell(length, max_len, embedding, embedding_dim, rnn_size, upper_bound)
        tvars = tf.compat.v1.trainable_variables()
        grads, _ = tf.compat.v1.clip_by_global_norm(tf.gradients(lstm.loss, tvars), max_grad_norm)
        saver = tf.compat.v1.train.Saver()
        # 计算时间
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))
        #记录参数
        loss_summary = tf.compat.v1.summary.scalar("loss", lstm.loss)
        summary_op = tf.compat.v1.summary.merge([loss_summary])
        summary_dir = os.path.join(out_dir, "summary", "train")
        summary_writer = tf.compat.v1.summary.FileWriter(summary_dir, sess.graph)
        #根据之前参数更新参数
        update_params()
        # 计算参数
        cal_params()
