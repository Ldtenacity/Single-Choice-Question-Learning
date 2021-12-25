# Single-Choice-Question-Learning

##### anyone can download the corpus.word2vec by the `pan link`: https://pan.baidu.com/s/1dJi-DzsjK9WBuKSuXKjhcQ, `password`:n7hr

### 代码结构
<img src="image.assets/structure.png" alt="structure" style="zoom:67%;" />

### 数据来源
##### 题库来源: http://igeocn.com/igeocn/tiku/tk-1st/igeocn-qa-1.html
##### 背景知识来源: https://www.liuxue86.com/gaokao/dilizhishidian/


### 参数设置

| **k**         | **rnn_size**   | **max_grad_norm** |
|---------- | -------------- | --------------- |
| 5       | 100           | 5              |
| **embedding_dim** | **max_sentence_length** | **lr** |
| 50  | 100             | 0.4               |
| **epoches_num** | **checkpoints_num** | **batch_size** |
| 20  | 20             | 256               |

### 训练中
##### evaluation accuracy of `test set` = 0.363
##### evaluation accuracy of `training set` = 0.437
<img src="image.assets/running.png" alt="running" style="zoom:67%;" />


### 训练结果
##### evaluation accuracy of `test set` = 0.441
##### evaluation accuracy of `training set` = 0.766
<img src="image.assets/res.png" alt="res" style="zoom:67%;" />

### 结果分析
##### 由于背景知识标注较少，所以最终训练结果较低(0.44)。代码模拟了一个学生学习中学地理课本知识点去做地理竞赛题的过程，最终得分为44分(百分制)，高于随机选择25分(百分制)，但结果也不算很高，究其原因，我分析后认为主要有2个：
###### 1. 地理竞赛题中有一些中学课本中没有涉及到的知识点
###### 2. 课本中知识体系用BiLstm简单建模效率较低。

### Reference
##### 参考学习了一些对知识库进行NLP建模的模型
