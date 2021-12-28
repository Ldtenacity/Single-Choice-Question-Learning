# Single-Choice-Question-Learning

##### anyone can download the corpus.word2vec by the `pan link`: https://pan.baidu.com/s/1dJi-DzsjK9WBuKSuXKjhcQ, `password`:n7hr

网络的搭建使用到了 BAMnet 的思想，这个思想来自于 《Bidirectional Attentive Memory Networks for Question Answering over Knowledge Base》
<img src="image.assets/BAMnet.png" alt="BAMnet" style="zoom:67%;" />
### 在知识的查询过程中同时考虑了背景知识和知识库的知识，在一定程度上解决了 incomplete KB 问题，这个思想参考了《improving QA over incomplete KBs with Knowledge-Aware Reader》
<img src="image.assets/SubGraphReader_KnowledgeAwareReader.png" alt="Readers" style="zoom:67%;" />
### 代码结构
<img src="image.assets/structure.png" alt="structure" style="zoom:50%;" />

### 数据来源
##### 题库来源: http://igeocn.com/igeocn/tiku/tk-1st/igeocn-qa-1.html (详情见`data/geo-multi-question.txt`)
##### 背景知识来源: https://www.liuxue86.com/gaokao/dilizhishidian/ (详情见`model/database_org.txt`)


### 参数设置

| **sim_num**         | **rnn_size**   | **sigma** |
|---------- | -------------- | --------------- |
| 5       | 100           | 0.45              |
| **embedding_dim** | **max_len** | **lr** |
| 50  | 100             | 0.4               |
| **reg_factor** | **train_period** | **batch_size** |
| 0.5  | 50             | 256               |

### 训练中
##### 按照 **7:3** 比例将题库 `geo-multi-question.txt` 划分为 `test set` 和 `training set`
##### 运行 `process_db.py` 将 `database_org.txt` 进行预处理，输入到 `database.txt`
##### 训练轮数 = 150
##### evaluation accuracy of `test set` = 0.363
##### evaluation accuracy of `training set` = 0.437
<img src="image.assets/running.png" alt="running" style="zoom:67%;" />


### 训练结果
##### 训练轮数 = 3199
##### evaluation accuracy of `test set` = 0.441
##### evaluation accuracy of `training set` = 0.766
<img src="image.assets/res.png" alt="res" style="zoom:67%;" />

### 结果分析
##### 由于背景知识标注较少，所以最终训练结果较低(0.44)。代码模拟了一个学生学习中学地理课本知识点去做地理竞赛题的过程，最终得分为44分(百分制)，高于随机选择25分(百分制)，但结果也不算很高，究其原因，我分析后认为主要有2个：
###### 1. 地理竞赛题中有一些中学课本中没有涉及到的知识点
###### 2. 课本中知识体系用BiLstm简单建模效率较低。

### Reference
##### 参考学习了一些对知识库进行NLP建模的模型
