# Single-Choice-Question-Learning

##### anyone can download the corpus.word2vec by the `pan link`: https://pan.baidu.com/s/1dJi-DzsjK9WBuKSuXKjhcQ, `password`:n7hr

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
