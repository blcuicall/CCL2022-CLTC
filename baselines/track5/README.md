# 赛道五（语法纠错质量评估）基线模型

## 1. 模型介绍

本赛道提供基于 seq2seq 预训练语言模型的基线模型，直接输出改正句的质量评估分数，并根据分数进行重排序，从而进一步获得最高质量评估分数的原句id、改正句id、改正句得分。
本赛道提供两种不同类型的质量评估方式的基线模型，分别为bert_gqe、bert_qe。   
     
两个基线模型使用的具体模型为：   
- bert-base-chinese

其中，bert_gqe使用softmax<sub>ys=1</sub>(W · H<sup>k</sup><sub>0</sub>)表示改正句的质量评估分数；bert_qe使用 sigmoid(W · H<sup>k</sup><sub>0</sub> )表示改正句的质量评估分数[^1]。   
（W:参数； ys标签分为两种：正确的（ys=1）、错误的（ys=0）；H<sup>k</sup><sub>0</sub>：用改正句的[CLS]表示）


### 2. 实验结果

本节列出上述模型在开发集集（development set）上的结果。

-  minimal 维度上的结果

|          | Precision | Recall |  F0.5  |   Pcc   |
| -------- | --------- | ------ |  ----- |   ---   |
| BERT-GQE |    0.5577 | 0.2933 | 0.4725 | -0.0630 |
| BERT-QE  |    0.5649 | 0.4675 | 0.5423 | 0.2016  |


- fluency 维度上的结果

|          | Precision | Recall |  F0.5  |   Pcc    |
| -------- | --------- | ------ |  ----- |   ---    |
| BERT-GQE |    0.2990 | 0.0975 | 0.2115 | -0.1025  |
| BERT-QE  |    0.3085 | 0.1904 | 0.2744 | 0.1469   |


[^1]:Zhenghao Liu, Xiaoyuan Yi, Maosong Sun, Liner Yang, and Tat-Seng Chua. 2021. Neural quality estimation with multiple hypotheses for grammatical error correction. In Proceedings of NAACL-HLT, pages 5441–5452. 

