# 赛道三（多维度汉语学习者文本纠错）基线模型

## 1. 模型介绍

本赛道提供 [seq2edit](seq2edit) 和 [seq2seq](seq2seq) 两种类型的基线模型。seq2edit 方式基于 GECtoR[^1] 模型，使用非自回归方法，输出给定错误句子的修改（edits）。seq2seq 方式则使用 encoder-decoder 模型，直接输出改正后的句子。具体来讲，我们使用以下几种不同的模型：

- seq2edit
  - chinese-bert-wwm-ext
- seq2seq
  - bart-base-chinese



## 2. 实验结果

本节列出上述模型在开发集（development set）上的结果。

-  minimal 维度上的结果

|      | Precision | Recall | F0.5  |
| ---- | --------- | ------ | ----- |
| BERT | 61.12     | 37.06  | 54.10 |
| BART | 63.34     | 37.58  | 55.70 |

- fluency 维度上的结果

|      | Precision | Recall | F0.5  |
| ---- | --------- | ------ | ----- |
| BERT | 33.97     | 12.89  | 25.59 |
| BART | 34.33     | 12.54  | 25.47 |



[^1]:Kostiantyn Omelianchuk, Vitaliy Atrasevych, Artem Chernodub, Oleksandr Skurzhanskyi.  GECToR – Grammatical Error Correction: Tag, Not Rewrite. BEA 2020.↩
