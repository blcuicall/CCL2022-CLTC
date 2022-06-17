# 赛道一（中文拼写检查）基线模型

## 1. 模型介绍

本赛道提供1个基于 BERT 的基线模型，使用非自回归方法，输出对错误句子的修改。

基线模型中使用的预训练模型是 bert-base-chinese

## 2. 实验结果本节列出基线模型在测试集上的结果。

| dataset      | D-A   | D-P   | D-R   | D-F   | C-A   | C-P   | C-R   | C-F   |
| ------------ | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| yaclc-track1 | 69.27 | 66.26 | 49.27 | 56.52 | 66.45 | 58.68 | 43.64 | 50.05 |

注：评价指标的 D 为 detection level，C 为 correction level，A 为 Accuracy，P 为 Precision，R 为 Recall，F为 F1 score
