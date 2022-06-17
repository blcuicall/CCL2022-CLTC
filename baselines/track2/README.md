# 赛道二（中文语法错误检测）基线模型

## 1. 模型介绍

本赛道提供 seq2edit 作为基线模型。seq2edit 方式基于 GECToR[^1] 模型，使用非自回归方法，输出给定错误句子的修改（edits）。我们使用以下几种不同的模型：

- seq2edit
  - chinese-bert-wwm-ext
  - chinese-roberta-wwm-ext-base
  - chinese-electra-base-discriminator

## 2. 使用方法

### 环境配置

使用python3.6环境，通过

~~~~bash
pip install -r requirements.txt
~~~~

配置环境。

### 训练

分别使用lang8数据和往年公开的CGED进行两阶段训练

1. 使用预训练模型的tokenizer对语料进行分词
3. 使用预处理脚本将数据处理成gecotr需要的格式  
```
python utils/preprocess_data.py -s SOURCE -t TARGET -o OUTPUT_FILE
```
4. 使用lang8数据，train_stage1.sh脚本训练模型
5. 使用合并的CGED数据，train_stage2.sh脚本继续训练 

### 推理

使用predict.sh进行推理

## 3. 实验结果

本节列出上述模型在2021CGED测试集上的F1结果。

|         | FPR   | Decection | Identification | Position | Correction |
| ------- | ----- | --------- | -------------- | -------- | ---------- |
| BERT    | 31.93 | 74.45     | 46.34          | 27.53    | 14.63      |
| RoBERTa | 30.24 | 74.26     | 46.83          | 27.82    | 15.25      |
| ELECTRA | 29.54 | 73.08     | 45.71          | 27.64    | 14.03      |

[^1]:Kostiantyn Omelianchuk, Vitaliy Atrasevych, Artem Chernodub, Oleksandr Skurzhanskyi.  GECToR – Grammatical Error Correction: Tag, Not Rewrite. BEA 2020.↩