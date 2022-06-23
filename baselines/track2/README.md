# 赛道二（中文语法错误检测）基线模型

## 1. 模型介绍

本赛道提供 GECToR[^1] 模型作为基线模型，该模型使用非自回归方法，输出给定错误句子的修改（edits）。我们使用以下几种不同的模型：

- chinese-bert-wwm-ext
- chinese-roberta-wwm-ext-base
- chinese-electra-base-discriminator

## 2. 使用方法

### 2.1 环境配置

使用python3.6环境，通过

~~~~bash
pip install -r requirements.txt
~~~~

配置环境。

### 2.2 训练

分别使用lang8数据和往年公开的CGED进行两阶段训练

1. 使用预训练模型的tokenizer对语料进行分词
3. 使用预处理脚本将数据处理成gecotr需要的格式  
```
python utils/preprocess_data.py -s SOURCE -t TARGET -o OUTPUT_FILE
```
4. 使用lang8数据，随机划分5000句话作为开发集，其余做训练集，使用train_stage1.sh脚本训练模型
5. 使用2021年前的所有训练数据作为训练集，所有测试数据作为开发集（繁体字数据集需经过繁简转换），使用train_stage2.sh脚本继续训练 

### 2.3 推理

使用 predict.sh 进行推理

## 3. 实验结果

上述模型在 CGED2021 测试集上的F1结果如下表。

|         | FPR   | Detection | Identification | Position | Correction | Comprehensive |
| ------- | ----- | --------- | -------------- | -------- | ---------- | ------------- |
| BERT    | 31.93 | 74.45     | 46.34          | 27.53    | 14.63      | 32.83         |
| RoBERTa | 30.24 | 74.26     | 46.83          | 27.82    | 15.25      | 33.48         |
| ELECTRA | 29.54 | 73.08     | 45.71          | 27.64    | 14.03      | 32.73         |

表中的 FPR，Detection，Identification，Position，Correction 分数可以使用 [metrics/track2](https://github.com/blcuicall/CCL2022-CLTC/tree/main/metrics/track2) 计算。Comprehensive 为加权平均后的综合得分，计算公式为：

Comprehensive = 0.25 * (Detecion + Identification + Position + Correction - FPR)

[^1]:Kostiantyn Omelianchuk, Vitaliy Atrasevych, Artem Chernodub, Oleksandr Skurzhanskyi.  GECToR – Grammatical Error Correction: Tag, Not Rewrite. BEA 2020.↩