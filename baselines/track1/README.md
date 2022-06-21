# 赛道一（中文拼写检查）基线模型

## 1. 模型介绍

本赛道提供1个基于 BERT 的基线模型，使用非自回归方法，输出对错误句子的修改。

基线模型中使用的预训练模型是 bert-base-chinese

## 2. 环境配置

1. 安装pytorch
   
   ```
   conda create -n csc python=3.7
   conda activate csc
   conda install pytorch=1.7.1 cudatoolkit -c pytorch
   ```

2. 安装其他依赖
   
   ```bash
   python 3.7.13
   transformers 4.19.1
   ```

## 3. 模型训练

- 数据预处理+模型训练，编辑 pipeline.sh，运行
  
  ```bash
  bash pipeline.sh
  ```

## 4. 模型预测

- 编辑 decode.sh，运行
  
  ```bash
  bash decode.sh
  ```

## 5. 实验结果

本节列出基线模型在测试集上的结果。

Sentence Level 

| dataset      | D-A   | D-P   | D-R   | D-F   | C-A   | C-P   | C-R   | C-F   |
| ------------ | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| yaclc-track1 | 69.27 | 66.26 | 49.27 | 56.52 | 66.45 | 58.68 | 43.64 | 50.05 |

Character Level

| dataset      | D-A   | D-P   | D-R   | D-F   | C-A   | C-P   | C-R   | C-F   |
| ------------ | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| yaclc-track1 | 69.27 | 82.12 | 49.27 | 61.59 | 66.45 | 80.27 | 43.64 | 56.54 |

注：评价指标的 D 为 detection level，C 为 correction level，A 为 Accuracy，P 为 Precision，R 为 Recall，F为 F1 score
