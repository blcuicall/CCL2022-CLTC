# 赛道三（多维度汉语学习者文本纠错）评价指标

本赛道使用的评价指标参考自 [ChERRANT](https://github.com/HillZhang1999/MuCGEC/tree/main/scorers/ChERRANT)，可以计算预测结果的精确度、召回度、F值指标，从而评估语法纠错模型的性能。

## 1. 环境配置

```shell
pip install -r requirements
```

## 2. 使用方式

```shell
python parallel_eval.py -f [hyp_file] -r [ref_file]
```

其中，`[hyp_file]` 为 `.para` 格式文件，文件格式参见[结果提交格式](https://github.com/blcuicall/CCL2022-CLTC/blob/main/datasets/track3/README.md#2-%E7%BB%93%E6%9E%9C%E6%8F%90%E4%BA%A4%E6%A0%BC%E5%BC%8F)；`[ref_file]` 为 `.m2` 格式的参考答案文件。
