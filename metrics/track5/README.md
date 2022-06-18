# 赛道五（语法纠错质量评估）评价指标

本赛道使用的评价指标: 
1. 预测生成的修改句子的质量评估分数和真实F0.5之间的皮尔逊相关系数（Pearson Correlation Coefficient，PCC)，最后根据全部评测样例求取平均值用以衡量语法纠错质量评估分数与真实 F0.5 分数之间的相关性。
2. 根据生成的质量评估分数对修改句子进行重排序，获取top-1的修改句子进行参考自 [ChERRANT](https://github.com/HillZhang1999/MuCGEC/tree/main/scorers/ChERRANT)，可以计算预测结果的精确度、召回度、F值指标，从而评估语法纠错模型的性能(同track3)。

## 1. 环境配置

```shell
pip install -r requirements
```

## 2. 使用方式

### 2.1 评估pcc分数并获得top-1改正句
```shell
Python pcc_hyps_eval.py -t [test_f05_file]  -c [hyp_file] -b [basline_results_file]
```
其中： 
`[test_f05_file]`为`.json`格式带有真实F0.5的参考答案文件；   
`[hyp_file]`为重排序得到的top-1结果输出文件，用于第二个评价指标，文件格式参考track3 [结果提交格式](https://github.com/blcuicall/CCL2022-CLTC/blob/main/datasets/track3/README.md#2-%E7%BB%93%E6%9E%9C%E6%8F%90%E4%BA%A4%E6%A0%BC%E5%BC%8F)；   
`[baseline_results_path]`为`.para`格式文件，文件格式参见[结果提交格式](https://github.com/blcuicall/CCL2022-CLTC/blob/main/datasets/track5/README.md#2-%E7%BB%93%E6%9E%9C%E6%8F%90%E4%BA%A4%E6%A0%BC%E5%BC%8F)。


### 2.2 对top-1改正句进行评估（同track3）
```shell
python parallel_eval.py -f [hyp_file] -r [ref_file]
```
**注意：parallel_eval.py 评测脚本来自[metrics/track3](https://github.com/blcuicall/CCL2022-CLTC/tree/main/metrics/track3)。**

其中： 
`[hyp_file]` 为 `.para` 格式文件，与 2.1 中的`[hyp_file]`为同一个文件；    
`[ref_file]` 为 `.m2` 格式的参考答案文件。

