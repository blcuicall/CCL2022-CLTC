# 赛道三（多维度汉语学习者文本纠错）数据集

本页面主要内容为：

- [1. 数据集下载](#1-数据集下载)

- [2. 结果提交格式](#2-结果提交格式)

## 1. 数据集下载

### 1.1 训练集

本赛道对 NLPCC2018-GEC[^1] 发布的采集自 Lang8 平台的中介语数据进行了处理，供参赛者使用。

> 下载地址：http://yunpan.blcu.edu.cn:80/link/EDBB933F1FCD49C054F9AB7F65B0A746
>
> 有效期限：2022-10-31 23:59
>
> 访问密码：eSPB

该数据格式为：

```
[原始句子]	[修改句子]
```

其中原始句子与修改句子一一对应，中间使用 "\t" 分隔。

**注意：参赛者仅允许使用上述数据进行训练。数据增强方法仅允许基于上述数据进行。**

### 1.2 开发集

本赛道基于 YACLC 数据集[^2]，提供 minimal 和 fluency 两个维度上的开发集，并各提供 `.para` 和 `.m2` 两种格式的文件。

`.para` 文件格式为：

```
[id]	[原始句子]	[修改句子1]	[修改句子2]
```

一个原始句子对应多条修改句子，每列之间使用 "\t" 分隔。

`.m2` 文件为每个修改句子中修改的部分，用于模型评估，详见[baselines/track3](https://github.com/blcuicall/CCL2022-CLTC/baselines/track3)。

### 1.3 测试集

本赛道的评测分两阶段进行，两阶段的测试数据分别位于 `testA` 和 `testB` 文件夹。

每个阶段提供 minimal 和 fluency 维度上的原始句子，文件格式为：

```
[id]	[原始句子]
```

句子 id 和原始句子之间使用 "\t" 分隔。

## 2. 结果提交格式

参赛者需要提供给定原始句子对应的修改后句子，文件格式应为：

```
[id]	[原始句子]	[修改句子]
```

每列之间使用 "\t" 分隔。

提交前，文件需依规范正确命名，并压缩成 `.zip` 格式文件的压缩包。

阶段一提交结果命名：

```
track3_testA.zip	#压缩包名字
    ├── yaclc-minimal_testA.para	# minimal 维度结果
    └── yaclc-fluency_testA.para	# fluency 维度结果
```

阶段二提交结果命名：

```
track3_testB.zip	#压缩包名字
    ├── yaclc-minimal_testB.para	# minimal 维度结果
    └── yaclc-fluency_testB.para	# fluency 维度结果
```



[^1]:Yuanyuan Zhao, Nan Jiang, Weiwei Sun, and Xiaojun Wan. 2018. Overview of the nlpcc 2018 shared task: Grammatical error correction. In CCF International Conference on Natural Language Processing and Chinese Computing (NLPCC), pages 439–445. ([pdf](http://tcci.ccf.org.cn/conference/2018/papers/EV11.pdf)) 
[^2]:Yingying Wang, Cunliang Kong, Liner Yang, Yijun Wang, Xiaorong Lu, Renfen Hu, Shan He, Zhenghao Liu, Yun Chen, Erhong Yang, and Maosong Sun. 2021. YACLC: A Chinese Learner Corpus with Multidimensional Annotation. arXiv preprint arXiv:2112.15043. ([pdf](https://arxiv.org/abs/2112.15043)) 