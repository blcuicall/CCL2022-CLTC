# 赛道一（中文拼写检查）数据集

本页面主要内容为：

- [1. 数据集下载](#1-数据集下载)

- [2. 结果提交格式](#2-结果提交格式)

## 1. 数据集下载

### 1.1 训练集

- 中文 Lang8 学习者数据（与赛道三相同）下载：

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

- 中文拼写检查数据（SIGHAN[^1]+Wang271K[^2]）下载

> 下载地址：http://yunpan.blcu.edu.cn:80/link/EF0963CBC2A979A71971BFECE8A34234
>
> 有效期限：2022-10-31 23:59
>
> 访问密码：lO42

该数据格式为：

```
[原始句子] ||| [修改句子]
```

其中原始句子与修改句子一一对应，中间使用 " ||| " 分隔。

**注意：除Lang-8外，参赛者可使用任意开源数据进行训练。但Lang-8数据只能使用本比赛提供的版本。SIGHAN和Wang271K推荐使用本比赛提供的版本。**

### 1.2 开发集

本赛道提供平行句对（`.src`, `.trg`）及修改结果标签(`.lbl`)文件。

其中`.src`和`.trg`的文件格式为：

```
[id]	[原始句子/修改句子]
```

每行包含该行的索引及对应的句子两列，每列之间使用 "\t" 分隔。

`.lbl` 文件为每个句子的错误探测和错误纠正结果，用于模型评估，格式为：

```
[id], [错误位置], [正确字符], [错误位置], [正确字符], ... 
```

如果该句不包含错误，则格式为：

```
[id], 0
```

每项之间用", "分隔。

### 1.3 测试集

本赛道的测试数据位置位于`test`文件夹。

测试数据输入格式为：

```
[id]	[原始句子]
```

句子 id 和原始句子之间使用 "\t" 分隔。

## 2. 结果提交格式

参赛者需要提供与给定原始句子索引对应的预测标签，格式与验证集的`.lbl`格式相同。

如该句预测出错误及纠正结果，则格式为：
```
[id], [错误位置], [正确字符], [错误位置], [正确字符], ... 
```
如该句没有预测出错误，则格式为：
```
[id], 0
```

每项之间用", "分隔。

提交前，文件需依规范正确命名，并压缩成 `.zip` 格式文件的压缩包。

提交结果命名：

```
track1_test.zip	#压缩包名字
    └── yaclc-csc-test.lbl	# 预测结果文件
```

## 参考文献
[^1]:Tseng Yuen-Hsien, Lung-Hao Lee, Li-Ping Chang, and Hsin-Hsi Chen. 2015. Introduction to SIGHAN 2015 Bake-off for Chinese Spelling Check. In Proceedings of the Eighth SIGHAN Workshop on Chinese Language Processing, pages 32–37. ([pdf](https://aclanthology.org/W15-3106))
[^2]:Wang Dingmin, Yan Song, Jing Li, Jialong Han, and Haisong Zhang. 2018. A Hybrid Approach to Automatic Corpus Generation for Chinese Spelling Check. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2517–27. ([pdf](https://aclanthology.org/D18-1273))
