# 赛道一 （中文拼写检查）评价指标

本项目提供了中文拼写检查任务的常用指标，包括句子级别（Sentence Level）和字符级别（Character Level）两个层面，及错误检测（Error Detection）和错误纠正（Error Detection）两个评价维度。

其中，句子级别的评价脚本参考了ReaLiSe，字符级别的评价脚本参考了SIGHAN 2015。

## 1. 使用方式

```bash
bash csc_eval.sh hyp_file gold_file
```
其中，`hyp_file`和`gold_file`的格式可参看`demo`中的`demo.hyp.labels`和`demo.gold.labels`。

## 2. 输出格式

```
CSC Evaluation Report:

========== Sentence Level ==========
Detection:
Accuracy: 86.67, Precision: 81.25, Recall: 81.25, F1: 81.25
Correction:
Accuracy: 86.67, Precision: 81.25, Recall: 81.25, F1: 81.25

========== Character Level ==========
Detection:
Accuracy: 86.67, Precision: 92.86, Recall: 81.25, F1: 86.67
Correction:
Accuracy: 86.67, Precision: 92.86, Recall: 81.25, F1: 86.67
```

## 参考文献
[1] Heng-Da Xu, Zhongli Li, Qingyu Zhou, et al. “Read, Listen, and See: Leveraging Multimodal Information Helps Chinese Spell Checking.” In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, 716–28. 

[2] Yuen-Hsien Tseng, Lung-Hao Lee, Li-Ping Chang, et al. “Introduction to SIGHAN 2015 Bake-off for Chinese Spelling Check.” In Proceedings of the Eighth SIGHAN Workshop on Chinese Language Processing, 32–37.
