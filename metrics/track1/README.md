# 赛道一 （中文拼写检查）评价指标

本项目提供了中文拼写检查任务的常用指标，包括句子级别（Sentence Level）和字符级别（Character Level）两个层面，及错误检测（Error Detection）和错误纠正（Error Detection）两个评价维度。

其中，句子级别的评价脚本参考了ReaLiSe[^1]，字符级别的评价脚本修改自 Wang et al. (2019)[^2] 所提供的[开源代码](https://github.com/sunnyqiny/Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check)。

> 注意：根据 Wang et al. (2019) 所提供的评价脚本，字符级别的 Correction 指标只检测那些被正确 Detect 出来的位置是否被修改正确。如果修改正确，则计入TP，若修改错误，则计入FP，若该被修改的位置，未被修改则被记为FN。因此，字符级别的 Correction 指标超过 Detection 指标是正常现象。

## 1. 使用方式

```bash
# char
python eval_char_level.py --src src_file --gold gold_file --hyp hyp_file
# sent
python eval_sent_level.py --hyp hyp_file --gold gold_file
```

其中，`src_file`、`hyp_file`和`gold_file`的格式可参看`samples`中的样例。

## 2. 输出格式

Character Level

```
========== Character Level ==========
Detection: 
Precision: 75.0, Recall: 52.75, F1: 61.94
Correction: 
Precision: 89.58, Recall: 80.27, F1: 84.6
```

Sentence Level

```
========== Sentence Level ==========
Detection: 
Precision: 66.26, Recall: 49.27, F1: 56.52
Correction: 
Precision: 58.68, Recall: 43.64, F1: 50.05
```

## 参考文献

[^1]:Heng-Da Xu, Zhongli Li, Qingyu Zhou, et al. “Read, Listen, and See: Leveraging Multimodal Information Helps Chinese Spell Checking.” In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, 716–28. 
[^2]:Dingmin Wang, Yi Tay, and Li Zhong. 2019. Confusionset-guided pointer networks for chinese spelling check. In Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pages 5780–5785.
