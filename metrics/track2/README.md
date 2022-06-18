# 赛道二（中文语法错误检测）评价指标

## 1. 环境配置

```shell
pip install -r requirements.txt
```

## 2. 使用方式

```shell
bash run_eval.sh
```

其中evaluation.pl为官方评测脚本，pair2edits_word.py与pair2edits_char.py为将句对转换为编辑的简易脚本，区别在于pair2edits_word.py考虑了词语的边界信息，用于适配CGED21测试集；本次评测推荐使用pair2edits_char.py脚本
