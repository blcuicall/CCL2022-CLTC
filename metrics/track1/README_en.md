# CSC-metrics
Evalution Metrics for Chinese SpellCheck

## Intro
The project provides scripts of commonly used metric calculation for Chinese Spelling Checking. Both sentence-level and character-level metrics are supported.

Note: We followed ReaLiSe and SIGHAN 2015 bake-off to implement metrics for sentence level and character level.

## Usage

```bash
bash csc_eval.sh hyp_file gold_file
```

## Output Format

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

## References
[1] Heng-Da Xu, Zhongli Li, Qingyu Zhou, et al. “Read, Listen, and See: Leveraging Multimodal Information Helps Chinese Spell Checking.” In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, 716–28. 

[2] Yuen-Hsien Tseng, Lung-Hao Lee, Li-Ping Chang, et al. “Introduction to SIGHAN 2015 Bake-off for Chinese Spelling Check.” In Proceedings of the Eighth SIGHAN Workshop on Chinese Language Processing, 32–37.
