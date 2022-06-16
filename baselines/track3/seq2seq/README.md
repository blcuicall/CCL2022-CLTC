# seq2seq 基线模型使用说明

## 1. 环境配置

- pytorch >= 1.8.0
- fariseq

```shell
git clone https://github.com/pytorch/fairseq
cd fairseq
git reset --hard 06c65c82973969
pip install --editable ./
```

- transformers >= 4.7.0
- apex (optional)

## 2. 预训练模型下载

将预训练模型放到 `pretrained-models/` 目录下，需要的模型为：

- bart-base-chinese: https://huggingface.co/fnlp/bart-base-chinese

## 3. 数据预处理

训练和验证数据需分别处理成 `.src`  和 `.tgt` 两个文件并放在 `data/raw` 目录下，目录结构如：

```
data
  └── raw
    ├── train.src
    ├── train.tgt
    ├── valid.src
    └── valid.tgt
  ├── bpe
  └── processed
```

然后运行 `data_process.sh` 进行数据预处理，预处理后的数据放在 `data/raw/processed` 中

## 4. 模型训练、推断、评价

- 训练使用 `train.sh` 脚本
- 推断使用脚本 `interactive.sh`
- 推断之前需要先对 `test.src` 数据做 BPE，BPE后的数据可以放在 `data/bpe` 中，便于使用
- 生成结果评价使用 `score-test.sh` 和 `score-valid.sh` 两个脚本
