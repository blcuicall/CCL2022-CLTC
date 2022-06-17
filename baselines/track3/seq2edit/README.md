# seq2edit基线模型使用说明

## 环境配置

1. 安装pytorch
    ```
    conda create -n gector_env python=3.7.6 -y
    conda activate gector_env
    conda install pytorch=1.10.1 cudatoolkit -c pytorch
    ```

2. 安装NVIDIA-Apex
    ```bash
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    ```
3. 安装其他依赖
    ```bash
    python==3.7.6
    transformers==4.14.1
    scikit-learn==1.0.2
    numpy==1.21.2
    deepspeed==0.5.10
    ```

## 数据预处理
1. 字符按空格分开，每句一行

2. 生成编辑序列
    ```bash
    python utils/preprocess_data.py -s source_file -t target_file -o output_edit_file
    ```

3. \*(可选) 定义自己的词表 (data/vocabulary/labels.txt)

## 模型训练
- 编辑deepspeed_config.json. 需要注意的是，lr和batch_size会被args参数覆盖。然后编辑train.sh，并运行
   ```bash
   bash train.sh
   ```

## 模型预测
- 编辑deepspeed_config.json，然后编辑predict.sh，并运行
    ```bash
    bash predict.sh
    ```
