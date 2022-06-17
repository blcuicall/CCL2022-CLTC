# 基线模型使用说明

## 环境配置

1. 安装pytorch
   
   ```
   conda create -n csc python=3.7
   conda activate csc
   conda install pytorch=1.7.1 cudatoolkit -c pytorch
   ```

2. 安装其他依赖
   
   ```bash
   python 3.7.13
   transformers 4.19.1
   ```

## 模型训练

- 数据预处理+模型训练，编辑 pipeline.sh，运行
  
  ```bash
  bash pipeline.sh
  ```

## 模型预测

- 编辑 decode.sh，运行
  
  ```bash
  bash decode.sh
  ```
