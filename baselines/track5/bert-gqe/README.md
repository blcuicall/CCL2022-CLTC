# bert_gqe基线模型使用说明
## 1.环境配置
- pytorch >=1.8.0
- transformers>=4.7.0
## 2.预训练模型
只需将预训练模型名称放在```.sh```文件中的```--model_name_or_path```后即可，需要的模型为：
- bert-base-chinese
## 3.代码结构说明
以下是本项目主要代码结构及说明：
```
├── script                # 存放基线脚本
├── bert_model.py         # bert模型的相应定义
├── data_loader.py        # Dataset子类，完成数据读取
├── file_utils.py         # 处理本地数据集缓存
├── trans_data.py         # 将train set数据处理成含有0,1标签的数据
├── generate_feature.py   # 对修改句生成相应质量评估分数
├── models.py             # 完成模型结构设计以及前向传播流程
├── results_to_file.py    # 将质量评估分数调整成最终提交格式
├── train.py              # 完成训练和验证并保存最佳模型
└── README.md             # 文档说明
```
## 4.模型训练、生成、结果排序及输出
- 训练前使用`python trans_data.py`对训练集数据进行处理
- 训练使用```train.sh```脚本，其中训练数据采用track3中的训练集
- 生成质量评估分数使用```generate_feature.sh```脚本
- 根据质量评估分数并排序输出top-1结果使用```results_to_file.sh```脚本

### 以下为脚本详细说明：
### train.sh
```
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python train.py --outdir ./save_model \
--train_path ./data_gqe/train.json \
--valid_path ./data_gqe/yaclc-fluency_dev.json \
--model_name_or_path bert-base-chinese
```
关键参数解析：
- ```CUDA_VISIBLE_DEVICES```：使用的gpu卡号
- ```output```：训练生成的最佳模型保存路径
- ```valid_path```:验证集路径
- ```train_path```:训练集路径
- ```model_name_or_path```:预训练使用的模型

### generate_feature.sh
```
python generate_feature.py --test_path ./data_gqe/yaclc-fluency_testA.json \
--out_path ./data_feature/output_phase1_fluency.json \
--model_name_or_path bert-base-chinese \
--checkpoint ./save_model/model.best.pt
```
关键参数解析：
- ```test_path```：测试集路径
- ```output_path```：质量评估分数输出路径
- ```checkpoint```:最佳模型存放路径

### results_to_file.sh
```
python results_to_file.py --test_path ./data_qe/yaclc-fluency_testA.jsonn \
--test_score_path ./data_feature/output_phase1_fluency.json \
--output_path ./output/yaclc-fluency_testA.para
```
关键参数解析：
- ```test_path```：测试集路径
- ```test_score_path```：质量评估分数输出路径
- ```output_path```:最终提交数据存放路径
