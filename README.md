
# CCL 2022 汉语学习者文本纠错评测


## 最新消息
| 时间 | 消息 |
|-----| ----- |
| 7 月 30 日 | [关于队伍重组功能限时上线的说明](docs/07-31-reorganize.md)  <br>[关于赛道三、赛道五第二阶段提交的说明](07-31-phrase-2.md) |
| 7 月 8 日 | 赛道一、二及赛道三、五的第一阶段评测[提交入口](http://cuge.baai.ac.cn/#/ccl_yaclc)开放，<br>点击各个赛道的排行榜可见“提交评测”按钮。点击查看提交[注意事项](docs/07-08-submit-guidline.md)。 |
| 7 月 6 日 | 公开赛道二数据集，见 [datasets/track2](datasets/track2) 目录 |
| 6 月 19 日 | 公开各赛道基线模型（[baselines](baselines) ）和评测代码（[metrics](metrics)） |
| 6 月 14 日 | 已建立 CCL2022-CLTC 比赛交流微信群，点击[查看详情](docs/06-14-wechat-group.md) |
| 6 月 14 日 | 由于平台故障，[智源平台](http://cuge.baai.ac.cn/#/ccl_yaclc)账号及队伍需重新注册，请点击[查看详情](docs/06-14-cuge-re-register.md) |
| 6 月 12 日 | track1 dev 数据集已重新上传，请以最新版为准 |
| 6 月 10 日 | 数据集已公开，见 [datasets](datasets) 目录 |
| 6 月 5 日 | 报名入口已开通，详情请见：[报名方式](#6-报名方式) |
| 5 月 31 日 | 报名入口开放时间更新为 6 月 5 日 |


## 目录
- [1. 比赛介绍](#1-比赛介绍)
- [2. 任务内容](#2-任务内容)
- [3. 评测数据](#3-评测数据)
- [4. 评价标准](#4-评价标准)
- [5. 评测赛程](#5-评测赛程)
- [6. 报名方式](#6-报名方式)
- [7. 奖项设置](#7-奖项设置)

## 1. 比赛介绍
汉语学习者文本纠错任务（Chinese Learner Text Correction，CLTC）旨在自动检测并修改汉语学习者文本中的标点、拼写、语法、语义等错误，从而获得符合原意的正确句子。近年来，该任务越来越受到关注，也出现了一些有潜在商业价值的应用。为了推动这项研究的发展，研究者通过专家标注以及众包等形式构建一定规模的训练和测试数据，在语法检查以及语法纠错等不同任务上开展技术评测。同时，由于汉语学习者文本纠错任务相对复杂、各评测任务以及各数据集之间存在差异，在一定程度上限制了文本纠错的发展。因此，我们希望通过汇聚、开发数据集，建立基于多参考答案的评价标准，完善文本纠错数据及任务，聚焦该研究领域中的前沿问题，进一步推动汉语学习者文本纠错研究的发展。

我们依托第二十一届中国计算语言学大会（CCL 2022），组织汉语学习者文本纠错评测。本次评测既整合了已有的相关评测数据和任务，又有新开发的数据集，以设置多赛道、统一入口的方式开展比赛任务。同时，我们研制了各赛道具有可比性的评测指标，立足于构建汉语学习者文本纠错任务的基准评测框架。 

- 组织者
  - 杨麟儿（北京语言大学）
  - 杨尔弘（北京语言大学）
  - 李正华（苏州大学）
  - 孙茂松（清华大学）
  -  张民（苏州大学）
  - 刘正皓（东北大学）
  - 饶高琦（北京语言大学）
  - 李辰（阿里巴巴达摩院）

- 联系人
  - 王莹莹（北京语言大学博士生，总负责，blcuicall@163.com）
  - 孔存良（北京语言大学博士生，赛道三）
  - 章岳（苏州大学硕士生，赛道四）
  - 梁念宁（清华大学硕士生，赛道一）
  - 方雪至（北京语言大学硕士生，赛道二）
  - 周天硕（东北大学硕士生，赛道五）

评测任务更详细内容可查看评测网站：[https://github.com/blcuicall/CCL2022-CLTC](https://github.com/blcuicall/CCL2022-CLTC)，遇到任何问题请发邮件或者在[Issue](https://github.com/blcuicall/CCL2022-CLTC/issues)中提问，欢迎大家参与。

[--返回目录--](#目录)

## 2. 任务内容

本次评测设置下述五个赛道：

**赛道一**：中文拼写检查（Chinese Spelling Check）任务目的是检测并纠正中文文本中的拼写错误（Spelling Errors）。对于给定的一段输入文本，最终需给出拼写错误的位置及对应的修改结果，其中拼写错误包含：音近、形近、形音兼近三种。如表 1 所示，“14”“15”为两个错误位置，“印”“象”为对应位置的修改结果。如该句没有错误，则输出“（id=xxx） 0”即可。

<p align='center'>表1：中文拼写检查任务示例</p>
<table align='center'>
<tr>
<td> 原句  </td>
<td> (id=012) 我觉得春天给人留下清爽的好影响。 </td>
</tr>
<tr>
<td> 拼写错误检测及纠正  </td>
<td> (id=012) 14,印,15,象 </td>
</tr>
</table>


**赛道二**：中文语法错误检测（Chinese Grammatical Error Diagnosis）任务目的是检测出中文文本中每一处语法错误的位置、类型。语法错误的类型分为赘余(Redundant Words，R)、遗漏(Missing Words，M)、误用(Word Selection，S)、错序(Word Ordering Errors，W)四类。评测任务要求参加评测的系统输入句子（群），其中包含有零个到多个错误。参赛系统应判断该输入是否包含错误，并识别错误类型，标记出其在句子中的位置和范围，对缺失和误用给出修正答案。

<p align='center'>表2：中文语法错误检测任务示例</p>
<table align='center'>
<tr>
<td> 原句  </td>
<td> (sid=00038800481)  我根本不能了解这妇女辞职回家的现象。在这个时代，为什么放弃自己的工作，就回家当家庭主妇？ </td>
</tr>
<tr>
<td> 语法错误检测  </td>
    <td>00038800481, 6, 6, S, 理 </br> 00038800481, 8, 8, R </br>（“了解”应为“理解”，删去“这”）</td>
</tr>
<tr>
    <td> 原句 </td>
    <td>(sid=00038800464)我真不明白。她们可能是追求一些前代的浪漫。</td>
</tr>
<tr>
    <td>语法错误检测</td>
    <td>00038800464, correct</br>（原句正确，没有错误）</td>
</tr>
</table>


**赛道三**：多维度汉语学习者文本纠错（Multidimensional Chinese Learner Text Correction）。同一个语法错误从不同语法点的角度可被划定为不同的性质和类型[^1]，也会因语言使用的场景不同、具体需求不同，存在多种正确的修改方案。赛道三的数据中提供针对一个句子的多个参考答案，并且从最小改动（Minimal Edit，M）和流利提升（Fluency Edit，F）两个维度对模型结果进行评测。最小改动维度要求尽可能好地维持原句的结构，尽可能少地增删、替换句中的词语，使句子符合汉语语法规则；流利提升维度则进一步要求将句子修改得更为流利和地道，符合汉语母语者的表达习惯。如表 3 中所示，原句在两个维度均有多个语法纠错的参考答案。

<p align='center'>表3：多参考中文语法纠错任务示例</p>
<table align='center'>
    <tr>
        <td> 原句 </td>
        <td colspan='2'>因为我的中文没有好，我还要努力学汉语。</td>
    </tr>
    <tr>
        <td rowspan='2'>最小改动</td>
        <td>参考答案1</td>
        <td>因为我的中文<del> 没有 </del><b>不</b>好，我还<del> 要 </del><b>在</b>努力学汉语。</td>
    </tr>
    <tr>
        <td>参考答案2</td>
        <td>因为我的中文<del> 没有 </del><b>不</b>好，<i>所以</i>我还要努力学汉语。</td>
    </tr>
    <tr>
        <td rowspan='2'>流利提升</td>
        <td>参考答案1</td>
        <td><del>因为</del>我的中文没有<i>那么</i>好，<i>因此</i>我还要努力学汉语。</td>
    </tr>
    <tr>
        <td>参考答案2</td>
        <td>因为我的中文<i>还</i>没有<i>学</i>好，<i>所以</i>我还要<i>更加</i>努力<i>地</i>学<del> 汉语 </del>中文。</td>
    </tr>
</table>

注：其中，<b>加粗</b>表示替换字符，<i>斜体</i>表示插入字符，<del> 删除线 </del>表示删除字符。

**赛道四**：多参考多来源汉语学习者文本纠错（Multi-reference Multi-source Chinese Learner Text Correction）。不同来源的文本，其蕴含的语法错误类型也可能含有一定的差异。赛道四提供来自于三个不同文本源的中文学习者语法纠错评测数据，对于每一个句子提供多个遵循流利提升的修改答案，希望能够准确而全面地评估各参赛队伍的纠错系统性能。

**赛道五**：语法纠错质量评估（Quality Estimation），是评价语法纠错模型修改结果质量的方法[^2]。如表4所示，该方法通过预测每一个语法纠错结果的质量评估分数（QE Score）来对语法纠错的结果进行质量评估，以期望对冗余修改、错误修改以及欠修改情况进行评估。该分数可以通过句子级别和词级别的质量评估分数得到[^3]，可以对语法纠错系统生成的多个纠错结果进行重新排序，以期望进一步提升语法纠错效果。

<p align='center'>表4：语法纠错质量评估任务示例</p>
<table align='center'>
    <tr>
        <td>原句</td>
        <td>他今天去田里干活，我不只道他何时从田返回回来。</td>
        <td>质量评估分数</td>
    </tr>
    <tr>
        <td>修改结果1</td>
        <td>他今天去田里干活，我不<b>知</b>道他何时从田<i>里</i>返回<del> 回来 </del>。</td>
        <td>1.0</td>
    </tr>
    <tr>
        <td>修改结果2</td>
        <td>他今天去田里干活，我不<b>知</b>道他何时从田<i>里</i><del> 返回 </del>回来。</td>
        <td>1.0</td>
    </tr>
    <tr>
        <td>修改结果3</td>
        <td>他今天去田里干活，我不<b>知</b>道他何时从田返回回来。</td>
        <td>0.3846</td>
    </tr>
</table>

注：其中，<b>加粗</b>表示替换字符，<i>斜体</i>表示插入字符，<del> 删除线 </del>表示删除字符。

[--返回目录--](#目录)

## 3. 评测数据

本节介绍各赛道数据集来源及数据集使用规则，各赛道比赛数据及结果文件的提交格式请参看：[数据集页面](datasets)。

### 3.1 赛道一：中文拼写检查

#### 训练集

本赛道允许使用任意开源数据用于训练。例如，可使用现有的真实开源数据集进行训练，如 SIGHAN 2013[^4]、CLP 2014[^5]、SIGHAN 2015[^6]等，也可以使用伪数据，如 Wang et al. [^7]提供的数据集。

为便于参赛者使用，本赛道提供一份上述数据经过处理后的版本。参赛者也可以重新处理这些数据，或自行选用其他数据。

此外，SIGHAN 历年赛事中也给出了音近、形近混淆集（Confusion Set）作为参考，参赛者可按需使用。

#### 开发集与测试集

本赛道提供基于 YACLC-CSC 数据集[^9]的开发集与测试集。在拼写错误标注方面，YACLC-CSC 继承前人的研究，规定只标注和修正“音近”和“形近”有关的错误。判定为“音近”或“形近”或“形音兼近”的依据来自相关的汉语语音学、文字学理论及对外汉语教学理论。标注过程采用多人标注再由专家审核的方式以保证标注质量。

上述训练、开发与测试数据可于 [datasets/track1](datasets/track1) 下载。

### 3.2 赛道二：中文语法错误检测

#### 训练集

本赛道提供两个中介语数据集：

- 中文 Lang8 数据集
- CGED 历年数据

参赛者可以使用上述**中介语数据集**及任意**开源中文母语数据**用作训练。

#### 测试集

提供 CGED-8 数据集。数据来源为 HSK 动态作文语料库[^11]和全球汉语中介语语料库[^12][^13]。CGED-8 共包括约 1,400 个段落单元、3,000 个错误。每个单元包含 1-5 个句子，每个句子都被标注了语法错误的位置、类型和修改结果。

上述训练与测试数据可于 [datasets/track2](datasets/track2) 下载。

### 3.3 赛道三：多维度汉语学习者文本纠错

#### 训练集

本赛道对 NLPCC2018-GEC[^8] 发布的采集自 Lang8 平台的中介语数据进行了处理。

参赛者**仅允许**使用上述数据用于训练。

#### 开发集与测试集

本赛道提供最小改动和流利提升两个维度的多参考数据集 YACLC-Minimal[^9]、YACLC-Fluency[^9] 。其中 YACLC-Minimal 属于最小改动维度，YACLC-Fluency属于流利提升维度。

上述训练、开发与测试数据可于 [datasets/track3](datasets/track3) 下载。

### 3.4 赛道四：多参考多来源汉语学习者文本纠错

#### 训练集

需要注意本次评测不提供官方训练数据集，参赛选手可自行使用任何公开的训练数据或是人造数据。

#### 开发集与测试集

提供基于流利提升的多参考数据集MuCGEC[^10]。

具体要求请参看[赛道四主页](https://tianchi.aliyun.com/dataset/dataDetail?dataId=131328)。

### 3.5 赛道五：语法纠错质量评估

#### 训练集

本赛道训练数据基于赛道三提供的中文 lang8 数据构建。本赛道使用 BART-large 训练了基于 seq2seq 结构的语法纠错模型，并将该模型在柱搜索解码过程中排名前 5 的结果作为待进行质量评估的语法纠错候选方案。同时评测数据给出了训练集和开发集中每个语法纠错方案的真实 F0.5 分值。

参赛者**仅允许**使用赛道三提供的中文 lang8 数据，以及本赛道提供的带有语法纠错候选方案的数据。

#### 开发集与测试集

开发集与测试集基于赛道三提供的 YACLC-Minimal 和 YACLC-Fluency 进行构建，数据划分与赛道三相同。

**注意**：本赛道要求语法纠错结果重排序过程中只能对所提供的语法纠错候选进行重排序，不得混合其他语法纠错模型所提供的语法纠错结果。

上述训练、开发与测试数据可于 [datasets/track5](datasets/track5) 下载。

### 3.6 数据使用规则


本次评测在模型训练方面的具体规则如下：

1. 非本比赛提供的数据必须可以开源获取，并应在论文中说明或以其他方式向比赛组织方公开，不得使用闭源及私有数据。
2. 参与者禁止注册多账户报名，经发现将取消成绩并严肃处理。
3. 参与者禁止在指定考核技术能力的范围外利用规则漏洞或技术漏洞等不良途径提高成绩排名，经发现将取消成绩并严肃处理。
4. 可以接触到赛题相关数据的人员，其提交结果将不计入排行榜及评奖。
5. **评测环节、评测数据一切最终解释权归评测组委会所有。**
6. **成功报名评测视同接受此规则及组委会对此规则进行的后续修改。**

[--返回目录--](#目录)

## 4. 评价标准

本次评测 5 个赛道的测试数据集采用封闭方式给出，即仅给定输入文本，需要参赛队伍进行推理预测，并将结果文件打包上传至在线评测平台。平台随后会给出相应赛道的指标得分。每支队伍每天仅可提交 3 次测试集结果。

**注意**：本次评测中，赛道三、四、五的评测分**两阶段**进行，在第一阶段评测分数达到前50%（暂定，视报名情况最终决定）的队伍，可以参与第二阶段的最终评测。赛道一、二的评测只有一个阶段。

### 4.1 赛道一：中文拼写检查

赛道一所需的结果文件格式见表 1 的任务示例。文件每行是对应一个原句的校对结果，每行内容为原句的 id，错误位置及纠正结果。所使用的评测指标分为两种：一种是句级（Sentence Level），即一句输入文本中所有错别字都检测或纠正正确，则算作正确；一种是字级（Character Level），即不考虑当前句的限制，最终的评价是基于整个测试集所有汉字的错误检测或纠正结果确定。对每个级别来说，又分为错误检测（Error Detection）和错误纠正（Error Correction）两个维度。错误检测评估的是错误位置的侦测效果，错误纠正评估的是对应位置错误修正的效果。对于每个维度的评测，我们统一使用准确率（Precision）、召回率（Recall）和 F<sub>1</sub> 作为评价指标。

### 4.2 赛道二：中文语法错误检测

赛道二所需的结果文件格式见表 2 的任务示例。文件每行是对应一个原句的一处检测结果，每行内容为原句的 id、错误位置、错误类型及纠正结果。从下述六个方面以精确率、召回率和 F<sub>1</sub> 值对系统性能进行评价：

1. 假阳性（False Positive）：正确句子被判包含错误的比例。
2. 侦测层（Detective-level）：对句子是否包含错误做二分判断。
3. 识别层（Identification-level）：给出错误点的错误类型。
4. 定位层（Position-level）：对错误点的位置和覆盖范围进行判断，以字符偏移量计。
5. 修正层（Correction-level）：提交针对字符串误用（S）和缺失（M）两种错误类型的修正词语。修正词语可以是一个词，也可以是一个词组。
6. 综合打分（Comprehensive Score）：2022 年 CGED-8 引入 1-5 这五项指标的加权平均分数作为综合打分，各分项排名均会公布，而最终获奖名单将由综合分数排名决定。

### 4.3 赛道三：多维度汉语学习者文本纠错

赛道三所需的结果文件格式是每行对应一个原句的纠正结果，且每个原句仅需提供一个结果。采用的评测指标为字级别的 F<sub>0.5</sub> 指标。

### 4.4 赛道四：多参考多来源汉语学习者文本纠错

赛道四与赛道三的文件格式及评估指标相同。

### 4.5 赛道五：语法纠错质量评估

赛道五需要最终提供一个语法纠错质量评估结果，该结果可以由多个语法纠错质量评估模型整合得到。评测分为两个方面：

1. 评价质量评估模型所生成的质量评估分数，具体而言是计算模型给出同一个输入文本的不同语法纠错结果的质量评估分数与真实 F<sub>0.5</sub> 分数之间的皮尔逊相关系数（Pearson Correlation Coefficient，PCC)，最后根据全部评测样例求取平均值用以衡量语法纠错质量评估分数与真实 F<sub>0.5</sub> 分数之间的相关性。
2. 对给定的语法纠错结果进行重新排序，并选取分数最高的语法纠错结果作为最终的语法纠错结果，用以评价质量评估模型在语法纠错任务上的效果，采用的评价指标和计算方式与赛道三相同。

[--返回目录--](#目录)

## 5. 评测赛程

|          时间          |          事项          |
|-----------------------|------------------------|
| 6 月 5 日 ~ 7 月 20 日 | 开放报名 |
| 6 月 10 日 | 发布所有赛道的训练集和开发集 |
| 6 月 12 日 | 发布赛道三、四、五的第一阶段测试集 |
| 6 月 15 日 | 发布所有赛道的 Baseline 代码及结果 |
| 6 月 20 日 | 参赛系统结果提交入口开放 |
| 8 月 5 日 | 赛道三、四、五第一阶段结束 |
| 8 月 10 日 | 发布赛道一、二的测试集，赛道三、四、五的第二阶段测试集 |
| 8 月 25 日 | 平台测试集结果提交入口关闭 |
| 9 月 10 日 | 公布评测结果 |
| 9 月 25 日 | 截止提交评测任务技术报告 |
| 10 月 14 日 ~ 16 日 | 评测研讨会 |

[--返回目录--](#目录)

## 6. 报名方式

### 赛道一、二、三、五
报名入口：[智源平台](http://cuge.baai.ac.cn/#/ccl_yaclc)

操作方式：注册智源平台，由队长创建队伍后，凭邀请码邀请其余组员入队。每个队伍需指定一位提交人提交评测，默认为队伍创建人。队伍可以在一、二、三、五四个赛道上提交结果。

### 赛道四
报名方式：在[天池平台](https://tianchi.aliyun.com/dataset/dataDetail?dataId=131328)页面下载报名表，按照规定要求填写后，以附件形式发送邮件到邮箱：ccl2022track4@163.com 进行报名

[--返回目录--](#目录)

## 7. 奖项设置

本次评测将评选出一、二、三等奖，奖池共计 50000 元人民币：
- 一等奖 0-5 名，奖金合计 25000 元；
- 二等奖 0-5 名，奖金合计 15000 元；
- 三等奖 0-5 名，奖金合计 10000 元。

另外，中国中文信息学会将为本次评测获奖队伍提供荣誉证书。

[--返回目录--](#目录)

**参考文献**

[^1]: 张宝林. 2013. 关于通用型汉语中介语语料库标注模式的再认识. 世界汉语教学, 01:128-140.
[^2]: Shamil Chollampatt and Hwee Tou Ng. 2018. Neural quality estimation of grammatical error correction. In Proceedings of EMNLP, pages 2528–2539. ([pdf](https://aclanthology.org/D18-1274))
[^3]: Zhenghao Liu, Xiaoyuan Yi, Maosong Sun, Liner Yang, and Tat-Seng Chua. 2021. Neural quality estimation with multiple hypotheses for grammatical error correction. In Proceedings of NAACL-HLT, pages 5441–5452. ([pdf](https://aclanthology.org/2021.naacl-main.429.pdf))
[^4]: Wu Shih-Hung, Chao-Lin Liu, and Lung-Hao Lee. 2013. Chinese Spelling Check Evaluation at SIGHAN Bake-off 2013. In Proceedings of the Seventh SIGHAN Workshop on Chinese Language Processing, pages 35–42. ([pdf](https://aclanthology.org/W13-4406))
[^5]: Yu Liang-Chih, Lung-Hao Lee, Yuen-Hsien Tseng, and Hsin-Hsi Chen. 2014. Overview of SIGHAN 2014 Bake-off for Chinese Spelling Check. In Proceedings of The Third CIPS-SIGHAN Joint Conference on Chinese Language Processing, pages 126–32. ([pdf](https://aclanthology.org/W14-6820))
[^6]: Tseng Yuen-Hsien, Lung-Hao Lee, Li-Ping Chang, and Hsin-Hsi Chen. 2015. Introduction to SIGHAN 2015 Bake-off for Chinese Spelling Check. In Proceedings of the Eighth SIGHAN Workshop on Chinese Language Processing, pages 32–37. ([pdf](https://aclanthology.org/W15-3106))
[^7]: Wang Dingmin, Yan Song, Jing Li, Jialong Han, and Haisong Zhang. 2018. A Hybrid Approach to Automatic Corpus Generation for Chinese Spelling Check. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2517–27. ([pdf](https://aclanthology.org/D18-1273))
[^8]: Yuanyuan Zhao, Nan Jiang, Weiwei Sun, and Xiaojun Wan. 2018. Overview of the nlpcc 2018 shared task: Grammatical error correction. In CCF International Conference on Natural Language Processing and Chinese Computing (NLPCC), pages 439–445. ([pdf](http://tcci.ccf.org.cn/conference/2018/papers/EV11.pdf))
[^9]: Yingying Wang, Cunliang Kong, Liner Yang, Yijun Wang, Xiaorong Lu, Renfen Hu, Shan He, Zhenghao Liu, Yun Chen, Erhong Yang, and Maosong Sun. 2021. YACLC: A Chinese Learner Corpus with Multidimensional Annotation. arXiv preprint arXiv:2112.15043. ([pdf](https://arxiv.org/abs/2112.15043))
[^10]: Yue Zhang, Zhenghua Li, Zuyi Bao, Jiacheng Li, Bo Zhang, Chen Li, Fei Huang, and Min Zhang. 2022. MuCGEC: a Multi-Reference Multi-Source Evaluation Dataset for Chinese Grammatical Error Correction. In Proceedings of NAACL-HLT. ([pdf](https://arxiv.org/pdf/2204.10994.pdf))
[^11]: Gaoqi Rao, Erhong Yang, and Baolin Zhang. Overview of NLPTEA-2020 Shared Task for Chinese Grammatical Error Diagnosis. In Proceedings of the 6th Workshop on Natural Language Processing Techniques for Educational Applications. ([pdf](https://aclanthology.org/2020.nlptea-1.4.pdf))
[^12]: 张宝林. 2009. “HSK动态作文语料库”的特色与功能. 汉语国际教育, 4:71–79.
[^13]: 张宝林,崔希亮. 2022. “全球汉语中介语语料库”的特点与功能. 世界汉语教学, 01:90-100.
