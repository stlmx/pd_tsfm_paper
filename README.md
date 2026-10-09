# pd_tsfm_paper

基于通用时序预训练模型（Time-Series Foundation Model, TSFM）的**小样本局部放电（PD）模式识别**。毕业论文用稿，目标是以最小成本、最快速度投出并录用。

## 一句话定位

通用 TSFM 的输入约定（定长上下文 + 实例归一化）丢掉了局放测量中两个被标定的参考量：**工频相位**和**放电幅值**。本文用**测量锚定的输入接口**（按相位分 bin + 幅值回注），配合**冻结主干 + 轻量适配**，让通用 TSFM 在少量标注的实验室局放数据上稳定迁移。

> 暂定题目：*Phase-Anchored Adaptation of Pretrained Time-Series Foundation Models for Few-Shot Partial Discharge Pattern Recognition*

## 投稿顺序

| 顺序 | 期刊 | 理由 |
|---|---|---|
| 1 主投 | Measurement Science and Technology（IOP，ISSN 0957-0233） | 门槛最低，一投命中概率最高；测量视角最贴合 |
| 2 被拒转投 | Electric Power Systems Research（Elsevier，ISSN 0378-7796） | 受众换成电力方向，只需调整引言和讨论的侧重 |
| 3 保底 | Measurement（Elsevier，ISSN 0263-2241） | 诊断类文章多 |

三本刊都在学院「2022 最有国际影响力学术期刊目录」内（序号 706 / 266 / 705）。写作时保证换刊只需要换模板。

## 范围

- **本篇只做局放。** 断路器机械特性数据留给第二篇，复用同一套流程，把相位锚定换成事件锚定。
- 与 PowerBrain（TII 重投，截止 2026-11-29）分开推进，互不占用。

## 目录

| 路径 | 内容 |
|---|---|
| `docs/plan.md` | **执行底本**：分阶段任务、验收门、时间线、待拍板决策、风险 |
| `docs/data_card.md` | 数据卡：缺陷类别、试品、采样、CSV 格式、相位参考（pilot 前必须填完） |
| `docs/paper_outline.md` | 论文骨架、图表预算、实验到图表的映射 |
| `experiments/registry.csv` | **实验登记表**：每个实验的目的、配置、状态、结果文件 |
| `experiments/runs.csv` | **运行记录**：每次运行的 commit、配置、种子、指标 |
| `research_notes/`、`reports/` | 文献调研笔记与报告 |
| `data/` | 本地原始 CSV（已 gitignore，**永不提交**） |
| `outputs/` | 逐样本预测与指标文件（已 gitignore；冻结版本另行归档） |

## 红线

1. **按试品或实验批次分组切分**，绝不按窗口随机切分。少样本抽样只能从训练组的试品里抽。
2. 论文里的每个数字都必须能追溯到 `experiments/runs.csv` 中的一行，以及一份逐样本预测文件。
3. 强基线不得缺席：统计特征 + SVM/RF、PRPD + CNN、1D-ResNet、InceptionTime、MiniRocket、未加锚定的原始 TSFM、同架构随机初始化。
4. 原始实验数据不进 git。
