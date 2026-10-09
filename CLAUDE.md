# CLAUDE.md

## 仓库性质

毕业论文用稿的工作区：**基于通用时序预训练模型的小样本局部放电模式识别**。目标是以最小成本、最快速度投出并录用，不追求顶刊。

- 主投 Measurement Science and Technology（IOP），被拒转投 Electric Power Systems Research，Measurement 保底。
- 本篇只做局放；断路器机械特性数据留给第二篇。
- 与 PowerBrain（另一个仓库，TII 重投，截止 2026-11-29）分开推进。

## 先读这些文件

| 文件 | 作用 |
|---|---|
| `docs/plan.md` | **唯一执行底本**：阶段、验收门 G0–G3、任务 T01–T23、待拍板决策 D1–D6、风险 |
| `docs/data_card.md` | 数据事实；没填完不得开始 pilot |
| `experiments/registry.csv` | 实验定义（一个实验对应论文里的一张表或一张图） |
| `experiments/runs.csv` | 运行记录（训练脚本自动追加） |
| `docs/paper_outline.md` | 论文骨架与图表预算 |
| `reports/` | 文献调研报告 |

## 红线

1. **按试品或实验批次分组切分**，绝不按窗口随机切分；少样本抽样只能从训练组的试品里抽。
2. 论文里的数字只能来自 `outputs/frozen/`，由脚本生成，不手抄、不估计、不用占位数字。
3. 强基线不得缺席：统计特征 + SVM/RF、PRPD-CNN、1D-ResNet、InceptionTime、MiniRocket、原始 TSFM、同架构随机初始化。
4. 如果使用 LLM 类时序主干，必须做「去掉 LLM」的消融（对应 Tan et al., NeurIPS 2024）。
5. 原始数据（`data/`）不进 git。
6. 引用写入 bib 前逐篇核对 DOI；`research_notes/` 里标为未核实的条目不能直接引用。

## 叙事定位

不要写成「我们用了 TSFM + LoRA」的模块罗列。核心论点是：局放的判别信息锚定在**工频相位**上，通用 TSFM 学到的是无锚定的时间先验；**物理参考通过输入接口进入，不靠重训主干**，因此标注需求低。

## 算力与数据

- 8 × RTX 4090（每卡 24 GB）。
- 原始数据为 CSV，放在本地 `data/`。

## Commit 约定

使用 `feat:` / `fix:` / `docs:` / `chore:` / `exp:` 前缀，主题行用祈使句、小写短语。`exp:` 用于新增或更新实验结果的记录。
