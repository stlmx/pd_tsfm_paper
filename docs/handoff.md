# 交接文档：从云端会话迁移到本地

> 2026-10-09 写于云端会话迁移前。本文件是本地接手的唯一入口：读完它，再按 `docs/plan.md` 推进。
> 前情：云端会话无法访问局域网，无法用校园网 IP 下载订阅论文，也接触不到实验室数据与 GPU，故迁回本地。

## 0. 本地第一步

```bash
git clone https://github.com/stlmx/pd_tsfm_paper
cd pd_tsfm_paper

# 两个隔离环境：momentfm 锁定 transformers 4.33.3，与 chronos-forecasting 冲突，必须分开
uv venv -p 3.11 .venv-main   && uv pip install -p .venv-main/bin/python   -r envs/requirements-main.txt
uv venv -p 3.11 .venv-moment && uv pip install -p .venv-moment/bin/python -r envs/requirements-moment.txt
```

`envs/*.txt` 里的 torch 是 CPU 版（云端冒烟测试用）。**本地装 CUDA 版**，其余版本号照抄。

复跑一遍冒烟测试，确认本地环境与云端结论一致：

```bash
.venv-main/bin/python scripts/smoke_backbones.py --models mantis_v1 mantis_v2 chronos_bolt chronos2 minirocket
.venv-moment/bin/python scripts/smoke_backbones.py --models moment_small moment_base moment_large
cd scripts && ../.venv-main/bin/python probe_phase_amplitude.py --models raw mantis_v1 chronos2 minirocket
```

预期结果见 `docs/backbone_validation.md` 第 2 节。若本地数字与之不符，先排查环境，不要直接改文档。

## 1. 这个项目是什么

毕业论文用稿。目标是**以最小成本、最快速度投出并录用**，不追求顶刊。

- 题目（暂定）：*Measurement-Anchored Adaptation of Pretrained Time-Series Foundation Models for Few-Shot Partial Discharge Pattern Recognition*
- 投稿顺序：**MST**（主投）→ **EPSR**（被拒转投）→ **Measurement**（保底）。三刊均在学院「2022 最有国际影响力学术期刊目录」内，序号 706 / 266 / 705。
- 本篇**只做局放**。断路器机械特性数据留给第二篇，复用同一套流程，把相位锚定换成事件锚定（线圈电流的 Latch / Buffer / MCON / ACON / End 特征时刻）。
- 与 PowerBrain（`power_llava_latex` 仓库，TII 重投，截止 2026-11-29）分开推进。

## 2. 核心论点（已被实测修正一次，不要退回旧版）

**现在的说法**：通用时序基础模型（TSFM）的输入约定（定长上下文 + 实例归一化）丢掉了局放测量中两个**被标定的参考量**——工频相位与放电幅值。**测量锚定接口**（按相位分 bin + 幅值回注）把它们补回来；物理参考从输入接口进入，不靠重训主干，因此标注需求低。

**已作废的说法**：~~「通用 TSFM 无法表达相位」~~。合成探针显示，只要窗口对齐相位 0，所有主干都能 100% 区分只差相位的两类。证据见 `docs/backbone_validation.md` 第 3 节。

写作时不要罗列模块（「我们用了 TSFM + LoRA」）——那正是 novelty 攻击的来源。

## 3. 已完成的工作

| 产出 | 文件 | 说明 |
|---|---|---|
| 执行底本 | `docs/plan.md` | 7 阶段、验收门 G0–G3、任务 T01–T23、决策 D1–D6、风险 R1–R6 |
| 实验定义 | `experiments/registry.csv` | 12 项实验，每项对应论文里的一张表或一张图 |
| 运行记录 | `experiments/runs.csv` | 空表，训练脚本结束时自动追加；规范见 `experiments/README.md` |
| 基座选型与实测 | `docs/backbone_validation.md` | CPU 冒烟测试、合成探针、评测清单、环境冲突 |
| 文献调研 | `reports/三刊局放诊断与时序基础模型调研.md` | 约 58 KB；三刊约 50 篇局放 ML 论文、竞品分析、写作模板、范围要求 |
| 调研笔记 | `research_notes/三刊局放诊断与时序基础模型调研/` | 5 份原始笔记，分刊与分主题 |
| 论文骨架 | `docs/paper_outline.md` | 按 MST 风格，7000–8000 词，5–6 图、5–6 表 |
| 决策日志 | `docs/decisions.md` | 已拍板：D5 投稿顺序、本篇范围 |
| 数据卡 | `docs/data_card.md` | **空白模板，等作者填** |

调研的核心结论：三刊中**没有一篇**局放论文用通用 TSFM（仅按标题和元数据检索）。但「小样本局放」很拥挤，西安交大 Wang / Yan / Geng 团队在三刊反复发表，大概率是审稿人，必须引用。最接近的三篇竞品是 EPSR 248:111993、EPSR 254:112685、Measurement 256:118139，区分方式见报告。

## 3b. Pilot 代码已写好（2026-10-09 追加）

`src/pd/` 下的六个模块已实现并测试，70 个测试在两个环境下都通过，全流程已用合成夹具（`scripts/make_synthetic_data.py`）验证跑通。结构与三个内置防护见 `CLAUDE.md` 的「代码结构」一节。

**数据一到，只需要三步**：
1. 按数据卡改 `configs/pilot.yaml` 的 `data` 段（布局、列名、是否需要 `qualify_specimen_with_defect`；`raw_waveform` 还要 `fs` 和 `dead_time_s`）。
2. 跑 `scripts/run_pilot.py`，MOMENT 单独一次调用。
3. 按 G1 判据表读标签效率曲线，拍板 D3。

合成夹具上的数字**不是结果**，不得入论文——它的类别差异是按文献定性设的，不是物理模型。

## 4. 本地接手后的三件事，按这个顺序

### 第一件：填数据卡（阻塞一切）

`docs/data_card.md` 没填完，pilot 无法开始。最关键的三项：

1. **每类缺陷有几个独立试品？** 决定能否按试品分组切分。少于 3 个就要改按实验批次分组，并在风险 R1 中记录。
2. **一个 CSV 对应什么？** 一次采集、一个工频周期、一个脉冲，还是一张 PRPD 图？决定 D1（输入表征）。
3. **有没有同步记录工频相位？** 没有相位参考，锚定接口无从做起，见风险 R2 的退路。

### 第二件：下载 4 篇付费论文（本地才能做）

用校园网下载，放进 `local_materials/`（已 gitignore，不会误传）。需要补的信息是各篇的数据规模、切分方式、基线名单和准确率。

| 期刊 | DOI | 为什么要 |
|---|---|---|
| EPSR 248:111993 | `10.1016/j.epsr.2025.111993` | 最直接的竞品，提出「ViT 硬切 patch 破坏 PRPD 相位连续性」。摘要里没有实验支撑，这正是本文的切入口 |
| EPSR 254:112685 | `10.1016/j.epsr.2025.112685` | 需确认其「相位」是傅里叶谱相位还是工频相位角 |
| Measurement 256:118139 | `10.1016/j.measurement.2025.118139` | GB-SMoE，PRPS 相位级融合 |
| Measurement 255:118054 | `10.1016/j.measurement.2025.118054` | 两处报的准确率不一致（96.91% / 91.28% 与 97.63%），须以原文为准 |

另外两篇**可免费读**，建议先读：
- MST 的 Chang & Boyanapalli 2024（`10.1088/1361-6501/ad366c`）：MST 测量章节的写法模板，11 页、9 图、4 表、21 篇参考文献。
- EPSR 的 Shamsoddini 2025（`10.1016/j.epsr.2025.111601`）：EPSR 章节架构模板，按不同电缆切分训练与测试集。

### 第三件：拍板两项待决决策

| ID | 决策 | 建议 |
|---|---|---|
| D2 | TSFM 主干选哪几个 | **Mantis（主力）+ MOMENT-1-base + Chronos-2**，覆盖对比 / 重建 / 预测三种预训练范式。依据见 `docs/backbone_validation.md` 第 1 节 |
| 新增 | 是否加「泄漏量化」实验 | **建议加**。同一模型分别报告随机窗口切分与按试品切分的结果。三刊中没有局放论文做过，成本几乎为零，又能挡住「数字太好」的质疑。拍板后加入 `registry.csv` |

## 5. 云端做过的事与注意事项

- **基座实测**：7 个主干在 CPU 上全部加载跑通，参数量、嵌入维度、上下文长度、延迟均已实测，解决了文献中的数字冲突。详见 `docs/backbone_validation.md` 第 2 节。
- **两个实测发现**（均来自**合成数据**，必须在真实数据上复核，见任务 P1c）：
  1. MOMENT 与 Chronos 系列对幅值完全不敏感（`cos(x, 10x) = 1.000`，只差幅值的探针精度 0.43–0.54）；Mantis 与 MiniRocket 保留幅值（1.000）。
  2. 窗口对齐相位 0 时，所有主干都能区分只差相位的两类。
- **文献报告的更正**：报告开头有一段「本地实测更正」，指出 Chronos-2 实际有 `embed()` API、参数量冲突已解决、方法定位已扩展为「相位 + 幅值」。报告正文未逐处改写，**以更正段和 `docs/backbone_validation.md` 为准**。
- **一个隐私事项**：某一路调研在首次查 Crossref 时，把作者邮箱放进了一次 API 请求的 `mailto` 参数（Crossref 礼貌池的常规用法），之后的请求均已去掉。无需处理，仅作告知。

## 6. 红线（抄自 `CLAUDE.md`，本地同样适用）

1. **按试品或实验批次分组切分**，绝不按窗口随机切分；少样本的 support 集只能从训练组试品里抽。
2. 论文里的数字只能来自 `outputs/frozen/`，由脚本生成，不手抄、不估计、不用占位数字。
3. 强基线不得缺席：多数类、Gulski 统计算子 + SVM/RF、PRPD-CNN、1D-ResNet、InceptionTime、MiniRocket、原始 TSFM、同架构随机初始化。
4. 若使用 LLM 类时序主干，必须做「去掉 LLM」的消融（Tan et al., NeurIPS 2024）。
5. 原始数据（`data/`）与付费论文（`local_materials/`）不进 git。
6. 引用写入 bib 前逐篇核对 DOI。`research_notes/` 中标为「未核实」「仅题名」的条目不能直接引用——报告末尾列了一批只来自二手渠道的条目，尤其注意。

## 7. 时间线

| 阶段 | 原定时间窗 | 目标 |
|---|---|---|
| P0 立项与数据审计 | 10/12 – 10/18 | 数据卡填完，切分协议定稿 |
| P1 Pilot | 10/19 – 11/01 | **G1 叙事方向拍板** |
| P2 方法实现 | 11/02 – 11/22 | 接口、适配、基线流水线 |
| P3 正式实验 | 11/23 – 12/20 | **G2 结果冻结** |
| P4 写作 | 12/01 – 01/10 | 与 P3 并行 |
| P5 投稿 | 01/11 – 01/24 | **G3 投出 MST** |

**与 PowerBrain 的协调**：11 月以 PowerBrain 重投为主（截止 11/29），本项目 P2 期间只维持 GPU 批量实验，不占写作精力。**如果 PowerBrain 不由你主笔，P2–P5 整体前移约 4 周**，目标 2026-12 下旬投出。
