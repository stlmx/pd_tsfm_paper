# 实验记录规范

## 两张表

| 文件 | 粒度 | 谁写 |
|---|---|---|
| `registry.csv` | **实验**：一个实验回答一个问题，对应论文里的一张表或一张图 | 手工维护，增删实验只改这里 |
| `runs.csv` | **运行**：一次训练或评测，对应一个 seed × shot × 方法组合 | 训练脚本结束时自动追加一行 |

## 规则

1. 每次运行必须记录 `git_commit`。工作区有未提交改动时，脚本拒绝写入 `runs.csv`，避免结果无法复现。
2. `output_dir` 下必须有 `predictions.csv`，至少包含 `sample_id, specimen_id, y_true, y_pred, prob_*` 这些列，以及 `metrics.json` 和 `config.yaml`。
3. `status` 只取 `ok`、`failed`、`superseded` 三个值。重跑的旧结果标为 `superseded`，不删除。
4. 论文里的数字只能取自 `outputs/frozen/` 下的冻结结果（G2 之后），由 `scripts/make_tables.py` 生成。
5. `registry.csv` 中的 `status` 取 `todo`、`running`、`done`、`dropped`，`dropped` 的实验在 `notes` 里写明原因。
6. CSV 字段内不要用英文逗号，多个取值用全角分号「；」分隔。
