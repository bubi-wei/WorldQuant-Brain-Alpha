# WorldQuant-Brain-Alpha 常规操作手册

本文档用于日常运行、回测、排障与骨架管理。

## 1) 环境准备

在项目根目录执行：

```bash
source myenv/bin/activate
python --version
```

建议始终使用 `myenv`，避免依赖版本混乱。

## 2) 推荐运行顺序

Notebook 推荐顺序：

1. `notebooks/01_setup_and_field_semantics.ipynb`
2. `notebooks/02_build_operator_and_paper_kb.ipynb`
3. `notebooks/06_skeleton_lab.ipynb`（可选）
4. `notebooks/03_idea_to_expression_demo.ipynb`
5. `notebooks/04_closed_loop_generation.ipynb`
6. `notebooks/05_results_and_novelty_analysis.ipynb`

## 3) 先探索，再加骨架（当前策略）

当前策略是先让 AI 做探索，暂不使用初始化骨架。

- `alpha_agent/data/assets/skeleton_seeds.yaml` 目前为 `skeletons: []`
- `data/skeleton_registry.db` 的 `skeletons` / `skeleton_instances` 已清空

如果你想继续纯探索，建议在 `notebooks/04_closed_loop_generation.ipynb` 中设置：

```python
TRACK_MODE = "explorer_only"
```

## 4) 清空骨架表（重置）

在项目根目录执行：

```bash
source myenv/bin/activate
python -c "import duckdb; c=duckdb.connect('data/skeleton_registry.db'); c.execute('DELETE FROM skeleton_instances'); c.execute('DELETE FROM skeletons'); print('done'); c.close()"
```

## 5) 后续添加新骨架（有好模板后）

### 5.1 编辑种子文件

在 `alpha_agent/data/assets/skeleton_seeds.yaml` 中补充模板，格式示例：

```yaml
skeletons:
  - template_str: "group_rank(ts_rank($X1 / $X2, $W1), $G1)"
    operators_used: [group_rank, ts_rank]
    field_slots:
      - name: "$X1"
        semantic_hint: "numerator signal"
      - name: "$X2"
        semantic_hint: "denominator signal"
    param_slots:
      - name: "$W1"
        type: int_window
        range: [20, 120]
        seen: [60]
    group_slots:
      - name: "$G1"
        candidates: [industry, subindustry, sector]
    origin_hypothesis: "example"
```

### 5.2 导入到注册表

```bash
source myenv/bin/activate
python -c "import yaml, pathlib; from alpha_agent.knowledge.skeleton_registry import SkeletonRegistry; reg=SkeletonRegistry(db_path=pathlib.Path('data/skeleton_registry.db')); seeds=yaml.safe_load(pathlib.Path('alpha_agent/data/assets/skeleton_seeds.yaml').read_text())['skeletons']; [reg.upsert(template_str=s['template_str'], operators_used=s.get('operators_used',[]), field_slots=s.get('field_slots',[]), param_slots=s.get('param_slots',[]), group_slots=s.get('group_slots',[]), origin_hypothesis=s.get('origin_hypothesis','')) for s in seeds]; print(reg.stats())"
```

## 6) 文献入库（提升检索上下文）

将 `.pdf/.md/.txt` 放到 `notebooks/data/papers/`，执行：

```bash
source myenv/bin/activate
python -c "from alpha_agent.knowledge.vector_store import VectorStore; from alpha_agent.knowledge.paper_ingest import PaperIngest; from pathlib import Path; pi=PaperIngest(VectorStore()); print(pi.ingest_directory(Path('notebooks/data/papers')))"
```

## 7) 常用回测模式

在 `notebooks/04_closed_loop_generation.ipynb`：

- `TRACK_MODE = "explorer_only"`：纯探索（推荐冷启动）
- `TRACK_MODE = "hybrid"`：探索 + 骨架并行
- `TRACK_MODE = "skeleton_only"`：只跑骨架变体（需要骨架库已有数据）

并发可通过 `.env` 调整：

```env
WQB_CONCURRENCY=10
```

建议从 10 起步，稳定后再逐步上调。

## 8) 结果查看

主要看两类输出：

- Notebook 输出（round summary、通过率）
- 数据库与文件：
  - `data/alpha_memory.db`
  - `data/skeleton_registry.db`
  - `data/skeleton_only_validation.json`（如使用验证脚本）

## 9) 常见问题排查

### 9.1 DuckDB 锁冲突

症状：报错 `Conflicting lock is held`。

处理：

1. 关闭其他占用同一 DB 的 notebook/kernel
2. 重新运行当前任务

### 9.2 长时间无输出

常见原因：

- WQB 服务端排队
- LLM 返回慢或格式异常

建议：

- 先缩小单次预算（`N_ROUNDS=1`、较小 `IDEAS_PER_ROUND`）
- 先用 `explorer_only` 跑小样本验证链路

## 10) 一条推荐工作流

1. 清空骨架表
2. `explorer_only` 跑 1-2 轮，积累候选
3. 从有效表达式提取骨架并人工筛选
4. 导入骨架后切到 `hybrid`，逐步放量
