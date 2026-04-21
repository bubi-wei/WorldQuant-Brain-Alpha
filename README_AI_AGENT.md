# AI Alpha Strategy Generator

> LLM-Agent + RAG 驱动的 WorldQuant Brain Alpha 智能生成系统  
> **双轨架构：自由探索（Explorer）+ 骨架复用（Skeleton）并行**

---

## 核心设计理念

系统采用"熵减"思想：

- **Explorer 轨道**（熵增）：LLM 自由探索新的因子假设，生成全新表达式结构
- **Skeleton 轨道**（熵减）：从历史成功 alpha 中提取算子结构骨架，只替换字段/参数，成功概率更高
- **TrackBandit**：UCB1 算法动态平衡两条轨道的回测配额，自适应收敛到更高效的路径

> **核心洞察**：如果一个 alpha 表达式能够提交成功，保留其算子结构（骨架）只替换数据字段，成功概率显著高于完全随机探索。

---

## 架构总览

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         TrackBandit (UCB1 配额分配)                          │
│                  explorer_ratio ←动态→ skeleton_ratio                        │
└──────────────────┬───────────────────────────────┬───────────────────────────┘
                   │                               │
        ┌──────────▼──────────┐         ┌──────────▼──────────┐
        │   Explorer 轨道      │         │   Skeleton 轨道      │
        │                     │         │                     │
        │  IdeaAgent          │         │  SkeletonRegistry   │
        │  (RAG + LLM 假设)   │         │  (UCB 种子选择)     │
        │        ↓            │         │        ↓            │
        │  ExprSynthAgent     │         │  SkeletonAgent      │
        │  (LLM 合成, T=0.9)  │         │  (RAG 字段选择)     │
        │        ↓            │         │  (LLM 填充, T=0.3)  │
        │  Novelty(strict)    │         │        ↓            │
        │  embedding 过滤      │         │  Novelty(field_cov) │
        │                     │         │  字段组合去重        │
        └──────────┬──────────┘         └──────────┬──────────┘
                   └───────────┬───────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  WQBClient (async)  │  httpx + asyncio + Semaphore
                    │  /simulations 并发  │  Retry-After 轮询 + 401 重认证
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Qualifier          │  指标阈值硬筛
                    │  qualified / soft   │  (sharpe/fitness/turnover/ic)
                    └──────────┬──────────┘
                               │
               ┌───────────────┼───────────────┐
               │               │               │
    ┌──────────▼──────┐  ┌─────▼─────┐  ┌─────▼────────────┐
    │  AlphaMemory    │  │ Reflector │  │ SkeletonExtractor │
    │  (DuckDB)       │  │ (LLM 反思)│  │ AST → 骨架模板   │
    │  skeleton_id列  │  │           │  │ 写回 Registry    │
    └─────────────────┘  └───────────┘  └──────────────────┘
```

---

## 骨架系统详解

### 什么是骨架 (Skeleton)？

骨架是 FASTEXPR 表达式的**算子结构模板**，将具体字段和参数替换为占位符：

```
原表达式:   group_rank(ts_std_dev(returns, 20) / ts_mean(volume, 20), subindustry)
                    ↓ SkeletonExtractor
骨架模板:   group_rank(ts_std_dev($X1, $W1) / ts_mean($X2, $W1), $G1)

占位符说明:
  $X1, $X2  → 数据字段 (field slots)，附带 LLM 生成的语义提示
  $W1, $W2  → 数值参数 (param slots)，记录类型/范围/历史值
  $G1, $G2  → 分组枚举 (group slots)，如 subindustry/industry/sector
```

**同构判定**：两个表达式经 AST 规范化后模板字符串完全一致 → 同一骨架，只更新统计，不重复插入。

**同值复用**：`20` 出现两次 → 同一 `$W1`（保留"两边使用同等窗口"的结构语义）。

### 骨架库数据结构

```sql
CREATE TABLE skeletons (
    skeleton_id        VARCHAR PK,
    template_str       VARCHAR,           -- group_rank(ts_std_dev($X1,$W1)/$X2,$G1)
    operators_used     VARCHAR[],         -- ['group_rank','ts_std_dev','ts_mean']
    field_slots_json   VARCHAR,           -- [{name:$X1, semantic_hint:'momentum signal'}]
    param_slots_json   VARCHAR,           -- [{name:$W1, type:int_window, range:[5,252]}]
    group_slots_json   VARCHAR,           -- [{name:$G1, candidates:['subindustry',..]}]
    attempt_count      INTEGER,           -- 跑了多少次
    success_count      INTEGER,           -- 通过次数
    soft_success_count INTEGER,           -- 软通过次数
    avg_sharpe         FLOAT,             -- 平均 Sharpe（通过的变体）
    archived           BOOLEAN,           -- 尝试 >20 次且成功率 <5% → 自动冷宫
    created_at         TIMESTAMP
);
```

### SkeletonAgent 工作流

```
1. SkeletonRegistry.pick_seeds(strategy='ucb')
   → UCB1 优先选择"成功率高 + 被用少"的骨架

2. 对每个骨架:
   - 读 field_slots.semantic_hint
   - VectorStore.query('datafields', hint) → top-15 候选字段
   - 已见 combo 从 skeleton_instances 表取出（避免重复）
   - LLM (T=0.3) 从候选中选最合适的 5-10 个组合

3. SkeletonExtractor.instantiate(template, field_map, param_map)
   → 生成候选表达式

4. ExprValidator → 过滤语法错误
5. field_coverage_novelty → 过滤已见过的字段组合
```

---

## 与原架构的核心对比

| 维度 | 原项目 (`brain_batch_alpha.py`) | v1 (单轨 Explorer) | v2 (双轨 + Skeleton) |
|---|---|---|---|
| 策略生成 | 硬编码 f-string 模板 | LLM 自由合成 | Explorer + Skeleton 双轨并行 |
| 熵控制 | 无 | 无（全熵增） | TrackBandit 动态平衡熵增/熵减 |
| 结构复用 | 人工模板 | 无 | 自动提取骨架 + 受控变异 |
| 字段语义 | 硬判断字段名 | Chroma RAG | Chroma RAG + slot semantic_hint |
| 新颖性过滤 | 无 | embedding 相似度 | strict (Explorer) + field_coverage (Skeleton) |
| 配额分配 | 串行固定 | 固定每轮 | UCB1 Bandit 动态分配 |
| 学习路径 | 无 | 失败反思 RAG | 失败反思 + 骨架成功率更新 |
| 冷启动 | 硬编码模板即为 | 无法热启动 | skeleton_seeds.yaml 10 条预置骨架 |

---

## 快速上手

### 1. 安装依赖

```bash
# 推荐 Python 3.11
python -m venv myenv && source myenv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少填写：
```ini
WQB_USERNAME=your@email.com
WQB_PASSWORD=your_password
LLM_MODEL=openai/gpt-4o
OPENAI_API_KEY=sk-...

# 骨架系统配置（可选，有默认值）
TRACK_EXPLORER_RATIO=0.3      # 初始给 Explorer 的配额比例
SKELETON_MIN_SEEDS=3          # 骨架库 <3 条时 Skeleton 臂不启用
SKELETON_VARIANTS_PER_SEED=5  # 每骨架每轮生成变体数
SOFT_ENABLE_SKELETON=true     # 软合格 alpha 也沉淀为骨架
SKELETON_PICK_STRATEGY=ucb    # ucb | top_sharpe | recent
EXPLORER_FLOOR=0.1            # Explorer 最低保底配额比例
```

### 3. 初始化知识库（首次运行）

```
notebooks/
  01_setup_and_field_semantics.ipynb   ← 拉取字段 + LLM 语义描述 + 写 Chroma
  02_build_operator_and_paper_kb.ipynb ← 算子/研报入库
```

### 4. 加载冷启动骨架（可选但推荐）

在 `06_skeleton_lab.ipynb` 中运行"Load cold-start seeds"部分，
将 `alpha_agent/data/assets/skeleton_seeds.yaml` 中的 10 条预置骨架导入注册表。

### 5. 运行双轨闭环

```
notebooks/
  03_idea_to_expression_demo.ipynb     ← 不调 WQB，调试 LLM 质量
  04_closed_loop_generation.ipynb      ← 完整双轨闭环
  05_results_and_novelty_analysis.ipynb← 结果分析 + 骨架家族树可视化
  06_skeleton_lab.ipynb                ← 骨架提取/实例化/调试
```

### 6. 代码中直接使用

```python
import asyncio
from alpha_agent.data.wqb_client import WQBClient
from alpha_agent.knowledge.vector_store import VectorStore
from alpha_agent.knowledge.alpha_memory import AlphaMemory
from alpha_agent.knowledge.skeleton_registry import SkeletonRegistry
from alpha_agent.data.operator_kb import OperatorKB
from alpha_agent.engine.orchestrator import Orchestrator

async def main():
    store    = VectorStore()
    memory   = AlphaMemory()
    registry = SkeletonRegistry()
    kb       = OperatorKB()

    async with WQBClient() as client:
        orch = Orchestrator(
            client=client,
            vector_store=store,
            alpha_memory=memory,
            skeleton_registry=registry,
            operator_kb=kb,
            auto_submit=False,
        )
        reports = await orch.run(
            direction="earnings quality and analyst revision momentum",
            dataset="fundamental6",
            universe="TOP3000",
            n_rounds=5,
            ideas_per_round=3,
            variants_per_idea=3,
            track_mode="hybrid",      # "explorer_only" | "skeleton_only" | "hybrid"
        )

asyncio.run(main())
```

---

## 目录结构

```
alpha_agent/
├── config.py                     # pydantic-settings，读 .env
├── data/
│   ├── wqb_client.py             # 异步 WQB API 客户端
│   ├── datafield_indexer.py      # 字段拉取 + LLM 语义 + Chroma 写入
│   ├── operator_kb.py            # FASTEXPR 算子加载器
│   └── assets/
│       ├── operators.yaml        # 算子知识库 (45+ 算子)
│       └── skeleton_seeds.yaml   # 10 条冷启动骨架模板 ← NEW
├── knowledge/
│   ├── vector_store.py           # Chroma 封装 (fields/operators/papers)
│   ├── paper_ingest.py           # PDF/MD/TXT 文档分块入库
│   ├── alpha_memory.py           # DuckDB 历史记录 (含 skeleton_id 列) ← UPDATED
│   └── skeleton_registry.py     # DuckDB 骨架表 + UCB 种子选择 ← NEW
├── engine/
│   ├── idea_agent.py             # RAG + LLM → 因子假设 (中性提示) ← UPDATED
│   ├── expr_synth_agent.py       # 假设 → FASTEXPR 变体 (鼓励骨架移植) ← UPDATED
│   ├── validator.py              # 本地语法预检 + extract_ast() ← UPDATED
│   ├── skeleton_extractor.py     # AST 规范化 + 占位符分配 + LLM 注释 ← NEW
│   ├── skeleton_agent.py         # 骨架内变异 Agent (T=0.3) ← NEW
│   ├── reflector.py              # 失败反思 → 改进建议
│   └── orchestrator.py           # 双轨主循环协调器 ← UPDATED
├── search/
│   ├── novelty.py                # 新颖性评分 (strict / field_coverage 双模式) ← UPDATED
│   ├── bandit.py                 # UCB1 研究方向选择
│   └── track_bandit.py           # Explorer vs Skeleton 两臂 UCB ← NEW
└── eval/
    └── qualifier.py              # 指标阈值筛选 (含 soft_qualified)

notebooks/
├── 01_setup_and_field_semantics.ipynb
├── 02_build_operator_and_paper_kb.ipynb
├── 03_idea_to_expression_demo.ipynb
├── 04_closed_loop_generation.ipynb      ← track_mode 开关 + 骨架输出演示 UPDATED
├── 05_results_and_novelty_analysis.ipynb← 骨架家族树 + 轨道对比曲线 UPDATED
└── 06_skeleton_lab.ipynb               ← 骨架提取/实例化/调试 NEW

tests/
└── test_skeleton_extractor.py          ← 32 个单测 (AST/同构/实例化/回归) NEW
```

---

## 可配置项 (`.env`)

### 基础配置

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_MODEL` | `openai/gpt-4o` | litellm 格式，支持任何 LLM |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | 本地 sentence-transformer |
| `WQB_CONCURRENCY` | `3` | 并发回测数，建议 2-4 |
| `IDEAS_PER_ROUND` | `5` | Explorer 轨道每轮假设数 |
| `VARIANTS_PER_IDEA` | `3` | 每假设生成变体数 |
| `MAX_ROUNDS` | `10` | 最大轮数 |
| `NOVELTY_SCORE_MIN` | `0.3` | Explorer 新颖性阈值 |
| `QUAL_SHARPE_MIN` | `1.5` | Sharpe 入门阈值 |
| `QUAL_FITNESS_MIN` | `1.0` | Fitness 入门阈值 |
| `QUAL_TURNOVER_MIN/MAX` | `0.1/0.9` | Turnover 区间 |
| `QUAL_IC_MEAN_MIN` | `0.02` | IC Mean 阈值 |

### 骨架系统配置（v2 新增）

| 变量 | 默认 | 说明 |
|---|---|---|
| `TRACK_EXPLORER_RATIO` | `0.3` | 初始 Explorer 配额占比，Bandit 自适应调整 |
| `SKELETON_MIN_SEEDS` | `3` | 骨架库数量低于此值时 Skeleton 臂不启用 |
| `SKELETON_VARIANTS_PER_SEED` | `5` | 每骨架每轮生成变体数 |
| `SOFT_ENABLE_SKELETON` | `true` | 软合格 alpha 也沉淀为骨架候选 |
| `SKELETON_PICK_STRATEGY` | `ucb` | 种子选择策略：`ucb` / `top_sharpe` / `recent` |
| `EXPLORER_FLOOR` | `0.1` | Explorer 最低保底比例（防 Skeleton 垄断） |

---

## 骨架冷启动

系统预置了 10 条从原项目 `alpha_strategy.py` 提炼的成熟骨架：

| 骨架模板 | 经济含义 |
|---|---|
| `group_rank(($X1 - $X2) / $X2, $G1)` | 日内/隔夜收益截面排名 |
| `group_rank(ts_std_dev($X1, $W1), $G1)` | 滚动波动率截面排名 |
| `group_rank(ts_std_dev($X1,$W1) / ts_mean($X2,$W2), $G1)` | 相对波动效率 |
| `ts_corr($X1, abs($X2), $W1)` | 量价相关性（价格冲击） |
| `group_rank(($X1 - ts_mean($X1,$W1)) / ts_std_dev($X1,$W1), $G1)` | 成交量 Z 分 |
| `trade_when(ts_rank(ts_std_dev($X1,$W1),$W2) < $W3, $X2, -1)` | 低波动状态过滤 |
| `group_rank(ts_corr($X1, $X2, $W1), $G1)` | 滚动相关性截面排名 |
| `regression_neut($X1, ts_std_dev($X2, $W1))` | 波动率中性化因子 |
| `if_else(rank($X1) > $W1, $X2, -1 * $X2)` | 条件方向翻转 |
| `group_neutralize(rank($X1), bucket(rank(cap), range='0.1,1,0.1'))` | 规模中性化 |

导入方式（在 `06_skeleton_lab.ipynb` 中）：
```python
import yaml
from alpha_agent.knowledge.skeleton_registry import SkeletonRegistry

registry = SkeletonRegistry()
with open("alpha_agent/data/assets/skeleton_seeds.yaml") as f:
    seeds = yaml.safe_load(f)["skeletons"]

for seed in seeds:
    registry.upsert(
        template_str=seed["template_str"],
        operators_used=seed.get("operators_used", []),
        field_slots=seed.get("field_slots", []),
        param_slots=seed.get("param_slots", []),
        group_slots=seed.get("group_slots", []),
        origin_hypothesis=seed.get("origin_hypothesis", ""),
    )
print(registry.stats())
```

---

## 扩展知识库

### 添加研究论文

```python
from alpha_agent.knowledge.paper_ingest import PaperIngest
from alpha_agent.knowledge.vector_store import VectorStore

pi = PaperIngest(VectorStore())
pi.ingest_file("papers/your_paper.pdf", source="YourPaper2024")
pi.ingest_directory("papers/")
```

### 自定义研究方向

```python
from alpha_agent.search.bandit import DirectionBandit

bandit = DirectionBandit(memory, directions=[
    "your custom direction 1",
    "your custom direction 2",
])
best = bandit.select()
```

### 手动添加骨架

```python
from alpha_agent.knowledge.skeleton_registry import SkeletonRegistry

registry = SkeletonRegistry()
sk_id = registry.upsert(
    template_str="ts_rank(ts_mean($X1, $W1), $W2)",
    operators_used=["ts_rank", "ts_mean"],
    field_slots=[{"name": "$X1", "literal": "?", "semantic_hint": "financial signal"}],
    param_slots=[
        {"name": "$W1", "type": "int_window", "range": [5, 60], "seen": [10]},
        {"name": "$W2", "type": "int_window", "range": [60, 252], "seen": [252]},
    ],
    origin_hypothesis="时序排名的滚动均值 — 捕捉均值回归速度",
)
print(f"Added skeleton: {sk_id}")
```

---

## 运行测试

```bash
source myenv/bin/activate
pytest tests/test_skeleton_extractor.py -v
# 32 passed in 0.35s
```

测试覆盖：
- `TestExtractAst`：基础 AST 解析、空输入、语法错误处理
- `TestExtractSkeleton`：字段/参数/分组占位符分配、算子捕获、同值复用
- `TestIsomorphism`：同结构不同字段/参数/分组 → 相同骨架；不同结构 → 不同骨架
- `TestInstantiation`：基础/分组/复合实例化、提取后回程还原
- `TestClassifyNumber`：整数窗口/分位数/浮点阈值分类
- `TestParamSlots`：历史值记录、同值复用验证
- `test_known_templates`：4 个典型表达式回归

---

## 已知局限与折中

| 问题 | 当前处理 |
|---|---|
| **AST 交换律**：`a*b` 与 `b*a` 被视为不同骨架 | 保守策略，不做规范化；如重复过多后续加 normalizer |
| **LLM 选字段偏保守**：可能反复选同一字段对 | `seen_field_combos` 显式告知已见过的组合，强制覆盖度 |
| **Explorer 被饿死**：Skeleton 轨道表现好时 | `EXPLORER_FLOOR=0.1` 保证至少 10% 配额 |
| **骨架库污染**：低质骨架占用资源 | 尝试 >20 次且成功率 <5% 自动 `archived=TRUE` |
| **WQB 配额消耗** | Validator + NoveltyScorer 前置减少无效调用 |
| **LLM 幻觉算子** | Validator 预检，不合格触发 LLM 重试（最多 3 次） |
| **首次 Chroma 构建慢** | 字段语义依赖 LLM API，TOP3000 首次建库需时间 |

---

## 技术栈

| 组件 | 库 |
|---|---|
| LLM | `litellm` (OpenAI / Anthropic / DeepSeek / 本地) |
| 向量检索 | `chromadb` + `sentence-transformers` |
| 持久存储 | `duckdb` |
| 异步 HTTP | `httpx[http2]` |
| 语法解析 | `lark` |
| 配置管理 | `pydantic-settings` |
| 日志/进度 | `rich` |
| 重试 | `tenacity` |
| 测试 | `pytest` |
