# AI Alpha Strategy Generator

> LLM-Agent + RAG 驱动的 WorldQuant Brain Alpha 智能生成系统

---

## 架构概览

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Research Direction (自然语言研究方向)                                     │
└───────────────────────────┬──────────────────────────────────────────────┘
                            │
                  ┌─────────▼──────────┐
                  │    IdeaAgent        │  ← RAG: 字段语义 + 算子 + 论文 + 历史
                  │  (假设生成, LLM)    │        + AlphaMemory 失败案例
                  └─────────┬──────────┘
                            │  N 条结构化因子假设 (JSON)
                  ┌─────────▼──────────┐
                  │  ExprSynthAgent    │  ← few-shot from AlphaMemory
                  │  (表达式合成, LLM)  │    K 个温度采样变体/假设
                  └─────────┬──────────┘
                            │
              ┌─────────────▼────────────────┐
              │       ExprValidator           │  本地 lark 语法校验
              │   (括号/arity/字段存在性)      │  不合格 → 重试 LLM
              └─────────────┬────────────────┘
                            │
              ┌─────────────▼────────────────┐
              │       NoveltyScorer           │  embedding 相似度 + 指标空间
              │   (去除近似重复表达式)         │  过低 → 跳过，节省配额
              └─────────────┬────────────────┘
                            │
              ┌─────────────▼────────────────┐
              │      WQBClient (async)        │  httpx + asyncio + Semaphore
              │   /simulations 并发回测        │  Retry-After 轮询 + 401 重认证
              └─────────────┬────────────────┘
                            │
              ┌─────────────▼────────────────┐
              │      Qualifier / Reflector    │  指标硬筛 → AlphaResult
              │   + LLM 失败原因分析           │  反思写回 AlphaMemory
              └─────────────┬────────────────┘
                            │
              ┌─────────────▼────────────────┐
              │       AlphaMemory (DuckDB)    │  持久化所有历史记录
              │   + VectorStore (Chroma)      │  embedding + 语义检索
              └──────────────────────────────┘
                            │
              ┌─────────────▼────────────────┐
              │      DirectionBandit (UCB1)   │  跨 session 选择最优研究方向
              └──────────────────────────────┘
```

---

## 与原项目的核心对比

| 维度 | 原项目 (brain_batch_alpha.py) | 新系统 (alpha_agent/) |
|---|---|---|
| 策略生成 | 硬编码 f-string 模板枚举 | LLM 基于假设 + RAG 自由合成 |
| 字段理解 | 无语义 (`if field in ['volume']`) | Chroma 向量库 + LLM 描述 |
| 超参数 | 全部魔法数字 | 合成时 LLM 建议区间，可搜索 |
| 反馈 | 无，失败不记录 | 所有结果写 DuckDB，反思 RAG 利用 |
| 新颖性 | 无控制 | embedding 相似度 + 指标空间去重 |
| 并发 | 串行 sleep(5) | httpx asyncio + Semaphore |
| 合法性校验 | 只能靠 WQB 试错 | 本地 lark 语法预检，节省配额 |
| 存储 | alpha_ids.txt | DuckDB (SQL 可查) |
| 方向选择 | 手动 | UCB1 Bandit 自动调度 |
| 知识库 | 无 | 论文/研报/算子文档 Chroma RAG |

---

## 快速上手

### 1. 安装依赖

```bash
pip install -e ".[dev]"
# 或
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少填写：
```
WQB_USERNAME=your@email.com
WQB_PASSWORD=your_password
LLM_MODEL=openai/gpt-4o
OPENAI_API_KEY=sk-...
```

### 3. 按顺序运行 Notebooks

```
notebooks/
  01_setup_and_field_semantics.ipynb   ← 首次运行必须
  02_build_operator_and_paper_kb.ipynb ← 首次运行必须
  03_idea_to_expression_demo.ipynb     ← 不调 WQB，调试 LLM 质量
  04_closed_loop_generation.ipynb      ← 完整闭环，消耗 WQB 配额
  05_results_and_novelty_analysis.ipynb← 结果分析与可视化
```

### 4. 代码中直接使用

```python
import asyncio
from alpha_agent.data.wqb_client import WQBClient
from alpha_agent.knowledge.vector_store import VectorStore
from alpha_agent.knowledge.alpha_memory import AlphaMemory
from alpha_agent.data.operator_kb import OperatorKB
from alpha_agent.engine.orchestrator import Orchestrator

async def main():
    store  = VectorStore()
    memory = AlphaMemory()
    kb     = OperatorKB()

    async with WQBClient() as client:
        orch = Orchestrator(client, store, memory, kb, auto_submit=False)
        reports = await orch.run(
            direction="earnings quality and analyst revision momentum",
            dataset="fundamental6",
            universe="TOP3000",
            n_rounds=3,
            ideas_per_round=5,
            variants_per_idea=3,
        )

asyncio.run(main())
```

---

## 目录结构

```
alpha_agent/
├── config.py                   # pydantic-settings, 读 .env
├── data/
│   ├── wqb_client.py           # 异步 WQB API 客户端
│   ├── datafield_indexer.py    # 字段拉取 + LLM 语义 + Chroma 写入
│   ├── operator_kb.py          # FASTEXPR 算子加载器
│   └── assets/
│       └── operators.yaml      # 算子知识库 (45+ 算子)
├── knowledge/
│   ├── vector_store.py         # Chroma 封装 (fields/operators/papers)
│   ├── paper_ingest.py         # PDF/MD/TXT 文档分块入库
│   └── alpha_memory.py         # DuckDB 历史记录 + 反思存储
├── engine/
│   ├── idea_agent.py           # RAG + LLM → 因子假设
│   ├── expr_synth_agent.py     # 假设 → FASTEXPR 变体
│   ├── validator.py            # 本地语法预检 (lark)
│   ├── reflector.py            # 失败反思 → 改进建议
│   └── orchestrator.py         # 主循环协调器
├── search/
│   ├── novelty.py              # 新颖性评分 (embedding + 指标空间)
│   └── bandit.py               # UCB1 研究方向选择
└── eval/
    └── qualifier.py            # 指标阈值筛选 (可配置)
```

---

## 可配置项 (`.env`)

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_MODEL` | `openai/gpt-4o` | litellm 格式，支持任何 LLM |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | 本地 sentence-transformer |
| `WQB_CONCURRENCY` | `3` | 并发回测数，建议 2-4 |
| `IDEAS_PER_ROUND` | `5` | 每轮生成假设数 |
| `VARIANTS_PER_IDEA` | `3` | 每个假设生成变体数 |
| `MAX_ROUNDS` | `10` | 最大轮数 |
| `NOVELTY_SCORE_MIN` | `0.3` | 新颖性阈值 (0-1) |
| `QUAL_SHARPE_MIN` | `1.5` | Sharpe 入门阈值 |
| `QUAL_FITNESS_MIN` | `1.0` | Fitness 入门阈值 |
| `QUAL_TURNOVER_MIN/MAX` | `0.1/0.9` | Turnover 区间 |
| `QUAL_IC_MEAN_MIN` | `0.02` | IC Mean 阈值 |

---

## 扩展知识库

### 添加研究论文

```python
from alpha_agent.knowledge.paper_ingest import PaperIngest
from alpha_agent.knowledge.vector_store import VectorStore

pi = PaperIngest(VectorStore())
pi.ingest_file("papers/your_paper.pdf", source="YourPaper2024")
# 或批量
pi.ingest_directory("papers/")
```

### 自定义研究方向池

```python
from alpha_agent.search.bandit import DirectionBandit

bandit = DirectionBandit(memory, directions=[
    "your custom direction 1",
    "your custom direction 2",
])
best = bandit.select()
```

---

## 已知局限

- **WQB 配额消耗**: 每次 `/simulations` 调用消耗一个配额。NoveltyScorer 和 Validator 前置减少了无效调用，但仍需控制 `n_rounds × ideas_per_round × variants_per_idea`
- **LLM 幻觉算子**: 尽管 validator 做预检，WQB 支持的 FASTEXPR 语法是超集，部分 LLM 生成的合法语法表达式在平台上仍可能报错
- **首次 Chroma 构建**: 字段语义生成依赖 LLM API，首次对 TOP3000 数据集建库需要一定时间
- **向量库版本漂移**: Chroma 版本升级可能需要重新索引
- **RL/GP 未实装**: 计划接口已预留在 `search/` 和 `engine/orchestrator.py` 中

---

## 技术栈

| 组件 | 库 |
|---|---|
| LLM | `litellm` (OpenAI / Anthropic / 本地) |
| 向量检索 | `chromadb` + `sentence-transformers` |
| 持久存储 | `duckdb` |
| 异步 HTTP | `httpx[http2]` |
| 语法解析 | `lark` |
| 配置 | `pydantic-settings` |
| 日志/进度 | `rich` |
| 重试 | `tenacity` |
