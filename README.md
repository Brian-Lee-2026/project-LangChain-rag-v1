# 具身机器人知识问答系统

一个面向“具身机器人使用场景与未来需求”主题的知识问答系统，采用 `FastAPI + LangChain + Prompt + RAG + DeepSeek` 构建，前端提供简洁的浏览器测试窗口，便于后端联调和快速演示。

## 项目特点

- 规范的分层结构：`api / core / services / models / static / tests`
- FastAPI 后端，适合作为后续微服务或企业内部知识平台的基础骨架
- LangChain RAG 链路，包含知识库加载、分块、向量检索、Prompt 编排
- 使用持久化 `Chroma` 向量库，本地重启后可直接复用索引
- DeepSeek 负责答案生成，支持通过环境变量切换模型
- 结构化日志与反馈日志，便于排查错误、追踪问答准确率
- 版本控制友好：补齐 `.gitignore`、环境变量样例、测试与说明文档

## 目录结构

```text
.
├── app
│   ├── api
│   ├── core
│   ├── models
│   ├── services
│   ├── static
│   └── main.py
├── data
│   └── knowledge
├── logs
├── scripts
├── tests
├── .env.example
├── .gitignore
├── Makefile
└── pyproject.toml
```

## 技术栈列表

### 1. 后端基础框架

- `Python 3.11+`：项目主语言，当前推荐使用 Python 3.12 或 3.13 进行开发与部署
- `FastAPI`：提供 REST API、依赖注入、请求校验和应用生命周期管理
- `Uvicorn`：作为 ASGI 服务启动器，支持开发热重载和生产多进程 worker
- `Pydantic`：定义请求/响应数据模型，保证接口输入输出结构清晰
- `pydantic-settings`：统一读取 `.env` 配置，管理运行环境、模型参数和路径配置

### 2. 大模型与 RAG 链路

- `LangChain`：负责组织问答链路，串联检索、提示词和模型调用
- `langchain-deepseek`：对接 `DeepSeek` 对话模型，完成最终答案生成
- `langchain-text-splitters`：对知识文档做分块切片，便于后续向量化检索
- `langchain-chroma` + `Chroma`：作为持久化向量库，负责文档向量存储与相似度检索
- `LangChain PromptTemplate`：统一管理 `SYSTEM_PROMPT` 和 `USER_TEMPLATE`

### 3. 向量化与知识库处理

- `langchain-huggingface`：可选接入 HuggingFace 嵌入模型
- `sentence-transformers`：加载中文嵌入模型，如 `BAAI/bge-small-zh-v1.5`
- `NumPy`：用于本地轻量哈希向量的数值计算
- `Markdown / TXT`：作为知识库原始文档格式，存放在 `data/knowledge/`
- `本地哈希嵌入回退策略`：当 HuggingFace 模型不可用时，自动回退到 `local_hash`

### 4. 可观测性与工程化

- `logging`：统一应用日志、错误日志和控制台输出格式
- `JSONL`：记录 RAG 审计日志和人工反馈日志，便于后续准确率统计与问题追踪
- `RotatingFileHandler`：按文件大小轮转应用日志，避免单个日志文件无限增长
- `Makefile`：统一项目安装、启动、测试和索引重建命令
- `Ruff`：代码静态检查与规范约束
- `Pytest`：接口、配置、服务逻辑等单元测试
- `setuptools editable install`：通过 `pip install -e ...` 支持本地开发模式

### 5. 前端与交互

- `FastAPI StaticFiles`：提供静态测试页面资源
- `HTML + JavaScript`：构建轻量测试窗口，验证问答、健康检查和反馈流程

## 核心设计逻辑

### 1. 分层架构

项目按 `api / core / services / models / static / tests` 分层：

- `api`：暴露 HTTP 接口，如聊天、健康检查、反馈统计
- `core`：处理配置、日志等基础设施能力
- `services`：承载知识库加载、向量检索、问答生成、反馈统计等核心业务
- `models`：统一定义请求与响应模型
- `static`：放置前端测试页资源
- `tests`：验证配置、API 和服务行为

这种分层方式的目标是让“接口层、业务层、基础设施层”职责清晰，便于后续继续扩展。

### 2. 启动时一次性初始化核心服务

应用在 `lifespan` 生命周期中完成以下初始化：

- 加载配置并校验目录
- 配置日志系统
- 初始化 `RAGService`
- 初始化 `FeedbackService`

这样做的好处是避免在每个请求里重复创建模型、向量库和服务对象，减少重复开销。

### 3. RAG 主链路设计

当前问答链路采用“检索增强生成”模式，核心流程如下：

1. 从 `data/knowledge/` 读取 Markdown/TXT 知识文档
2. 将文档转换为 LangChain `Document`
3. 按配置进行文本切片
4. 使用嵌入模型将切片转成向量
5. 将向量持久化到 `Chroma`
6. 用户提问时先做相似度检索
7. 将问题、历史对话和检索结果填充进 Prompt
8. 调用 `DeepSeek` 生成最终答案
9. 返回答案、引用来源和请求耗时

这保证了模型回答尽量基于知识库资料，而不是完全依赖模型自身记忆。

### 4. 检索与索引复用设计

项目没有使用临时内存向量库，而是使用本地持久化 `Chroma`，并额外设计了索引清单机制：

- 记录知识内容摘要、切片参数、嵌入模型信息和集合名
- 启动时校验当前知识库与已持久化索引是否一致
- 一致时直接复用索引
- 不一致时自动重建索引

这样做的价值是：

- 服务重启后无需每次重新嵌入全部文档
- 知识库更新后又能自动发现变化并重建
- 更适合本地开发和小型部署场景

### 5. Prompt 设计逻辑

Prompt 拆分为两部分：

- `SYSTEM_PROMPT`：约束回答原则、领域范围和输出要求
- `USER_TEMPLATE`：填充问题、历史对话、检索内容和提示词版本

这种设计有几个优点：

- 规则与数据分离，便于单独调整提示词
- 支持对 Prompt 版本进行审计和回溯
- 能把检索内容、用户问题和历史对话统一组织成结构化输入

### 6. 配置驱动设计

项目通过 `.env` + `Settings` 管理关键参数，包括：

- 运行环境
- 端口与 worker
- 模型参数
- 检索参数
- 向量库目录
- 嵌入策略
- 日志路径

这样做可以让“代码逻辑”和“部署参数”解耦，方便开发、测试和生产切换。

### 7. 日志与可追溯性设计

项目同时保留三类追踪能力：

- 应用日志：记录启动、异常和运行状态
- RAG 审计日志：记录问题、答案、来源、耗时和错误
- 人工反馈日志：记录回答是否准确、评分和备注

另外，每个请求会补充 `request_id`，用于串联：

- 前端请求
- 应用日志
- RAG 审计记录
- 错误排查过程

这为后续做准确率统计、问题回放和线上排障提供了基础。

### 8. 开发与生产分离设计

项目通过 `Makefile` 区分开发和生产模式：

- `make run`：开发模式，开启热重载，方便调试
- `make run-prod`：生产模式，关闭热重载，启用多 worker

同时日志输出也会跟随环境变化：

- 开发环境优先控制台可读性
- 生产环境优先结构化 JSON 日志

这种设计让项目在“本地演示”和“正式部署”之间切换成本更低。

## 快速开始

### 1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,hf]"
```

建议优先使用 Python 3.12 或 3.13。当前仓库在 Python 3.14 下可以完成基础测试，但 `LangChain` 上游依赖会给出兼容性警告，因此更推荐用 3.12/3.13 做开发和部署。

当前项目默认使用 HuggingFace 嵌入模型：

```bash
pip install -e ".[dev,hf]"
```

如果你只想做一个不下载 HuggingFace 模型的轻量本地演示，也可以安装：

```bash
pip install -e ".[dev]"
```

然后把 `.env` 里的 `EMBEDDING_STRATEGY` 改回 `local_hash`。

### 2. 配置环境变量

```bash
cp .env.example .env
```

至少需要配置：

- `DEEPSEEK_API_KEY`

默认还额外配置了这几个运行参数：

- `APP_ENV=development`：仅在你不通过 Makefile 启动时作为默认环境值
- `EMBEDDING_CACHE_DIR=data/model_cache`：HuggingFace 模型本地缓存目录
- `VECTOR_STORE_DIR=data/vector_store`：持久化向量库目录
- `VECTOR_STORE_COLLECTION_NAME=embodied_robot_knowledge`：Chroma 集合名称
- `EMBEDDING_DEVICE=cpu`：嵌入模型运行设备，单机默认 `cpu`
- `EMBEDDING_HF_LOCAL_FILES_ONLY=false`：是否只使用本地缓存，不访问 HuggingFace 网络
- `EMBEDDING_FALLBACK_TO_LOCAL_HASH=true`：当 HuggingFace 拉取失败时自动回退到轻量本地向量
- `APP_WORKERS=2`：生产模式下的 FastAPI worker 数量

这些目录类配置如果写成相对路径，会自动按项目根目录解析。

### 3. 启动服务

```bash
make run
```

首次使用 HuggingFace 嵌入时，会把模型下载到 `EMBEDDING_CACHE_DIR` 指定的目录。后续只要缓存还在，就不会重复下载。

如果当前网络访问 HuggingFace 不稳定，系统会按 `EMBEDDING_FALLBACK_TO_LOCAL_HASH` 自动回退到 `local_hash` 嵌入，并在当前环境对应的 `logs/<environment>/app.log` 中记录告警，保证服务先可用。

如果模型已经完整缓存，并且你希望生产环境完全离线启动，可以把 `EMBEDDING_HF_LOCAL_FILES_ONLY=true`。这样服务会只读本地缓存，不再请求 HuggingFace。

启动后访问：

- 前端测试页：`http://127.0.0.1:8000/`
- 健康检查：`http://127.0.0.1:8000/api/v1/health`
- 反馈摘要：`http://127.0.0.1:8000/api/v1/feedback/summary`

### 4. 生产启动建议

开发联调用：

```bash
make run
```

生产或准生产环境建议用：

```bash
make run-prod
```

`run-prod` 会关闭热重载，并按 `APP_WORKERS` 启动多进程 worker，更适合稳定运行。默认命令等价于：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2 --proxy-headers
```

如果你后面把服务放到 Nginx 或网关后面，这种方式会更稳妥。

### 5. APP_ENV 说明

为了避免歧义，当前项目对 `APP_ENV` 的处理规则是：

- 如果你使用 `make run`，启动脚本会把环境强制设为 `development`
- 如果你使用 `make run-prod`，启动脚本会把环境强制设为 `production`
- 只有你绕过 Makefile，直接执行 `uvicorn app.main:app ...` 或自定义脚本时，`.env` 里的 `APP_ENV` 才会作为默认值生效

也就是说，在标准用法下，`APP_ENV` 更像是“兜底配置”或“说明性配置”，真正决定环境的是启动命令本身。

## 知识库维护

默认知识文件放在 `data/knowledge/` 目录下，建议使用 Markdown 文档，按主题拆分：

- 一个文件一个明确主题
- 一级标题写清场景名称
- 内容尽量包含场景、价值、约束、未来需求、风险与建议

新增或修改知识库后，服务下次启动时会校验持久化索引清单；如果发现知识内容、分块参数、嵌入配置或索引数量发生变化，会自动重建 `Chroma` 索引。

首次启动会构建 `data/vector_store/` 下的持久化向量库，后续重启会优先复用，不再每次都重新嵌入全部文档。

如果你想手动强制重建索引，可以执行：

```bash
make check-kb
```

如果只是想快速清空索引，也可以直接删除 `data/vector_store/` 目录，服务会在下次启动时重新构建。

## 日志与准确率追踪

系统默认生成两类追踪信息：

- `logs/development/app.log` 或 `logs/production/app.log`：应用运行日志与错误日志，支持多 worker 安全轮转
- `logs/development/telemetry.sqlite3` 或 `logs/production/telemetry.sqlite3`：统一保存 `rag_events` 和 `feedback_entries` 表的 SQLite 数据库

前端测试页内置“准确 / 不准确”反馈按钮，会把标注结果写入 SQLite，并在 `/api/v1/feedback/summary` 中输出简要统计。

日志系统会自动跟随当前启动模式：

- `make run`：把环境设为 `development`，控制台日志采用更适合本地调试的人类可读格式
- `make run-prod`：把环境设为 `production`，控制台日志采用更适合采集和分析的 JSON 结构化格式

无论哪种模式，文件日志都会写入对应环境目录，避免开发日志和生产日志混在一起。

## 推荐后续扩展

- 对接企业自有文档、行业报告、访谈纪要、研究论文
- 切换到更强的中文嵌入模型以提升检索质量
- 引入离线评测集，对场景分类、需求归纳、事实问答做基准评估
- 接入数据库与 LangSmith，形成更完整的线上可观测性
