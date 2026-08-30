# Cyent — 极简编码 Agent

从零自研的编码 Agent：核心能力（主循环、上下文、解析、工具、持久化）全部手写实现，仅以 `openai` SDK 作为模型客户端，可对接任意 OpenAI 兼容端点（官方 API、自建网关、DeepSeek、本地 vLLM 等）。消息协议采用 OpenAI `chat.completions` 格式（`system / user / assistant / tool` 四种角色与 `tool_calls` 机制）。

## 核心能力

### ReAct 主循环
调用模型 → 若返回工具调用则在本地执行并把观察结果回填 → 下一轮；无工具调用即输出最终答复。循环持续到以下任一条件：

1. 无工具调用（模型直接给出答复）
2. 用户中断（`Ctrl+C`，回到提示符继续）
3. 连续多轮无进展自动降级汇总
4. 不可恢复错误（鉴权失败、摘要后仍超长等）

### 本地工具集

| 类别 | 工具                                | 说明                                                                                                                                |
| ---- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 文件 | `read_file`                         | 读取文件，支持行范围，带行号                                                                                                        |
| 文件 | `write_file`                        | 创建/覆盖文件，自动建父目录                                                                                                         |
| 文件 | `edit_file`                         | 唯一匹配替换（多处匹配则拒绝）                                                                                                      |
| 文件 | `list_dir`                          | 列目录，支持递归深度                                                                                                                |
| 文件 | `search_text`                       | 文本/正则搜索，带 `file:line` 前缀                                                                                                  |
| 命令 | `run_command`                       | 捕获 stdout/stderr/退出码；**超时强制终止整个进程树**（Windows `taskkill /F /T`，POSIX `killpg`），杜绝孙进程持有管道导致的终端假死 |
| 信息 | `pwd` / `env_info` / `project_tree` | 工作目录、环境信息、项目树                                                                                                          |

### 上下文管理
token 近似估算与预算控制；超限时**摘要优先**（对旧历史做抽取式摘要回填，保留信息），仅必要时才丢弃最旧完整轮次兑底。全程保证 `assistant.tool_calls` 与 `tool` 回填消息**严格配对**，避免 API 400。

### 输出解析
提取原生 `tool_calls`，多级 JSON 容错（去代码围栏、修引号/尾逗号、截取最外层片段），解析失败将错误回填给模型限次重试。

### 错误处理
限流/网络错误指数退避+抖动重试；鉴权错误直接报错并提示检查 `.env`；上下文超长自动摘要后重试；工具异常一律转为可读观察文本，主循环持续运行。

### 安全与脱敏
`.env` 存密钥且被 `.gitignore` 忽略；`redact()` 全链路脱敏（日志、工具输出、异常信息、会话存档）；文件工具路径白名单限制在工作目录内；危险命令模式黑名单拦截；命令默认限定工作目录。

### 会话持久化
完整对话历史（含工具调用配对）以 JSONL 存档于 `.cyent/sessions/`，写盘前脱敏；空会话不落盘。恢复时校验配对完整性，损坏部分截断到最近合法边界。

### 可观测性
分级日志（DEBUG/INFO/WARNING/ERROR）写入轮转文件 `logs/cyent.log`，终端专用于流式渲染；日志过滤器自动掩码密钥。

## 架构分层

```
cli/（repl.py 组装+REPL+事件渲染 · commands.py 斜杠命令 · prompts.py 系统提示词）
  → core/engine.py（主循环，事件流）
      → core/context.py（上下文+观察者挂钩）+ core/parser.py（解析）
      → core/session.py（JSONL 会话存档，订阅上下文追加）
      → llm/client.py（openai 封装，流式 tool_calls 增量合并）
      → tools/executor.py（注册/校验/分发/隔离）→ tools/*（工具实现）
config/env.py（.env+Settings 单例）· log/logger.py（日志）· utils/errors.py（重试退避）· utils/redact.py（脱敏）
```

CLI 与引擎通过**事件流**解耦：引擎发布 `text_delta / thinking_delta / tool_start / tool_result / final / error / interrupted` 事件，CLI 只消费渲染。assistant 正文与思考内容按 token **实时流式打印**到终端。

## 使用方法

```powershell
# 1. 安装依赖（自动使用 Python 3.14）
uv sync

# 2. 配置：复制 .env.example 为 .env，填入
#    OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL
Copy-Item .env.example .env

# 3. 交互模式（REPL，流式输出）
uv run cyent
uv run cyent -C E:\some\project   # 指定工作目录

# 4. 单任务模式
uv run cyent -p "统计 src 下 Python 行数"

# 5. 会话恢复
uv run cyent -c                   # 继续最近会话
uv run cyent --resume <ID>        # 恢复指定会话
uv run cyent --list-sessions      # 列出可恢复会话

# 6. 运行测试
uv run pytest
```

REPL 斜杠命令：`/help` `/model [NAME]` `/clear` `/tools` `/stats` `/sessions` `/resume <ID>` `/new` `/quit`

## 设计取舍

- 依赖仅 `openai` + `python-dotenv`，其余全部标准库，复杂度与出错面最小
- 上下文预算默认 24k token（可按模型调整），摘要优先、丢弃兑底
- 会话存档为冷存储（全量历史），上下文为热窗口（预算内），互不替代
- 命令工具超时采用进程树击杀，确保 Windows 下管道排空阻塞时也能可靠终止
