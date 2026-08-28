# Cyent — 极简编码 Agent

从零实现的编码 Agent：**不使用任何现成 agent 框架**（LangChain / OpenAI Agents SDK / AutoGen 等均未引入），仅以 `openai` SDK 作为模型客户端，可对接任意 OpenAI 兼容端点（官方 API、自建网关、DeepSeek、本地 vLLM 等）。消息协议只处理 OpenAI `chat.completions` 格式（`system / user / assistant / tool` 四种角色与 `tool_calls` 机制）。

## 核心能力

### ReAct 主循环
调用模型 → 若返回工具调用则在本地执行并把观察结果回填 → 下一轮；无工具调用即输出最终答复。五种终止条件：

1. 达到最大迭代数
2. 无工具调用（模型直接给出答复）
3. 用户中断（`Ctrl+C`，不崩溃，回到提示符）
4. 连续多轮无进展自动降级汇总
5. 显式退出

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
token 近似估算、超限裁剪（按完整轮次丢弃）、仍超限则对最旧段落做抽取式摘要回填。全程保证 `assistant.tool_calls` 与 `tool` 回填消息**严格配对**，避免 API 400。

### 输出解析
提取原生 `tool_calls`，多级 JSON 容错（去代码围栏、修引号/尾逗号、截取最外层片段），解析失败将错误回填给模型限次重试。

### 错误处理
限流/网络错误指数退避+抖动重试；鉴权错误不重试并提示检查 `.env`；上下文超长自动裁剪后重试；工具异常一律转为可读观察文本，绝不中断主循环。

### 安全与脱敏
`.env` 存密钥且被 `.gitignore` 忽略；`redact()` 全链路脱敏（日志、工具输出、异常信息）；文件工具路径白名单限制在工作目录内；危险命令模式黑名单拦截；命令默认限定工作目录。

### 可观测性
分级日志（DEBUG/INFO/WARNING/ERROR）**只写轮转文件** `logs/cyent.log`（不污染终端，流式渲染不被打断），日志过滤器自动掩码密钥。

## 架构分层

```
cli/repl.py（REPL+斜杠命令+事件渲染）
  → core/engine.py（主循环）
      → core/context.py（上下文）+ core/parser.py（解析）
      → llm/client.py（openai 封装，流式 tool_calls 增量按下标合并）
      → tools/executor.py（注册/校验/分发/隔离/超时）→ tools/*（工具实现）
config/env.py（.env+Settings）· log/logger.py（日志）· utils/errors.py（重试退避）· utils/redact.py（脱敏）
```

CLI 与引擎通过**事件流**解耦：引擎发布 `text_delta / tool_start / tool_result / final / error / interrupted` 事件，CLI 只消费渲染。assistant 正文按 token **实时流式打印**到终端。

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

# 5. 运行测试
uv run pytest
```

REPL 斜杠命令：`/help` `/model [NAME]` `/clear` `/tools` `/stats` `/quit`

## 设计取舍

- 只依赖 `openai` + `python-dotenv`，其余全部标准库，复杂度与出错面最小
- 流式输出同时暴露文本增量与工具调用增量，CLI 实时渲染，引擎侧合并成完整调用后再解析
- 上下文预算默认 24k token（可按模型调整），裁剪优先、摘要兜底
- 命令工具超时采用进程树击杀，避免 Windows 管道排空阻塞造成的假死
