# CANNBench-Lite

轻量级 NPU 算子 Benchmark CLI 与远程 Worker 工具包。

## 安装

```bash
# 使用 uv 安装（推荐）
uv tool install -e .

# 或使用 pip
pip install -e .
```

安装后提供两个 CLI 命令：

- `cannbench-lite` — 主命令：list / run-baseline / run-custom / compare
- `cannbench-worker` — 远程 Worker 服务器

## 快速开始

```bash
# 列出可用算子
cannbench-lite list

# 运行基线 benchmark
cannbench-lite run-baseline --op gelu --shape S1 --server 910b --device 0

# 运行自定义算子评测
cannbench-lite run-custom --op gelu --custom-dir output/gelu --server 910b --device 0

# 对比基线与自定义结果
cannbench-lite compare --baseline results/baseline/... --custom results/custom/...
```

## 项目结构

```
CANNBench-Lite/
├── src/cannbench/             # Python 包 (src-layout)
│   ├── cli.py                 # CLI 入口
│   ├── scripts/               # 业务逻辑模块
│   ├── remote/                # 远程 Worker + 编译 + 评测
│   └── data/                  # 打包数据
│       ├── registry/          # JSON 注册表
│       └── assets/            # 评测脚本 + 模板
│           ├── evaluate.py
│           ├── generate_pybind.py
│           ├── shared/        # 报告 + 用例注册模块
│           └── template/      # CppExtension 模板
├── repo-wiki/                 # LLM Agent 优化文档
├── llm.txt                    # LLM 文档索引
├── pyproject.toml
└── README.md
```

## Remote Worker

```bash
# 启动 Worker 服务器
cannbench-worker --host 0.0.0.0 --port 9027 --log-dir /path/to/logs --devices 0 --max-workers 4
```

Worker 运行前需设置 CANN 环境变量：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
source $ASCEND_HOME_PATH/bin/setenv.bash
```

## 文档

- [llm.txt](llm.txt) — LLM Agent 文档主索引
- [repo-wiki/](repo-wiki/) — 详细技术文档
