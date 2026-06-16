# baseline_reference/

23 个算子的**基线实现源码快照**归档(自包含 JSON,content 字段内嵌原始源码)。

## 数据来源

- 原位置: `AscendOpGenAgent/litebench/baseline_json/`(litebench 老版本生成)
- 迁移时间: 合并到 CANNBench-Lite 时一次性导入
- 总文件数: 25 (23 算子 + `summary.json` + `all_operators.json`)
- 总大小: ~620 KB

## 文件结构

```
baseline_reference/
├── README.md                  # 本文档
├── summary.json               # 索引汇总 (version, total_operators, 每个算子 files/size_kb)
├── all_operators.json         # 合并型大文件 (单文件含全部 23 算子)
├── add_layer_norm.json        # 单算子: 算子级文件清单 + 内嵌源码
├── add_rms_norm.json
├── ...                        # 其余 21 个算子 JSON
└── swi_glu.json
```

## 算子 JSON 字段

```json
{
  "operator_name": "add_layer_norm",        // 算子名
  "base_path": "<CANNBENCH_HOME>/src/cannbench/data/baseline_reference/add_layer_norm",
  "kernel_path": "<CANNBENCH_HOME>/src/cannbench/data/baseline_reference/add_layer_norm/kernel",
  "files": {
    "add_layer_norm_kernel.h": {
      "path": "add_layer_norm_kernel.h",   // 相对 base_path
      "full_path": "<...>/add_layer_norm_kernel.h",  // 绝对路径(目录可能不存在,仅供引用)
      "size_bytes": 9178,
      "md5": "3b887a03f3ef44c1c9bf993ae36e167f",
      "extension": ".h",
      "content": "#pragma once\n..."        // 源码内嵌
    },
    "pybind11.cpp": { ... }
  }
}
```

- **`content`** 字段是**自包含**的真实源码,反序列化即可使用
- `full_path` 指向的物理目录可能不存在(本归档仅保留 JSON 内嵌源码),需要 extract 时按 `base_path` 自动重建

## 与 registry 的关系

每个算子在 `lightweight_benchmark_registry_v2.json` 中新增软引用字段:

```json
{
  "gelu": {
    "benchmark_role": "anchor",
    "...": "...",
    "baseline_reference": "baseline_reference/gelu.json"   // 包内相对路径
  }
}
```

- 路径相对于 `cannbench.data` 包根,Python 端可通过 `importlib.resources.files()` 访问
- 加此字段**不破坏** registry 现有结构,只追加轻量指针

## 用途

- **历史基线对比**: 把新生成的算子实现与此归档做 diff,识别改动点
- **kope agent 引用**: 在性能/正确性分析阶段, agent 可读取 `baseline_reference/{op}.json` 作为"参考实现"
- **离线归档**: JSON 自包含, 不依赖原 `baseline_implementation/` 目录
- **跨环境复现**: 内嵌源码 + md5 校验,保证内容一致

## 路径改写历史

合并时一次性把所有算子 JSON 内的 `base_path` / `kernel_path` / `files.{name}.full_path` 从
`/home/hrl/AscendOpGenAgent/litebench/baseline_implementation/...` 改写为
`/home/hrl/AscendOpGenAgent/CANNBench-Lite/src/cannbench/data/baseline_reference/...`。

未改写的字段:
- `summary.json` / `all_operators.json` — 仅含索引信息,无绝对路径
- `files.{name}.path` — 相对路径,跨位置无意义差异
- `files.{name}.content` — 源码本身,位置无关
- `files.{name}.size_bytes` / `md5` / `extension` — 文件元数据,位置无关

## 维护说明

- **新增算子**: 把 `{op_name}.json` 放到本目录,并在 `lightweight_benchmark_registry_v2.json` 对应算子 entry 加 `baseline_reference` 字段
- **更新基线**: 重新生成 `litebench/baseline_json/{op_name}.json`,然后同步到本目录(同步时按需改写 `base_path` / `full_path`)
- **删除**: 先确认无 `runner_templates_v1.json` / `lightweight_benchmark_registry_v2.json` 引用,再删除
