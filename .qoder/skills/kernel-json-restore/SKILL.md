---
name: kernel-json-restore
description: >
  从 JSON 文件还原 AscendC Kernel 算子项目。
  读取 baseline_json 目录下的算子 JSON，解析并还原完整的 kernel 项目结构，
  包括源文件、头文件、CMakeLists.txt 等。
argument-hint: >
  输入：operator-name、output-path。
  输出：还原后的算子项目目录结构。
---

# Kernel JSON Restore Skill

<role>
你是一个 AscendC 算子项目还原专家。你的任务是从 JSON 文件中读取算子源码信息，并还原出完整的 kernel 项目目录结构，包括所有 .cpp、.h 文件和 CMakeLists.txt。
</role>

## 工作流程

```
输入：operator_name（如 layer_norm_v3）+ output_path
    ↓
[1. 定位 JSON 文件] → baseline_json/{operator_name}.json
    ↓
[2. 读取并解析 JSON]
    ↓
[3. 创建目录结构] → output-path/{operator_name}/kernel/
    ↓
[4. 写入源文件] → 还原所有 .cpp、.h、CMakeLists.txt
    ↓
[5. 验证完整性] → 检查 MD5、文件数量
    ↓
输出：完整的算子项目目录
```

---

## Step 1: 定位 JSON 文件

算子 JSON 文件存储在以下目录：

```
/home/hrl/AscendOpGenAgent/litebench/baseline_json/
```

查找规则：
- 单算子文件：`{operator_name}.json`
- 例如：`layer_norm_v3.json`

如果单算子文件不存在，尝试在 `all_operators.json` 中查找。

---

## Step 2: 读取并解析 JSON

JSON 文件结构说明：

```json
{
  "operator_name": "layer_norm_v3",
  "base_path": "/home/hrl/.../baseline_implementation/layer_norm_v3",
  "kernel_path": "/home/hrl/.../baseline_implementation/layer_norm_v3/kernel",
  "files": {
    "pybind11.cpp": {
      "path": "pybind11.cpp",
      "full_path": "...",
      "size_bytes": 2623,
      "md5": "3e9d3631db08734403a3fa8c212f0d8b",
      "extension": ".cpp",
      "content": "..."
    },
    "*.cpp": {...},
    "*.h": {...},
    "CMakeLists.txt": {...}
  },
  "metadata": {
    "total_files": 7,
    "total_size_bytes": 12939,
    "file_types": {
      ".cpp": 3,
      ".h": 3,
      ".txt": 1
    }
  }
}
```

---

## Step 3: 创建目录结构

根据 JSON 中的 `kernel_path` 信息创建目录：

```bash
mkdir -p output_path/{operator_name}/kernel
```

---

## Step 4: 写入源文件

遍历 JSON 中的 `files` 字段，将每个文件的 `content` 写入对应路径：

```python
for file_name, file_info in json_data["files"].items():
    file_path = Path(output_path) / operator_name / "kernel" / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(file_info["content"])
```

---

## Step 5: 验证完整性

### 5.1 检查文件数量

```python
actual_files = len(list(output_dir.rglob("*.cpp"))) + len(list(output_dir.rglob("*.h")))
expected_files = json_data["metadata"]["total_files"]

assert actual_files == expected_files, \
    f"文件数量不匹配：期望 {expected_files}，实际 {actual_files}"
```

### 5.2 验证 MD5

```python
import hashlib

for file_name, file_info in json_data["files"].items():
    file_path = output_dir / file_name

    md5_hash = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)

    actual_md5 = md5_hash.hexdigest()
    expected_md5 = file_info["md5"]

    assert actual_md5 == expected_md5, \
        f"{file_name} MD5 不匹配：期望 {expected_md5}，实际 {actual_md5}"
```

### 5.3 检查关键文件

必须包含以下关键文件：
- `.cpp` 源文件
- `.h` 头文件
- `CMakeLists.txt`

---

## 使用示例

### 示例 1: 还原单个算子

**输入**：
- operator_name: `layer_norm_v3`
- output_path: `/tmp/restored_ops`

**执行**：
```bash
# 创建输出目录
mkdir -p /tmp/restored_ops

# 使用 Python 脚本还原
python3 << 'PYEOF'
import json
from pathlib import Path

operator_name = "layer_norm_v3"
output_base = Path("/tmp/restored_ops")
json_file = Path("/home/hrl/AscendOpGenAgent/litebench/baseline_json") / f"{operator_name}.json"

# 读取 JSON
with open(json_file, 'r') as f:
    data = json.load(f)

# 创建目录并写入文件
output_dir = output_base / operator_name / "kernel"
output_dir.mkdir(parents=True, exist_ok=True)

for file_name, file_info in data["files"].items():
    file_path = output_dir / file_name
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(file_info["content"])
    print(f"✅ 还原: {file_name}")

print(f"\n✅ 算子 {operator_name} 还原完成")
print(f"📁 输出目录: {output_dir}")
PYEOF
```

**输出**：
```
✅ 还原: pybind11.cpp
✅ 还原: layer_norm_v3_kernel.h
✅ 还原: layer_norm_v3_fp16.cpp
✅ 还原: layer_norm_v3_fp32.cpp
✅ 还原: CMakeLists.txt

✅ 算子 layer_norm_v3 还原完成
📁 输出目录: /tmp/restored_ops/layer_norm_v3/kernel
```

### 示例 2: 批量还原所有算子

```bash
python3 << 'PYEOF'
import json
from pathlib import Path

json_dir = Path("/home/hrl/AscendOpGenAgent/litebench/baseline_json")
output_base = Path("/tmp/restored_ops")

# 读取汇总文件获取算子列表
with open(json_dir / "summary.json", 'r') as f:
    summary = json.load(f)

print(f"📊 开始还原 {summary['total_operators']} 个算子...\n")

for op_info in summary["operators"]:
    op_name = op_info["name"]

    # 读取单算子 JSON
    json_file = json_dir / f"{op_name}.json"
    if not json_file.exists():
        print(f"⚠️ 跳过 {op_name}（JSON 文件不存在）")
        continue

    with open(json_file, 'r') as f:
        data = json.load(f)

    # 还原文件
    output_dir = output_base / op_name / "kernel"
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_name, file_info in data["files"].items():
        file_path = output_dir / file_name
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(file_info["content"])

    print(f"✅ {op_name} ({op_info['files']} 文件)")

print(f"\n✅ 批量还原完成！")
print(f"📁 输出目录: {output_base}")
PYEOF
```

---

## 关键文件说明

### 目录结构

```
{operator_name}/
└── kernel/
    ├── CMakeLists.txt      # CMake 构建配置
    ├── *.cpp              # 源文件（kernel 实现）
    ├── *.h                # 头文件（kernel 定义）
    └── pybind11.cpp       # Python 绑定文件
```

### CMakeLists.txt 关键配置

通常包含：
- `cmake_minimum_required(VERSION 3.16.0)`
- `ascendc_library()` - AscendC 库构建
- `ascendc_include_directories()` - 包含目录
- 链接 `torch_npu` 等库

### 关键源文件

| 文件类型 | 说明 |
|---------|------|
| `*_kernel.h` | Kernel 类定义，包含 `Init()` 和 `Process()` |
| `*_fp16.cpp` | FP16 实现入口 |
| `*_fp32.cpp` | FP32 实现入口 |
| `*_tiling.h` | Tiling 参数定义 |
| `kernel_common.h` | 通用 kernel 工具 |
| `pybind11.cpp` | Python binding |

---

## 注意事项

1. **路径兼容性**：还原的项目可以在不同版本的 CANN 中使用
2. **MD5 验证**：确保还原的文件与原始文件一致
3. **目录结构**：保持与原始 `baseline_implementation` 一致
4. **编译依赖**：确保 CANN 环境变量正确设置（`ASCEND_CANN_PACKAGE_PATH`）

---

## 常见问题

### Q1: JSON 文件不存在
**解决**：检查 `operator_name` 是否正确，或使用汇总文件搜索

### Q2: MD5 验证失败
**解决**：重新从 `baseline_implementation` 打包 JSON

### Q3: 文件写入失败
**解决**：检查输出目录权限

### Q4: 缺少 CMakeLists.txt
**解决**：某些算子可能没有独立的 CMakeLists.txt，使用 kernel 目录下的配置

---

## 扩展功能

### 功能 1: 生成项目配置

```python
def generate_project_config(json_data: dict) -> dict:
    """生成项目配置信息"""
    return {
        "operator_name": json_data["operator_name"],
        "source_files": list(json_data["files"].keys()),
        "file_count": json_data["metadata"]["total_files"],
        "total_size_kb": json_data["metadata"]["total_size_bytes"] / 1024,
        "file_types": json_data["metadata"]["file_types"]
    }
```

### 功能 2: 统计还原结果

```python
def summarize_restore(output_dir: Path) -> dict:
    """统计还原结果"""
    cpp_files = list(output_dir.rglob("*.cpp"))
    h_files = list(output_dir.rglob("*.h"))

    return {
        "total_files": len(cpp_files) + len(h_files),
        "cpp_files": len(cpp_files),
        "h_files": len(h_files),
        "total_size_kb": sum(f.stat().st_size for f in cpp_files + h_files) / 1024
    }
```
