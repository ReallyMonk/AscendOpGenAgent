# 可迁移 Benchmark 架构设计

## 设计目标

1. **框架无关**：核心逻辑不依赖任何特定 Agent 框架
2. **协议驱动**：通过标准协议（HTTP/JSON-RPC）通信
3. **配置化**：所有行为通过配置文件定义
4. **插件化**：功能以插件形式扩展

---

## 架构分层

```
┌─────────────────────────────────────────────────────────────────┐
│                    应用层 (Application Layer)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  OpenCode   │  │  LangChain  │  │   AutoGen   │  ...         │
│  │   Agent     │  │   Agent     │  │   Agent     │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│         ▼                ▼                ▼                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              适配器层 (Adapter Layer)                        ││
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐                ││
│  │  │OpenCode   │  │LangChain  │  │ AutoGen   │                ││
│  │  │Adapter    │  │Adapter    │  │Adapter    │                ││
│  │  └───────────┘  └───────────┘  └───────────┘                ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/JSON-RPC
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    服务层 (Service Layer)                        │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Benchmark MCP Server                            ││
│  │                                                              ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          ││
│  │  │   Query     │  │  Execution  │  │   Result    │          ││
│  │  │   Module    │  │   Module    │  │   Module    │          ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘          ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Plugin Manager                                  ││
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐                ││
│  │  │  NPU      │  │  CUDA     │  │  Custom   │                ││
│  │  │  Plugin   │  │  Plugin   │  │  Plugin   │                ││
│  │  └───────────┘  └───────────┘  └───────────┘                ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    基础设施层 (Infrastructure Layer)              │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Registry   │  │   Worker    │  │   Storage   │              │
│  │   Store     │  │   Pool      │  │   Backend   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心协议定义

### 1. 通用请求/响应格式

```json
{
  "protocol_version": "1.0",
  "request_id": "uuid",
  "action": "benchmark.run",
  "payload": {
    "operator": "layer_norm",
    "shape": "S1",
    "mode": "baseline",
    "options": {}
  },
  "context": {
    "session_id": "optional-session-id",
    "metadata": {}
  }
}
```

### 2. 标准响应格式

```json
{
  "protocol_version": "1.0",
  "request_id": "uuid",
  "status": "success|error|pending",
  "result": {
    "data": {},
    "metrics": {}
  },
  "error": null,
  "metadata": {
    "duration_ms": 1234,
    "server_version": "1.0.0"
  }
}
```

---

## 适配器层设计

### 适配器接口规范

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    returns_schema: Dict[str, Any]

class BenchmarkAdapter(ABC):
    """框架适配器基类"""
    
    @abstractmethod
    def get_tools(self) -> list[ToolDefinition]:
        """返回该框架可识别的工具定义"""
        pass
    
    @abstractmethod
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具调用，返回结果"""
        pass
    
    @abstractmethod
    def format_response(self, result: Dict[str, Any]) -> Any:
        """将结果格式化为框架特定格式"""
        pass
```

### OpenCode 适配器示例

```python
class OpenCodeAdapter(BenchmarkAdapter):
    """OpenCode 框架适配器"""
    
    def get_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="benchmark_list",
                description="列出可用的 benchmark 算子",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "set_name": {"type": "string"},
                        "layer": {"type": "string"},
                        "category": {"type": "string"}
                    }
                },
                returns_schema={
                    "type": "object",
                    "properties": {
                        "operators": {"type": "array"},
                        "total": {"type": "integer"}
                    }
                }
            ),
            # ... 更多工具定义
        ]
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        # 调用 MCP Server 的 HTTP 接口
        response = httpx.post(
            f"{self.server_url}/invoke",
            json={
                "action": tool_name,
                "payload": params
            }
        )
        return response.json()
    
    def format_response(self, result: Dict[str, Any]) -> str:
        # OpenCode 期望返回字符串格式
        return json.dumps(result, indent=2, ensure_ascii=False)
```

### LangChain 适配器示例

```python
from langchain.tools import BaseTool
from pydantic import BaseModel

class BenchmarkListTool(BaseTool):
    """LangChain 工具封装"""
    
    name = "benchmark_list"
    description = "列出可用的 benchmark 算子"
    
    class InputSchema(BaseModel):
        set_name: Optional[str] = None
        layer: Optional[str] = None
        category: Optional[str] = None
    
    def _run(self, set_name: str = None, layer: str = None, category: str = None):
        response = httpx.post(
            f"{self.server_url}/invoke",
            json={"action": "benchmark.list", "payload": {
                "set_name": set_name,
                "layer": layer,
                "category": category
            }}
        )
        return response.json()

class LangChainAdapter(BenchmarkAdapter):
    """LangChain 框架适配器"""
    
    def get_tools(self) -> list[BaseTool]:
        return [
            BenchmarkListTool(),
            BenchmarkRunTool(),
            BenchmarkCompareTool()
        ]
```

---

## 配置驱动设计

### 服务配置 `benchmark_config.yaml`

```yaml
server:
  host: "0.0.0.0"
  port: 9028
  protocol: "http"  # http | grpc | json-rpc

registry:
  type: "file"  # file | database | api
  path: "./registry/lightweight_benchmark_registry_v2.json"

workers:
  default_pool:
    max_workers: 4
    timeout: 300
  servers:
    - name: "910b"
      url: "http://127.0.0.1:9027"
      device: 7
      health_check_interval: 30

plugins:
  - name: "npu"
    enabled: true
    config:
      framework: "torch_npu"
  - name: "cuda"
    enabled: false
    config:
      framework: "torch.cuda"

storage:
  results_dir: "./results"
  baseline_cache: "./results/baseline_cache"
  format: "json"  # json | parquet | sqlite

logging:
  level: "INFO"
  file: "./logs/benchmark.log"
```

### 工具定义配置 `tools_config.yaml`

```yaml
tools:
  - name: "benchmark_list"
    action: "list"
    description: "列出可用的 benchmark 算子"
    category: "query"
    parameters:
      set_name:
        type: "string"
        required: false
        description: "benchmark set 名称"
      layer:
        type: "string"
        required: false
        description: "算子层级"
      category:
        type: "string"
        required: false
        description: "算子类别"
    returns:
      operators:
        type: "array"
        description: "算子列表"
      total:
        type: "integer"
        description: "总数"

  - name: "benchmark_run"
    action: "run"
    description: "执行 benchmark 测试"
    category: "execution"
    parameters:
      operator:
        type: "string"
        required: true
        description: "算子名称"
      shape:
        type: "string"
        required: true
        description: "shape ID"
      mode:
        type: "string"
        required: true
        enum: ["baseline", "custom"]
        description: "运行模式"
      custom_dir:
        type: "string"
        required: false
        description: "自定义实现目录 (mode=custom 时必需)"
      profiling:
        type: "boolean"
        default: true
        description: "是否启用 profiling"
    returns:
      status:
        type: "string"
        description: "执行状态"
      metrics:
        type: "object"
        description: "性能指标"
      profiling_data:
        type: "object"
        description: "Profiling 数据"

  - name: "benchmark_compare"
    action: "compare"
    description: "比较 baseline 和 custom 结果"
    category: "analysis"
    parameters:
      baseline_result:
        type: "string"
        required: true
        description: "baseline 结果路径或 ID"
      custom_result:
        type: "string"
        required: true
        description: "custom 结果路径或 ID"
    returns:
      comparison:
        type: "object"
        description: "比较结果"
      speedup:
        type: "number"
        description: "加速比"
```

---

## 插件系统设计

### 插件接口

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BenchmarkPlugin(ABC):
    """Benchmark 插件基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """插件名称"""
        pass
    
    @property
    @abstractmethod
    def supported_frameworks(self) -> List[str]:
        """支持的框架列表"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """初始化插件"""
        pass
    
    @abstractmethod
    def run_baseline(self, operator: str, shape: str, **options) -> Dict[str, Any]:
        """运行 baseline 测试"""
        pass
    
    @abstractmethod
    def run_custom(self, operator: str, shape: str, custom_dir: str, **options) -> Dict[str, Any]:
        """运行自定义实现测试"""
        pass
    
    @abstractmethod
    def get_profiling(self, run_id: str) -> Dict[str, Any]:
        """获取 profiling 数据"""
        pass
```

### NPU 插件示例

```python
class NPUPlugin(BenchmarkPlugin):
    """NPU (Ascend) 插件"""
    
    @property
    def name(self) -> str:
        return "npu"
    
    @property
    def supported_frameworks(self) -> List[str]:
        return ["torch_npu"]
    
    def initialize(self, config: Dict[str, Any]) -> None:
        import torch_npu
        self.device = config.get("device", 0)
        torch_npu.npu.set_device(self.device)
    
    def run_baseline(self, operator: str, shape: str, **options) -> Dict[str, Any]:
        # 调用现有的 baseline_benchmark_runner
        from baseline_benchmark_runner import run_baseline_case_remote
        return run_baseline_case_remote(operator, shape, **options)
    
    def run_custom(self, operator: str, shape: str, custom_dir: str, **options) -> Dict[str, Any]:
        # 调用现有的 custom_remote_runner
        from custom_remote_runner import run_custom_case
        return run_custom_case(operator, shape, custom_dir, **options)
```

### CUDA 插件示例

```python
class CUDAPlugin(BenchmarkPlugin):
    """CUDA (NVIDIA) 插件"""
    
    @property
    def name(self) -> str:
        return "cuda"
    
    @property
    def supported_frameworks(self) -> List[str]:
        return ["torch.cuda", "cupy", "numba"]
    
    def initialize(self, config: Dict[str, Any]) -> None:
        import torch
        self.device = config.get("device", 0)
        torch.cuda.set_device(self.device)
    
    def run_baseline(self, operator: str, shape: str, **options) -> Dict[str, Any]:
        # CUDA 特定的 baseline 实现
        pass
    
    def run_custom(self, operator: str, shape: str, custom_dir: str, **options) -> Dict[str, Any]:
        # CUDA 特定的 custom 实现
        pass
```

---

## 统一 MCP Server 实现

```python
#!/usr/bin/env python3
"""
Benchmark MCP Server - 框架无关的实现
"""

import yaml
import httpx
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ToolContext:
    tool_name: str
    params: Dict[str, Any]
    session_id: Optional[str]
    config: Dict[str, Any]

class BenchmarkMCPServer:
    """框架无关的 Benchmark MCP Server"""
    
    def __init__(self, config_path: str = "benchmark_config.yaml"):
        self.config = self._load_config(config_path)
        self.tools = self._load_tools()
        self.plugins = self._load_plugins()
        self.registry = self._load_registry()
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        return yaml.safe_load(Path(path).read_text())
    
    def _load_tools(self) -> Dict[str, Dict]:
        tools_config = yaml.safe_load(
            Path("tools_config.yaml").read_text()
        )
        return {t["name"]: t for t in tools_config["tools"]}
    
    def _load_plugins(self) -> Dict[str, BenchmarkPlugin]:
        plugins = {}
        for plugin_config in self.config.get("plugins", []):
            if plugin_config["enabled"]:
                plugin = self._create_plugin(plugin_config)
                plugins[plugin.name] = plugin
        return plugins
    
    def invoke(self, action: str, payload: Dict[str, Any], 
               context: Optional[Dict] = None) -> Dict[str, Any]:
        """统一的调用入口"""
        tool = self.tools.get(action)
        if not tool:
            return {"status": "error", "error": f"Unknown action: {action}"}
        
        # 验证参数
        validation = self._validate_params(tool, payload)
        if not validation["valid"]:
            return {"status": "error", "error": validation["error"]}
        
        # 执行工具
        try:
            result = self._execute_tool(action, payload, context)
            return {
                "status": "success",
                "result": result,
                "metadata": {"action": action}
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _execute_tool(self, action: str, params: Dict[str, Any],
                      context: Optional[Dict]) -> Dict[str, Any]:
        """执行具体工具"""
        tool = self.tools[action]
        category = tool["category"]
        
        if category == "query":
            return self._execute_query(action, params)
        elif category == "execution":
            return self._execute_benchmark(action, params)
        elif category == "analysis":
            return self._execute_analysis(action, params)
        else:
            raise ValueError(f"Unknown category: {category}")
    
    def _execute_query(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行查询类工具"""
        if action == "benchmark_list":
            return self._list_operators(**params)
        # ... 其他查询工具
    
    def _execute_benchmark(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行 benchmark 类工具"""
        if action == "benchmark_run":
            mode = params["mode"]
            plugin = self._get_plugin_for_mode(mode)
            
            if mode == "baseline":
                return plugin.run_baseline(
                    params["operator"],
                    params["shape"],
                    **params.get("options", {})
                )
            elif mode == "custom":
                return plugin.run_custom(
                    params["operator"],
                    params["shape"],
                    params["custom_dir"],
                    **params.get("options", {})
                )
    
    def _get_plugin_for_mode(self, mode: str) -> BenchmarkPlugin:
        """根据模式获取插件"""
        # 从配置中读取默认插件
        plugin_name = self.config.get("default_plugin", "npu")
        return self.plugins[plugin_name]

# HTTP 服务入口
from fastapi import FastAPI, Request

app = FastAPI()
server = BenchmarkMCPServer()

@app.post("/invoke")
async def invoke(request: Request):
    body = await request.json()
    return server.invoke(
        action=body.get("action"),
        payload=body.get("payload", {}),
        context=body.get("context")
    )

@app.get("/tools")
async def list_tools():
    return {"tools": list(server.tools.keys())}

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

---

## 使用示例

### OpenCode 集成

```jsonc
// opencode.jsonc
{
  "mcpServers": {
    "benchmark": {
      "command": "python",
      "args": ["benchmark/mcp_server.py"],
      "config": "benchmark/benchmark_config.yaml"
    }
  }
}
```

### LangChain 集成

```python
from langchain.agents import initialize_agent
from benchmark.adapters.langchain import LangChainAdapter

adapter = LangChainAdapter(server_url="http://localhost:9028")
tools = adapter.get_tools()

agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
agent.run("列出所有 Normalization 类别的算子")
```

### AutoGen 集成

```python
from autogen import AssistantAgent
from benchmark.adapters.autogen import AutoGenAdapter

adapter = AutoGenAdapter(server_url="http://localhost:9028")

agent = AssistantAgent(
    name="benchmark_agent",
    llm_config=llm_config,
    function_map=adapter.get_function_map()
)
```

---

## 迁移性优势

| 特性 | 说明 |
|------|------|
| **协议无关** | 支持 HTTP/gRPC/JSON-RPC，框架只需实现客户端 |
| **配置驱动** | 所有工具定义和行为通过配置文件，无需修改代码 |
| **插件扩展** | 新硬件/框架只需实现插件接口 |
| **适配器模式** | 每个框架一个适配器，核心逻辑不变 |
| **标准化接口** | 统一的请求/响应格式，便于集成 |

---

## 下一步实现

1. 实现 `BenchmarkMCPServer` 核心类
2. 创建 `OpenCodeAdapter` 适配器
3. 编写配置文件模板
4. 实现 `NPUPlugin` 插件
5. 添加单元测试和集成测试
