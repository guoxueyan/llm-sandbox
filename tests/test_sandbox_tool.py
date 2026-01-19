import asyncio
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import json
from mcp.types import CallToolResult, TextContent, ImageContent

class Envelope(BaseModel):
    mcp_result: Dict[str, Any] = Field(default_factory=dict)
    extra_info: Dict[str, Any] = Field(default_factory=dict)


async def create_sandbox_session_tool(language: str = "python") -> CallToolResult:
    """创建一个新的代码执行会话
    
    Args:
        language: 编程语言，默认 python
        
    Returns:
        CallToolResult: 包含 session_id 和会话信息
    """
    # 导入 server 模块
    from llm_sandbox.mcp_server.server import create_session
    
    envelope = Envelope(
        mcp_result={"error": "failed to create session"},
        extra_info={"tool": "create_sandbox_session"}
    )
    
    try:
        # 调用 server.py 的 create_session
        result = create_session(language=language)
        
        # 解析返回的 TextContent
        result_data = json.loads(result.text)
        
        if result_data.get("status") == "success":
            envelope = Envelope(
                mcp_result={
                    "session_id": result_data.get("session_id"),
                    "language": result_data.get("language"),
                    "visualization_support": result_data.get("visualization_support"),
                    "timeout": result_data.get("timeout"),
                    "status": "success"
                },
                extra_info={
                    "tool": "create_sandbox_session",
                    "message": result_data.get("message")
                }
            )
        else:
            envelope = Envelope(
                mcp_result={
                    "status": "error",
                    "error": result_data.get("error", "Unknown error")
                },
                extra_info={
                    "tool": "create_sandbox_session",
                    "message": result_data.get("message", "Failed to create session")
                }
            )
    
    except Exception as e:
        envelope = Envelope(
            mcp_result={"status": "error", "error": str(e)},
            extra_info={"tool": "create_sandbox_session", "exception": str(e)}
        )
    
    content_text = json.dumps(envelope.mcp_result, default=str, ensure_ascii=False)
    return CallToolResult(
        content=[TextContent(type="text", text=content_text)],
        structuredContent=envelope.mcp_result,
        _meta={
            "extra_info": envelope.extra_info
        }
    )


async def execute_sandbox_code_tool(
    code: str,
    session_id: str,
    libraries: Optional[List[str]] = None,
    timeout: int = 30
) -> CallToolResult:
    """在指定会话中执行代码
    
    Args:
        code: 要执行的代码
        session_id: 会话ID（必填）
        libraries: 需要安装的库列表
        timeout: 执行超时时间（秒），默认30秒
        
    Returns:
        CallToolResult: 包含执行结果和可能的可视化图表
    """
    # 导入 server 模块
    from llm_sandbox.mcp_server.server import execute_code
    
    envelope = Envelope(
        mcp_result={"error": "failed to execute code"},
        extra_info={"tool": "execute_sandbox_code"}
    )
    
    try:
        # 调用 server.py 的 execute_code
        results = execute_code(
            code=code,
            session_id=session_id,
            libraries=libraries,
            timeout=timeout
        )
        
        # 处理返回结果（可能包含图片和文本）
        has_image = False
        image_data = None
        text_result = None
        
        for item in results:
            if isinstance(item, ImageContent):
                has_image = True
                image_data = {
                    "type": "image",
                    "data": item.data,
                    "mimeType": item.mimeType
                }
            elif isinstance(item, TextContent):
                text_result = json.loads(item.text)
        
        # 构建返回结果
        if text_result:
            if text_result.get("status") == "success":
                envelope = Envelope(
                    mcp_result={
                        "exit_code": text_result.get("exit_code"),
                        "stdout": text_result.get("stdout"),
                        "stderr": text_result.get("stderr"),
                        "session_id": text_result.get("session_id"),
                        "status": "success",
                        "has_visualization": has_image
                    },
                    extra_info={
                        "tool": "execute_sandbox_code",
                        "image": image_data if has_image else None
                    }
                )
            else:
                # 错误情况
                envelope = Envelope(
                    mcp_result={
                        "status": "error",
                        "error_type": text_result.get("error_type"),
                        "session_id": text_result.get("session_id"),
                        "message": text_result.get("message")
                    },
                    extra_info={
                        "tool": "execute_sandbox_code",
                        "error_details": text_result.get("error")
                    }
                )
        else:
            envelope = Envelope(
                mcp_result={"status": "error", "error": "No result returned"},
                extra_info={"tool": "execute_sandbox_code"}
            )
    
    except Exception as e:
        envelope = Envelope(
            mcp_result={"status": "error", "error": str(e)},
            extra_info={"tool": "execute_sandbox_code", "exception": str(e)}
        )
    
    content_text = json.dumps(envelope.mcp_result, default=str, ensure_ascii=False)
    
    # 如果有图片，添加到 content 中
    content_list = [TextContent(type="text", text=content_text)]
    if envelope.extra_info.get("image"):
        image_info = envelope.extra_info["image"]
        content_list.append(
            ImageContent(
                type="image",
                data=image_info["data"],
                mimeType=image_info["mimeType"]
            )
        )
    
    return CallToolResult(
        content=content_list,
        structuredContent=envelope.mcp_result,
        _meta={
            "extra_info": envelope.extra_info
        }
    )


export_tools = {
    "create_sandbox_session": {
        "function": create_sandbox_session_tool,
        "meta": {
            "description": "创建代码执行会话。输入: language(编程语言，默认python)。输出: session_id和会话信息",
            "tag": "sandbox"
        }
    },
    "execute_sandbox_code": {
        "function": execute_sandbox_code_tool,
        "meta": {
            "description": "在指定会话中执行代码。输入: code(代码), session_id(会话ID), libraries(可选的库列表), timeout(超时时间)。输出: 执行结果和可能的可视化图表",
            "tag": "sandbox"
        }
    }
}


# 使用示例
if __name__ == "__main__":
    # 示例1: 创建会话
    print("=== 创建会话 ===")
    session_result = asyncio.run(create_sandbox_session_tool(language="python"))
    print(session_result)
    print()
    
    # 从结果中提取 session_id
    session_data = json.loads(session_result.content[0].text)
    session_id = session_data.get("session_id")
    
    if session_id:
        # 示例2: 执行简单代码
        print("=== 执行简单代码 ===")
        code_result = asyncio.run(execute_sandbox_code_tool(
            code="print('Hello from sandbox!')\nprint(2 + 2)",
            session_id=session_id
        ))
        print(code_result)
        print()
        
        # 示例3: 执行带可视化的代码
        print("=== 执行可视化代码 ===")
        viz_code = """
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.show()

print('Plot generated!')
"""
        viz_result = asyncio.run(execute_sandbox_code_tool(
            code=viz_code,
            session_id=session_id
        ))
        print(viz_result)
