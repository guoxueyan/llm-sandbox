import asyncio
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import json
import os
import httpx
from mcp.types import CallToolResult, TextContent, ImageContent

# 从环境变量读取 HTTP 服务器配置
HTTP_HOST = os.environ.get("HTTP_HOST", "11.180.61.6")
HTTP_PORT = os.environ.get("HTTP_PORT", "8000")
BASE_URL = f"http://{HTTP_HOST}:{HTTP_PORT}"

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
    envelope = Envelope(
        mcp_result={"error": "failed to create session"},
        extra_info={"tool": "create_sandbox_session"}
    )
    
    try:
        # 通过 HTTP 请求调用 mcp_server.py 的创建会话接口
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{BASE_URL}/api/v1/sessions",
                json={"language": language}
            )
            
            # 检查 HTTP 状态码
            if response.status_code == 201:
                result_data = response.json()
                
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
            else:
                # HTTP 错误
                error_detail = response.json() if response.text else {"error": f"HTTP {response.status_code}"}
                envelope = Envelope(
                    mcp_result={
                        "status": "error",
                        "error": error_detail.get("error", f"HTTP {response.status_code}")
                    },
                    extra_info={
                        "tool": "create_sandbox_session",
                        "message": error_detail.get("message", "HTTP request failed"),
                        "http_status": response.status_code
                    }
                )
    
    except httpx.TimeoutException as e:
        envelope = Envelope(
            mcp_result={"status": "error", "error": "Request timeout"},
            extra_info={
                "tool": "create_sandbox_session",
                "exception": str(e),
                "message": "HTTP request timeout"
            }
        )
    except httpx.RequestError as e:
        envelope = Envelope(
            mcp_result={"status": "error", "error": f"Connection error: {str(e)}"},
            extra_info={
                "tool": "create_sandbox_session",
                "exception": str(e),
                "message": f"Failed to connect to {BASE_URL}"
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
    timeout: int = 600
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
    envelope = Envelope(
        mcp_result={"error": "failed to execute code"},
        extra_info={"tool": "execute_sandbox_code"}
    )
    
    try:
        # 通过 HTTP 请求调用 mcp_server.py 的执行代码接口
        async with httpx.AsyncClient(timeout=timeout + 10.0) as client:
            response = await client.post(
                f"{BASE_URL}/api/v1/execute",
                json={
                    "code": code,
                    "session_id": session_id,
                    "libraries": libraries,
                    "timeout": timeout
                }
            )
            
            # 检查 HTTP 状态码
            if response.status_code == 200:
                result_data = response.json()
                
                # 处理返回结果（可能包含图片）
                has_image = False
                image_data = None
                
                # 检查是否有图片数据
                if "images" in result_data and result_data["images"]:
                    has_image = True
                    # 取第一张图片
                    first_image = result_data["images"][0]
                    image_data = {
                        "type": "image",
                        "data": first_image.get("data"),
                        "mimeType": first_image.get("mime_type")
                    }
                
                if result_data.get("status") == "success":
                    envelope = Envelope(
                        mcp_result={
                            "exit_code": result_data.get("exit_code"),
                            "stdout": result_data.get("stdout"),
                            "stderr": result_data.get("stderr"),
                            "session_id": result_data.get("session_id"),
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
                            "error_type": result_data.get("error_type"),
                            "session_id": result_data.get("session_id"),
                            "message": result_data.get("message")
                        },
                        extra_info={
                            "tool": "execute_sandbox_code",
                            "error_details": result_data.get("error")
                        }
                    )
            else:
                # HTTP 错误（404, 400, 500 等）
                error_detail = response.json() if response.text else {"error": f"HTTP {response.status_code}"}
                envelope = Envelope(
                    mcp_result={
                        "status": "error",
                        "error_type": error_detail.get("error_type", "http_error"),
                        "session_id": session_id,
                        "message": error_detail.get("message", f"HTTP {response.status_code}")
                    },
                    extra_info={
                        "tool": "execute_sandbox_code",
                        "error_details": error_detail.get("error"),
                        "http_status": response.status_code
                    }
                )
    
    except httpx.TimeoutException as e:
        envelope = Envelope(
            mcp_result={
                "status": "error",
                "error": "Execution timeout",
                "session_id": session_id
            },
            extra_info={
                "tool": "execute_sandbox_code",
                "exception": str(e),
                "message": "Code execution timeout"
            }
        )
    except httpx.RequestError as e:
        envelope = Envelope(
            mcp_result={
                "status": "error",
                "error": f"Connection error: {str(e)}",
                "session_id": session_id
            },
            extra_info={
                "tool": "execute_sandbox_code",
                "exception": str(e),
                "message": f"Failed to connect to {BASE_URL}"
            }
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
#         viz_code = """
# import matplotlib.pyplot as plt
# import numpy as np

# x = np.linspace(0, 10, 100)
# y = np.sin(x)

# plt.figure(figsize=(10, 6))
# plt.plot(x, y)
# plt.title('Sine Wave')
# plt.xlabel('x')
# plt.ylabel('sin(x)')
# plt.grid(True)
# plt.show()

# print('Plot generated!')
# """
        viz_code = """
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from mcp.types import CallToolResult, TextContent
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import json
import aiohttp
import time
import random
import sys
from env_config_manager import get_config_value
from log import logger


class Envelope(BaseModel):
    mcp_result: Dict[str, Any] = Field(default_factory=dict)
    extra_info: Dict[str, Any] = Field(default_factory=dict)


async def request_completions_api_async(image: str, session: aiohttp.ClientSession, prompt: str = "no prompt", model_name: str = 'clinical-vlm-paddle-ocr-vl'):
    
    # API配置
    url = get_config_value('ocr_url')
    headers = {
        'Content-Type': 'application/json'
    }

    # 生成request_id
    request_id = "".join([chr(random.randint(0, 65536) % 26 + ord('a')) for _ in range(20)])

    # 构造multi_modal_data
    multi_modal_data = []
    if image:
        multi_modal_data.append({
            "mime_type": "image/url",
            "content": image
        })

    # 构造请求body
    body = {
        "request_id": request_id,
        "prompts": ["no prompt"],
        "multi_modal_data": multi_modal_data
    }


    # 发送异步请求
    try:
        async with session.post(
            url, 
            json=body, 
            headers=headers, 
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                text_data = await response.text()
                data = json.loads(text_data)
                return data
                
            else:
                error_text = await response.text()
                print(f"OCR 请求失败: {error_text[:500]}", file=sys.stderr)
                return ""
                
    except asyncio.TimeoutError:
        error_msg = f"OCR API 请求超时: {image}"
        print(error_msg, file=sys.stderr)
        return {
            'error_code': -1,
            'error_msg': 'timeout',
            'image': image
        }
    except Exception as e:
        error_msg = f"OCR API 请求异常: {type(e).__name__}: {str(e)}"
        print(error_msg, file=sys.stderr)
        return {
            'error_code': -1,
            'error_msg': str(e),
            'image': image
        }

async def ocr_tool(images: List[str]):
    envelope = Envelope(
            mcp_result={"error":"parameter error"},
            extra_info={"tool": "ocr_tool"}
        )
    if not images:
        return []
    
    try:
        # 创建异步HTTP会话
        async with aiohttp.ClientSession() as session:
            server_tasks = [
                request_completions_api_async(image_url, session)
                for image_url in images
            ]
            server_processed_results = []
            content_list = []
            # 并发执行所有请求ocr_server任务
            print(f"ocr_server请求  开始并发处理 {len(images)} 张图片...")
            server_results = await asyncio.gather(*server_tasks, return_exceptions=True)
            info_list = []
            message_list = []
            status_data = {
                "status": "success"
            }
            # 处理结果和异常
            for i, result in enumerate(server_results):
                if isinstance(result, Exception) or result['message'] != "success":
                    print(f"图片 {images[i]} 处理异常: {result}", file=sys.stderr)
                    message_list.append(
                        {
                            "error_msg": 'OCR 处理失败',
                        }
                    )
                    status_data['status'] = 'failed'
                else:
                    # info_list.append(
                    #     {
                    #         "data": result,
                    #         'image': images[i]
                    #     }
                    # )
                    if result.get('data', {}).get('choices', [])[0].get('message', {}).get('meta_data', {}).get('markdown_texts', ''):
                        message_list.append(
                            {
                                "markdown_text": result['data']['choices'][0]['message']['meta_data']['markdown_texts']
                            }
                        )
            content_text = json.dumps(message_list, default=str, ensure_ascii=False)

            envelope = Envelope(
                mcp_result={"ocr_result": message_list}
            )
            return CallToolResult(
                content=[TextContent(type="text", text=content_text)],     # 给模型看的
                structuredContent= envelope.mcp_result,                    # 给模型/程序看的结构化结果
                _meta={                                                    # 只给客户端/中间层看的
                    "extra_info": status_data
                }
            )
        
    except Exception as e:
        error_msg = f"ocr_tool 请求异常: {type(e).__name__}: {str(e)}"
        return CallToolResult(
                content=[TextContent(type="text", text=error_msg)],     # 给模型看的
                isError=True
            )


export_tools = {
    "ocr_tool": {
        "function": ocr_tool,
        "meta": {
            "description": "ocr图片分析工具",
            "tag": "medical"
        }
    }
}

if __name__ == '__main__':
    asyncio.run(ocr_tool(
            [
                "https://quarkmed-vlm.oss-cn-hangzhou.aliyuncs.com/business/mm-query-image-tool-nothink-checksheet-sythesis/56ac008b-c15b-4272-b833-c6845dfb2208/f57c1aea-1c0b-4973-9d3c-a0c80967bd6a.jpg?OSSAccessKeyId=LTAI5tAyZg5kEtQkcDJpBPNK&Expires=1796184638&Signature=ygPvIT5hTYf%2B2S23xSmsxDohiIs%3D",
                "https://b0.bdstatic.com/ugc/tPIfkmAJS7mekPy_5t-uyg8e2ba4e3b761f43e43a7e116a7f51450.jpg@h_1280"
            ]))
"""
        viz_result = asyncio.run(execute_sandbox_code_tool(
            code=viz_code,
            session_id=session_id,
            libraries=["mcp-server-fastmcp", "aiohttp"]
        ))
        print(viz_result)