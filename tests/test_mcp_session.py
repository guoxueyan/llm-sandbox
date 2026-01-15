#!/usr/bin/env python3
"""
Test MCP Session Management using MCP Client SDK
使用 MCP Client SDK 测试 Session 管理
"""

import json
import sys
import asyncio
from contextlib import asynccontextmanager

# ✅ 使用 MCP 的 StdioServerParameters 和 stdio_client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_session():
    """测试 MCP Session 的完整流程"""
    
    # ✅ 配置 MCP Server 参数（通过 stdio 启动）
    server_params = StdioServerParameters(
        command="python3",
        args=["-m", "llm_sandbox.mcp_server.server"],
        env={
            "BACKEND": "docker",
            "POOL_MAX_SIZE": "5",
            "POOL_MIN_SIZE": "2",
            "SESSION_TIMEOUT": "1800",
            "PREINSTALL_LIBS_PYTHON": "numpy,pandas,pydantic",
        }
    )
    
    try:
        # ✅ 通过 stdio 连接到 MCP Server
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                
                # 初始化连接
                await session.initialize()
                
                print("✅ 成功连接到 MCP Server\n")
                
                # 1. 创建调试 session
                print("=" * 60)
                print("步骤 1: 创建调试 Session")
                print("=" * 60)
                
                result = await session.call_tool(
                    "create_session",
                    arguments={"language": "python"}
                )
                
                session_data = json.loads(result.content[0].text)
                session_id = session_data["session_id"]
                print(f"✅ Session ID: {session_id}\n")
                
                # 2. 第一次执行：定义变量
                print("=" * 60)
                print("步骤 2: 第一次执行 - 定义变量")
                print("=" * 60)
                
                result = await session.call_tool(
                    "execute_code",
                    arguments={
                        "code": """
x = 10
y = 20
print(f'x = {x}, y = {y}')
""",
                        "session_id": session_id,
                        "language": "python"
                    }
                )
                
                result_data = json.loads(result.content[0].text)
                print(f"执行结果:")
                print(f"  stdout: {result_data.get('stdout', '')}")
                print(f"  exit_code: {result_data.get('exit_code', -1)}\n")
                
                # 3. 第二次执行：使用之前的变量
                print("=" * 60)
                print("步骤 3: 第二次执行 - 使用之前的变量")
                print("=" * 60)
                
                result = await session.call_tool(
                    "execute_code",
                    arguments={
                        "code": """
z = x + y
print(f'z = x + y = {z}')
""",
                        "session_id": session_id,
                        "language": "python"
                    }
                )
                
                result_data = json.loads(result.content[0].text)
                print(f"执行结果:")
                print(f"  stdout: {result_data.get('stdout', '')}")
                print(f"  exit_code: {result_data.get('exit_code', -1)}\n")
                
                # 4. 查看所有活跃 session
                print("=" * 60)
                print("步骤 4: 查看所有活跃 Session")
                print("=" * 60)
                
                result = await session.call_tool("list_sessions", arguments={})
                sessions_data = json.loads(result.content[0].text)
                print(f"活跃 Sessions:")
                print(json.dumps(sessions_data, indent=2, ensure_ascii=False))
                print()
                
                # 5. 关闭 session
                print("=" * 60)
                print("步骤 5: 关闭 Session")
                print("=" * 60)
                
                result = await session.call_tool(
                    "close_session",
                    arguments={"session_id": session_id}
                )
                
                close_data = json.loads(result.content[0].text)
                print(f"关闭结果:")
                print(json.dumps(close_data, indent=2, ensure_ascii=False))
                
                if close_data.get("status") == "success":
                    print("✅ Session 成功关闭")
                
                print("\n" + "=" * 60)
                print("✅ 所有测试完成!")
                print("=" * 60)
                
                return True
                
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_mcp_session())
    sys.exit(0 if success else 1)