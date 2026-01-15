#!/usr/bin/env python3
"""
Test MCP Session Management using MCP Client SDK
使用 MCP Client SDK 测试 Session 管理
"""

import json
import sys
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_session():
    """测试 MCP Session 的完整流程"""
    
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
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                
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
                
                if "error" in session_data:
                    print(f"❌ 创建 session 失败: {session_data['error']}")
                    return False
                
                if "session_id" not in session_data:
                    print(f"❌ 返回数据中没有 session_id: {session_data}")
                    return False
                
                session_id = session_data["session_id"]
                print(f"✅ Session ID: {session_id}\n")
                
                # 2. 第一次执行：定义变量
                print("=" * 60)
                print("步骤 2: 第一次执行 - 定义变量")
                print("=" * 60)
                
                result = await session.call_tool(
                    "execute_code",
                    arguments={
                        "code": """x = 10
y = 20
print(f'x = {x}, y = {y}')""",  # ✅ 修复：去掉前面的换行和缩进
                        "session_id": session_id,
                        "language": "python"
                    }
                )
                
                result_data = json.loads(result.content[0].text)
                print(f"执行结果:")
                print(f"  stdout: {result_data.get('stdout', '')}")
                print(f"  stderr: {result_data.get('stderr', '')}")
                print(f"  exit_code: {result_data.get('exit_code', -1)}\n")
                
                # 3. 第二次执行：使用之前的变量
                print("=" * 60)
                print("步骤 3: 第二次执行 - 使用之前的变量")
                print("=" * 60)
                
                result = await session.call_tool(
                    "execute_code",
                    arguments={
                        "code": """# 检查变量是否存在
try:
    print(f'x exists: {x}')
    print(f'y exists: {y}')
    z = x + y
    print(f'z = x + y = {z}')
except NameError as e:
    print(f'ERROR: Variable not found - {e}')
    import sys
    sys.exit(1)""",  # ✅ 修复：去掉前面的缩进，从第一列开始
                        "session_id": session_id,
                        "language": "python"
                    }
                )
                
                result_data = json.loads(result.content[0].text)
                print(f"执行结果:")
                print(f"  stdout: {result_data.get('stdout', '')}")
                print(f"  stderr: {result_data.get('stderr', '')}")
                print(f"  exit_code: {result_data.get('exit_code', -1)}")
                
                if result_data.get('exit_code') != 0:
                    print("⚠️  警告：执行失败")
                else:
                    print("✅ 变量成功保持！")
                print()
                
                # 4. 第三次执行：创建文件
                print("=" * 60)
                print("步骤 4: 第三次执行 - 创建文件")
                print("=" * 60)
                
                result = await session.call_tool(
                    "execute_code",
                    arguments={
                        "code": """with open('/sandbox/data.txt', 'w') as f:
    f.write(f'Result: {z}')
print('File created')""",
                        "session_id": session_id,
                        "language": "python"
                    }
                )
                
                result_data = json.loads(result.content[0].text)
                print(f"执行结果:")
                print(f"  stdout: {result_data.get('stdout', '')}")
                print(f"  exit_code: {result_data.get('exit_code', -1)}\n")
                
                # 5. 第四次执行：读取文件
                print("=" * 60)
                print("步骤 5: 第四次执行 - 读取文件")
                print("=" * 60)
                
                result = await session.call_tool(
                    "execute_code",
                    arguments={
                        "code": """with open('/sandbox/data.txt', 'r') as f:
    content = f.read()
print(f'File content: {content}')""",
                        "session_id": session_id,
                        "language": "python"
                    }
                )
                
                result_data = json.loads(result.content[0].text)
                print(f"执行结果:")
                print(f"  stdout: {result_data.get('stdout', '')}")
                print(f"  exit_code: {result_data.get('exit_code', -1)}\n")
                
                # 6. 查看所有活跃 session
                print("=" * 60)
                print("步骤 6: 查看所有活跃 Session")
                print("=" * 60)
                
                result = await session.call_tool("list_sessions", arguments={})
                sessions_data = json.loads(result.content[0].text)
                print(f"活跃 Sessions:")
                print(json.dumps(sessions_data, indent=2, ensure_ascii=False))
                print()
                
                # 7. 关闭 session
                print("=" * 60)
                print("步骤 7: 关闭 Session")
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