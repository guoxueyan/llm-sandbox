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
                
                try:
                    result = await session.call_tool(
                        "create_session",
                        arguments={"language": "python"}
                    )
                    
                    # ✅ 添加错误检查
                    if not result.content:
                        print("❌ 创建 session 失败：返回内容为空")
                        return False
                    
                    session_data = json.loads(result.content[0].text)
                    
                    # ✅ 检查是否有错误
                    if "error" in session_data:
                        print(f"❌ 创建 session 失败: {session_data['error']}")
                        print(f"详细信息: {session_data.get('status', 'unknown')}")
                        
                        # 检查是否是 Docker 权限问题
                        if "Permission denied" in str(session_data.get('error', '')):
                            print("\n💡 解决方案:")
                            print("1. 将用户添加到 docker 组: sudo usermod -aG docker $USER")
                            print("2. 重新登录或运行: newgrp docker")
                            print("3. 验证权限: docker ps")
                        
                        return False
                    
                    # ✅ 检查是否有 session_id
                    if "session_id" not in session_data:
                        print(f"❌ 返回数据中没有 session_id: {session_data}")
                        return False
                    
                    session_id = session_data["session_id"]
                    print(f"✅ Session ID: {session_id}\n")
                    
                except Exception as e:
                    print(f"❌ 创建 session 时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
                
                # 2. 第一次执行：定义变量
                print("=" * 60)
                print("步骤 2: 第一次执行 - 定义变量")
                print("=" * 60)
                
                try:
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
                    print(f"  exit_code: {result_data.get('exit_code', -1)}")
                    
                    if result_data.get('exit_code') != 0:
                        print(f"  stderr: {result_data.get('stderr', '')}")
                    print()
                    
                except Exception as e:
                    print(f"❌ 执行代码时出错: {e}")
                    return False
                
                # 3. 第二次执行：使用之前的变量
                # 步骤 3 的改进版本
                print("=" * 60)
                print("步骤 3: 第二次执行 - 使用之前的变量")
                print("=" * 60)

                try:
                    result = await session.call_tool(
                        "execute_code",
                        arguments={
                            "code": """
                # 先检查变量是否存在
                try:
                    print(f'x exists: {x}')
                    print(f'y exists: {y}')
                    z = x + y
                    print(f'z = x + y = {z}')
                except NameError as e:
                    print(f'ERROR: Variable not found - {e}')
                    import sys
                    sys.exit(1)
                """,
                            "session_id": session_id,
                            "language": "python"
                        }
                    )
                    
                    result_data = json.loads(result.content[0].text)
                    print(f"执行结果:")
                    print(f"  stdout: {result_data.get('stdout', '')}")
                    print(f"  stderr: {result_data.get('stderr', '')}")  # ✅ 显示错误
                    print(f"  exit_code: {result_data.get('exit_code', -1)}")
                    
                    if result_data.get('exit_code') != 0:
                        print("⚠️  警告：变量没有保持，session 绑定可能有问题")
                    print()
                    
                except Exception as e:
                    print(f"❌ 执行代码时出错: {e}")
                    return False
                
                # 4. 查看所有活跃 session
                print("=" * 60)
                print("步骤 4: 查看所有活跃 Session")
                print("=" * 60)
                
                try:
                    result = await session.call_tool("list_sessions", arguments={})
                    sessions_data = json.loads(result.content[0].text)
                    print(f"活跃 Sessions:")
                    print(json.dumps(sessions_data, indent=2, ensure_ascii=False))
                    print()
                except Exception as e:
                    print(f"❌ 查看 sessions 时出错: {e}")
                
                # 5. 关闭 session
                print("=" * 60)
                print("步骤 5: 关闭 Session")
                print("=" * 60)
                
                try:
                    result = await session.call_tool(
                        "close_session",
                        arguments={"session_id": session_id}
                    )
                    
                    close_data = json.loads(result.content[0].text)
                    print(f"关闭结果:")
                    print(json.dumps(close_data, indent=2, ensure_ascii=False))
                    
                    if close_data.get("status") == "success":
                        print("✅ Session 成功关闭")
                    
                except Exception as e:
                    print(f"❌ 关闭 session 时出错: {e}")
                
                print("\n" + "=" * 60)
                print("✅ 所有测试完成!")
                print("=" * 60)
                
                return True
                
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        
        # 检查是否是 Docker 相关错误
        error_msg = str(e)
        if "Permission denied" in error_msg or "docker" in error_msg.lower():
            print("\n💡 这看起来是 Docker 权限问题。解决方案:")
            print("1. 将用户添加到 docker 组:")
            print("   sudo usermod -aG docker $USER")
            print("2. 重新登录或运行:")
            print("   newgrp docker")
            print("3. 验证 Docker 权限:")
            print("   docker ps")
            print("4. 再次运行测试:")
            print("   python3 tests/test_mcp_session.py")
        
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_mcp_session())
    sys.exit(0 if success else 1)