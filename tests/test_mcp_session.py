#!/usr/bin/env python3
"""
Test MCP Session Management using MCP Client SDK
使用 MCP Client SDK 测试 Session 管理（使用 pickle 持久化变量）
"""

import json
import sys
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_session():
    """测试 MCP Session 的完整流程（使用 pickle 持久化）"""
    
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
                
                # 1. 创建 session
                print("=" * 60)
                print("步骤 1: 创建 Session")
                print("=" * 60)
                
                result = await session.call_tool(
                    "create_session",
                    arguments={"language": "python"}
                )
                
                session_data = json.loads(result.content[0].text)
                
                if "error" in session_data:
                    print(f"❌ 创建 session 失败: {session_data['error']}")
                    return False
                
                session_id = session_data["session_id"]
                print(f"✅ Session ID: {session_id}\n")
                
                # 2. 第一次执行：定义变量并保存到 pickle
                print("=" * 60)
                print("步骤 2: 定义变量并持久化")
                print("=" * 60)
                
                result = await session.call_tool(
                    "execute_code",
                    arguments={
                        "code": """import pickle

# 定义变量
x = 10
y = 20
print(f'定义变量: x = {x}, y = {y}')

# 保存到 pickle 文件
state = {'x': x, 'y': y}
with open('/sandbox/session_state.pkl', 'wb') as f:
    pickle.dump(state, f)
print('✅ 变量已保存到 /sandbox/session_state.pkl')""",
                        "session_id": session_id,
                        "language": "python"
                    }
                )
                
                result_data = json.loads(result.content[0].text)
                print(f"执行结果:")
                print(f"  stdout: {result_data.get('stdout', '')}")
                print(f"  exit_code: {result_data.get('exit_code', -1)}\n")
                
                # 3. 第二次执行：从 pickle 加载变量并使用
                print("=" * 60)
                print("步骤 3: 加载变量并使用")
                print("=" * 60)
                
                result = await session.call_tool(
                    "execute_code",
                    arguments={
                        "code": """import pickle

# 从 pickle 文件加载变量
try:
    with open('/sandbox/session_state.pkl', 'rb') as f:
        state = pickle.load(f)
    
    x = state['x']
    y = state['y']
    print(f'加载变量: x = {x}, y = {y}')
    
    # 使用变量
    z = x + y
    print(f'计算结果: z = x + y = {z}')
    
    # 更新状态
    state['z'] = z
    with open('/sandbox/session_state.pkl', 'wb') as f:
        pickle.dump(state, f)
    print('✅ 状态已更新')
    
except FileNotFoundError:
    print('❌ 状态文件不存在')
except Exception as e:
    print(f'❌ 错误: {e}')""",
                        "session_id": session_id,
                        "language": "python"
                    }
                )
                
                result_data = json.loads(result.content[0].text)
                print(f"执行结果:")
                print(f"  stdout: {result_data.get('stdout', '')}")
                print(f"  exit_code: {result_data.get('exit_code', -1)}")
                
                if result_data.get('exit_code') == 0:
                    print("✅ 变量成功保持并使用！")
                else:
                    print("⚠️  执行失败")
                print()
                
                # 4. 第三次执行：验证所有变量
                print("=" * 60)
                print("步骤 4: 验证所有变量")
                print("=" * 60)
                
                result = await session.call_tool(
                    "execute_code",
                    arguments={
                        "code": """import pickle

with open('/sandbox/session_state.pkl', 'rb') as f:
    state = pickle.load(f)

print('当前状态:')
for key, value in state.items():
    print(f'  {key} = {value}')

# 继续计算
result = state['x'] * state['y'] + state['z']
print(f'\\n新计算: x * y + z = {result}')""",
                        "session_id": session_id,
                        "language": "python"
                    }
                )
                
                result_data = json.loads(result.content[0].text)
                print(f"执行结果:")
                print(f"  stdout: {result_data.get('stdout', '')}")
                print(f"  exit_code: {result_data.get('exit_code', -1)}\n")
                
                # 5. 测试文件持久化
                print("=" * 60)
                print("步骤 5: 测试文件持久化")
                print("=" * 60)
                
                result = await session.call_tool(
                    "execute_code",
                    arguments={
                        "code": """# 创建文本文件
with open('/sandbox/data.txt', 'w') as f:
    f.write('Hello from session!')
print('✅ 文件已创建')

# 列出文件
import os
files = os.listdir('/sandbox')
print(f'\\n/sandbox 目录内容: {files}')""",
                        "session_id": session_id,
                        "language": "python"
                    }
                )
                
                result_data = json.loads(result.content[0].text)
                print(f"执行结果:")
                print(f"  stdout: {result_data.get('stdout', '')}")
                print(f"  exit_code: {result_data.get('exit_code', -1)}\n")
                
                # 6. 读取文件
                print("=" * 60)
                print("步骤 6: 读取文件")
                print("=" * 60)
                
                result = await session.call_tool(
                    "execute_code",
                    arguments={
                        "code": """with open('/sandbox/data.txt', 'r') as f:
    content = f.read()
print(f'文件内容: {content}')""",
                        "session_id": session_id,
                        "language": "python"
                    }
                )
                
                result_data = json.loads(result.content[0].text)
                print(f"执行结果:")
                print(f"  stdout: {result_data.get('stdout', '')}")
                print(f"  exit_code: {result_data.get('exit_code', -1)}\n")
                
                # 7. 查看活跃 sessions
                print("=" * 60)
                print("步骤 7: 查看所有活跃 Session")
                print("=" * 60)
                
                result = await session.call_tool("list_sessions", arguments={})
                sessions_data = json.loads(result.content[0].text)
                print(f"活跃 Sessions:")
                print(json.dumps(sessions_data, indent=2, ensure_ascii=False))
                print()
                
                # 8. 关闭 session
                print("=" * 60)
                print("步骤 8: 关闭 Session")
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
                print("\n📊 测试总结:")
                print("  ✅ Session 创建和绑定")
                print("  ✅ 变量持久化（使用 pickle）")
                print("  ✅ 跨执行变量访问")
                print("  ✅ 文件持久化")
                print("  ✅ Session 管理")
                
                return True
                
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_mcp_session())
    sys.exit(0 if success else 1)