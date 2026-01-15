#!/usr/bin/env python3
"""
Test MCP Session Management
测试 MCP Server 的 Session 管理功能
"""

import json
import sys

# ✅ 需要导入 mcp 客户端（根据实际的 MCP 客户端库调整）
# 注意：这里的 Client 导入方式取决于你使用的 MCP 客户端库
try:
    from mcp.client import Client  # 或者其他实际的导入路径
except ImportError:
    print("❌ 错误：无法导入 MCP Client")
    print("请确保已安装 MCP 客户端库")
    sys.exit(1)

def test_mcp_session():
    """测试 MCP Session 的完整流程"""
    
    # ✅ 连接到 MCP Server（根据实际情况调整 URL）
    # 如果是本地启动的 MCP Server，可能需要使用不同的连接方式
    try:
        client = Client("http://localhost:8000")  # 或者使用其他连接方式
        print("✅ 成功连接到 MCP Server\n")
    except Exception as e:
        print(f"❌ 连接 MCP Server 失败: {e}")
        print("请确保 MCP Server 已启动")
        sys.exit(1)
    
    try:
        # 1. 创建一个调试 session
        print("=" * 60)
        print("步骤 1: 创建调试 Session")
        print("=" * 60)
        
        session_info = client.call_tool("create_session", {
            "language": "python"
        })
        print(f"Session 创建结果: {session_info}")
        
        # ✅ 解析 session_id
        session_data = json.loads(session_info[0].text)
        session_id = session_data["session_id"]
        print(f"✅ Session ID: {session_id}\n")
        
        # 2. 第一次执行：定义变量
        print("=" * 60)
        print("步骤 2: 第一次执行 - 定义变量")
        print("=" * 60)
        
        result1 = client.call_tool("execute_code", {
            "code": """
x = 10
y = 20
print(f'x = {x}, y = {y}')
""",
            "session_id": session_id,
            "language": "python"
        })
        
        # ✅ 解析执行结果
        result1_data = json.loads(result1[0].text)
        print(f"执行结果:")
        print(f"  stdout: {result1_data.get('stdout', '')}")
        print(f"  exit_code: {result1_data.get('exit_code', -1)}")
        
        if result1_data.get('exit_code') != 0:
            print(f"  stderr: {result1_data.get('stderr', '')}")
        print()
        
        # 3. 第二次执行：使用之前定义的变量（同一个容器）
        print("=" * 60)
        print("步骤 3: 第二次执行 - 使用之前的变量")
        print("=" * 60)
        
        result2 = client.call_tool("execute_code", {
            "code": """
z = x + y  # x 和 y 应该还在
print(f'z = x + y = {z}')
""",
            "session_id": session_id,
            "language": "python"
        })
        
        result2_data = json.loads(result2[0].text)
        print(f"执行结果:")
        print(f"  stdout: {result2_data.get('stdout', '')}")
        print(f"  exit_code: {result2_data.get('exit_code', -1)}")
        
        if result2_data.get('exit_code') != 0:
            print(f"  stderr: {result2_data.get('stderr', '')}")
            print("⚠️  警告：变量可能没有保持，检查 session 是否正确绑定")
        print()
        
        # 4. 第三次执行：创建文件
        print("=" * 60)
        print("步骤 4: 第三次执行 - 创建文件")
        print("=" * 60)
        
        result3 = client.call_tool("execute_code", {
            "code": """
with open('/sandbox/data.txt', 'w') as f:
    f.write(f'Result: {z}')
print('File created')
""",
            "session_id": session_id,
            "language": "python"
        })
        
        result3_data = json.loads(result3[0].text)
        print(f"执行结果:")
        print(f"  stdout: {result3_data.get('stdout', '')}")
        print(f"  exit_code: {result3_data.get('exit_code', -1)}")
        print()
        
        # 5. 第四次执行：读取文件（文件还在）
        print("=" * 60)
        print("步骤 5: 第四次执行 - 读取文件")
        print("=" * 60)
        
        result4 = client.call_tool("execute_code", {
            "code": """
with open('/sandbox/data.txt', 'r') as f:
    content = f.read()
print(f'File content: {content}')
""",
            "session_id": session_id,
            "language": "python"
        })
        
        result4_data = json.loads(result4[0].text)
        print(f"执行结果:")
        print(f"  stdout: {result4_data.get('stdout', '')}")
        print(f"  exit_code: {result4_data.get('exit_code', -1)}")
        
        if result4_data.get('exit_code') != 0:
            print(f"  stderr: {result4_data.get('stderr', '')}")
            print("⚠️  警告：文件可能没有保持，检查 session 是否正确绑定")
        print()
        
        # 6. 查看所有活跃 session
        print("=" * 60)
        print("步骤 6: 查看所有活跃 Session")
        print("=" * 60)
        
        sessions = client.call_tool("list_sessions", {})
        sessions_data = json.loads(sessions[0].text)
        print(f"活跃 Sessions:")
        print(json.dumps(sessions_data, indent=2, ensure_ascii=False))
        print()
        
        # 7. 完成调试后关闭 session
        print("=" * 60)
        print("步骤 7: 关闭 Session")
        print("=" * 60)
        
        close_result = client.call_tool("close_session", {
            "session_id": session_id
        })
        
        close_data = json.loads(close_result[0].text)
        print(f"关闭结果:")
        print(json.dumps(close_data, indent=2, ensure_ascii=False))
        
        if close_data.get("status") == "success":
            print("✅ Session 成功关闭")
        else:
            print("❌ Session 关闭失败")
        
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
    success = test_mcp_session()
    sys.exit(0 if success else 1)