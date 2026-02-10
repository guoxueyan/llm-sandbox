"""LLM Sandbox MCP Server with Session Management.

A Model Context Protocol server that provides secure code execution capabilities using llm-sandbox.
Supports session-based container binding for stateful debugging.
"""

import json
import logging
import os
import time
import uuid
from typing import Dict, Optional
import threading
import re
import ast
from pathlib import Path
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
import asyncio
from asyncio import Lock as AsyncLock

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from llm_sandbox import SupportedLanguage
from llm_sandbox.data import ExecutionResult
from llm_sandbox.mcp_server.const import LANGUAGE_RESOURCES
from llm_sandbox.const import SandboxBackend
from llm_sandbox.session import _check_dependency

from llm_sandbox.pool import create_pool_manager, PooledSandboxSession, ArtifactPooledSandboxSession, PoolConfig

def setup_logging():
    """配置日志输出到文件和控制台，按日期自动切换"""
    # 创建 logs 目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 日志文件名格式：sandbox-2026-02-05.log
    log_filename = log_dir / f"sandbox-{datetime.now().strftime('%Y-%m-%d')}.log"
    
    # 配置日志格式
    log_format = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 创建根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除已有的 handlers（避免重复）
    root_logger.handlers.clear()
    
    # ✅ 文件 Handler - 按天切换
    file_handler = TimedRotatingFileHandler(
        filename=log_filename,
        when='midnight',  # 每天午夜切换
        interval=1,       # 间隔 1 天
        backupCount=30,   # 保留 30 天的日志
        encoding='utf-8'
    )
    # 设置文件名后缀格式
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # ✅ 控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # 添加 handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 为特定 logger 设置级别
    logging.getLogger("llm-sandbox-mcp").setLevel(logging.INFO)
    
    return root_logger

# ✅ 初始化日志（在导入后立即执行）
setup_logging()
logger = logging.getLogger("llm-sandbox-mcp")

mcp = FastMCP("llm-sandbox")

# ✅ 全局容器池管理器（按语言分组）
_pool_managers: Dict[str, any] = {}

# ✅ Session 绑定管理器（session_id -> 容器绑定信息）
_session_bindings: Dict[str, Dict] = {}

# 线程锁
_pool_lock = AsyncLock()
_session_lock = AsyncLock()

# Session 配置
SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", "3600"))  # 1 小时
SESSION_CLEANUP_INTERVAL = int(os.environ.get("SESSION_CLEANUP_INTERVAL", "300"))  # 5 分钟

LOCAL_MODULES = {'env_config_manager'}  # 本地模块列表，不应该通过 pip 安装

def _get_backend() -> SandboxBackend:
    """Get the backend to use for the sandbox session."""
    backend = SandboxBackend(os.environ.get("BACKEND", "docker"))
    _check_dependency(backend)
    return backend

def _get_pool_config() -> PoolConfig:
    """Get pool configuration from environment variables."""
    return PoolConfig(
        # 基础配置
        max_pool_size=int(os.environ.get("POOL_MAX_SIZE", "100")),
        min_pool_size=int(os.environ.get("POOL_MIN_SIZE", "10")),
        
        # 超时配置
        idle_timeout=float(os.environ.get("POOL_IDLE_TIMEOUT", "3600.0")),
        acquisition_timeout=float(os.environ.get("POOL_ACQUISITION_TIMEOUT", "30.0")),
        
        # 生命周期配置
        max_container_lifetime=float(os.environ.get("POOL_MAX_LIFETIME", "3600.0")),
        health_check_interval=float(os.environ.get("POOL_HEALTH_CHECK_INTERVAL", "60.0")),
        
        # 策略配置
        exhaustion_strategy=os.environ.get("POOL_EXHAUSTION_STRATEGY", "wait"),
        enable_prewarming=os.environ.get("POOL_ENABLE_PREWARMING", "true").lower() == "true",
    )

def _get_common_libraries(language: str) -> list[str]:
    """Get common libraries to pre-install for a language."""
    common_libs = {
        "python": ["pyyaml"],
        # "python": ["numpy", "pandas", "matplotlib", "requests", "pydantic"],
        # "javascript": ["lodash", "axios"],
    }
    return common_libs.get(language, [])

async def _get_or_create_pool(language: str):
    """Get or create a container pool for the specified language."""
    global _pool_managers
    
    async with _pool_lock:
        if language in _pool_managers:
            return _pool_managers[language]
        
        logger.info(f"Creating container pool for language: {language}")
        
        preinstall_libs_env = os.environ.get(f"PREINSTALL_LIBS_{language.upper()}", "")
        if preinstall_libs_env:
            preinstall_libs = [lib.strip() for lib in preinstall_libs_env.split(",")]
        else:
            preinstall_libs = _get_common_libraries(language)
        
        logger.info(f"Pre-installing libraries for {language}: {preinstall_libs}")
        
        pool = await asyncio.to_thread(
            create_pool_manager,
            backend=os.environ.get("BACKEND", "docker"),
            config=_get_pool_config(),
            lang=language,
            libraries=preinstall_libs,
            keep_template=True,
            verbose=False,
            kube_namespace=os.environ.get("NAMESPACE", "default"),
        )
        
        _pool_managers[language] = pool
        logger.info(f"Container pool created for {language}")
        
        return pool

async def _cleanup_expired_sessions_async():
    """Clean up expired sessions (background task)."""
    global _session_bindings
    
    while True:
        try:
            await asyncio.sleep(SESSION_CLEANUP_INTERVAL)
            
            current_time = time.time()
            expired_sessions = []
            
            async with _session_lock:
                for session_id, binding in _session_bindings.items():
                    if current_time - binding["last_access"] > SESSION_TIMEOUT:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    logger.info(f"Cleaning up expired session: {session_id}")
                    binding = _session_bindings[session_id]
                    
                    # 关闭 session（归还容器到池）
                    if binding.get("session"):
                        try:
                            await asyncio.to_thread(binding["session"].close)
                        except Exception as e:
                            logger.error(f"Error closing session {session_id}: {e}")
                    
                    del _session_bindings[session_id]
                
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                    
        except Exception as e:
            logger.error(f"Error in session cleanup: {e}")

# # ✅ 启动后台清理线程
# _cleanup_thread = threading.Thread(target=_cleanup_expired_sessions, daemon=True)
# _cleanup_thread.start()

def _start_cleanup_task():
    """Start the async cleanup task."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_cleanup_expired_sessions_async())
        logger.info("Started async cleanup task")
    except RuntimeError:
        # 如果没有运行的事件循环，创建一个新的
        logger.warning("No running event loop, cleanup task will start with server")

async def _get_or_create_session(session_id: str, language: str, use_artifact: bool = False) -> PooledSandboxSession:
    """Get or create a session bound to a specific session_id.
    
    Args:
        session_id: Unique session identifier
        language: Programming language
        use_artifact: Whether to use artifact session (for visualization)
        
    Returns:
        PooledSandboxSession bound to this session_id
    """
    global _session_bindings
    
    async with _session_lock:
        # 如果 session 已存在，更新访问时间并返回
        if session_id in _session_bindings:
            binding = _session_bindings[session_id]
            binding["last_access"] = time.time()
            
            # 检查语言是否匹配
            if binding["language"] != language:
                logger.warning(f"Session {session_id} language mismatch: {binding['language']} vs {language}")
                # 可以选择抛出异常或创建新 session
            
            logger.info(f"Reusing existing session: {session_id}")
            return binding["session"]
        
        # 创建新的 session
        logger.info(f"Creating new session: {session_id} for language: {language}")
        
        pool = await _get_or_create_pool(language)
        session_cls = ArtifactPooledSandboxSession if use_artifact else PooledSandboxSession
        
        # 创建并打开 session
        session = session_cls(pool_manager=pool)
        await asyncio.to_thread(session.open)
        
        # 记录绑定信息
        _session_bindings[session_id] = {
            "session": session,
            "language": language,
            "created_at": time.time(),
            "last_access": time.time(),
            "use_artifact": use_artifact,
        }
        
        logger.info(f"Session created and bound: {session_id}")
        return session

def _supports_visualization(language: str) -> bool:
    """Check if a language supports visualization capture."""
    lang_details = LANGUAGE_RESOURCES.get(language)
    return lang_details.get("visualization_support", False) if lang_details else False


def _extract_imports_from_code(code: str, language: str) -> list[str]:
    """Extract import statements from code to detect required packages.
    
    Args:
        code: Source code to analyze
        language: Programming language
        
    Returns:
        List of package names that might need installation
    """
    packages = []
    
    if language == "python":
        try:
            # 使用 AST 解析 Python 代码
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        packages.append(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        packages.append(node.module.split('.')[0])
        except SyntaxError:
            # 如果 AST 解析失败，使用正则表达式作为后备
            import_pattern = r'^\s*(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            for match in re.finditer(import_pattern, code, re.MULTILINE):
                packages.append(match.group(1))
    
    elif language == "javascript" or language == "typescript":
        # 匹配 require() 和 import 语句
        patterns = [
            r'require\([\'"]([^\'"]+)[\'"]\)',
            r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, code):
                packages.append(match.group(1))
    
    # 过滤掉标准库（Python 示例）
    if language == "python":
        import sys
        # ✅ 直接使用 sys.stdlib_module_names（Python 3.10+）
        if hasattr(sys, 'stdlib_module_names'):
            packages = [pkg for pkg in packages if pkg not in sys.stdlib_module_names]
        else:
            # ✅ Python 3.9 及以下的后备方案
            stdlib_modules = {
                'os', 'sys', 'json', 'time', 'datetime', 're', 'math', 'asyncio',
                'random', 'collections', 'itertools', 'functools', 'typing',
                'pathlib', 'io', 'logging', 'unittest', 'threading', 'subprocess'
            }
            packages = [pkg for pkg in packages if pkg not in stdlib_modules]
        packages = [pkg for pkg in packages if pkg not in LOCAL_MODULES]
    return list(set(packages))  # 去重

# ✅ 新增：创建 session
@mcp.tool()
async def create_session(language: str = "python", libraries: list[str] | None = None) -> TextContent:
    """Create a new debugging session with a dedicated container."""
    try:
        session_id = str(uuid.uuid4())
        use_artifact = _supports_visualization(language)
        
        logger.info(f"[CREATE_SESSION] Starting session creation: {session_id}, language: {language}, libraries: {libraries}")
        
        # 创建 session
        session = await _get_or_create_session(
            session_id=session_id,
            language=language,
            use_artifact=True
        )
        logger.info(f"[CREATE_SESSION] Session object created: {session_id}")
        
        # ✅ 如果用户指定了 libraries，则安装并验证
        if libraries:
            logger.info(f"[CREATE_SESSION] Installing user-specified libraries for session {session_id}: {libraries}")
            
            try:
                # ✅ 方案1：先安装库（不执行 import）
                logger.info(f"[CREATE_SESSION] Step 1: Installing libraries via session.install()")
                await asyncio.to_thread(session.install, libraries)
                logger.info(f"[CREATE_SESSION] Step 1 completed: Libraries installation command executed")
                
                # ✅ 方案2：验证库是否真正安装成功
                logger.info(f"[CREATE_SESSION] Step 2: Verifying library installation")
                verification_code = _generate_verification_code(language, libraries)
                logger.info(f"[CREATE_SESSION] Verification code:\n{verification_code}")
                
                verify_result = await asyncio.to_thread(
                    session.run,
                    verification_code,
                    [],  # libraries
                    30   # timeout
                )
                
                logger.info(f"[CREATE_SESSION] Verification result - exit_code: {verify_result.exit_code}")
                logger.info(f"[CREATE_SESSION] Verification stdout:\n{verify_result.stdout}")
                
                if verify_result.exit_code != 0:
                    logger.error(f"[CREATE_SESSION] Library verification failed for session {session_id}")
                    logger.error(f"[CREATE_SESSION] Verification stderr:\n{verify_result.stderr}")
                    
                    # ✅ 尝试重新安装
                    logger.warning(f"[CREATE_SESSION] Attempting to reinstall libraries: {libraries}")
                    import_statements = "\n".join([f"import {lib}" for lib in libraries])
                    reinstall_result = await asyncio.to_thread(
                        session.run,
                        import_statements,
                        libraries,
                        300
                    )
                    
                    logger.info(f"[CREATE_SESSION] Reinstall result - exit_code: {reinstall_result.exit_code}")
                    logger.info(f"[CREATE_SESSION] Reinstall stdout:\n{reinstall_result.stdout}")
                    
                    if reinstall_result.exit_code != 0:
                        logger.error(f"[CREATE_SESSION] Reinstall also failed!")
                        logger.error(f"[CREATE_SESSION] Reinstall stderr:\n{reinstall_result.stderr}")
                else:
                    logger.info(f"[CREATE_SESSION] ✅ Libraries verified successfully in session {session_id}")
                    
            except Exception as e:
                logger.error(f"[CREATE_SESSION] Failed to install/verify libraries in session {session_id}: {e}", exc_info=True)
                # 继续创建 session，但记录警告
                logger.warning(f"[CREATE_SESSION] Session {session_id} created but library installation may have failed")
        
        result = {
            "status": "success",
            "session_id": session_id,
            "language": language,
            "visualization_support": use_artifact,
            "timeout": SESSION_TIMEOUT,
            "message": f"Session created successfully. Container will be kept alive for {SESSION_TIMEOUT} seconds of inactivity."
        }
        
        logger.info(f"[CREATE_SESSION] Session creation completed: {session_id}")
        return TextContent(text=json.dumps(result, indent=2), type="text")
        
    except Exception as e:
        logger.exception(f"[CREATE_SESSION] Error creating session: {e}")
        return TextContent(
            text=json.dumps({
                "status": "error",
                "error": str(e),
                "message": "Failed to create session"
            }, indent=2),
            type="text"
        )

def _generate_verification_code(language: str, libraries: list[str]) -> str:
    """Generate code to verify library installation.
    
    Args:
        language: Programming language
        libraries: List of libraries to verify
        
    Returns:
        Verification code string
    """
    if language == "python":
        # 生成验证代码
        imports = "\n".join([f"import {lib}" for lib in libraries])
        versions = "\n".join([
            f"try:\n"
            f"    print(f'{lib}: {{__import__(\"{lib}\").__version__}}')\n"
            f"except AttributeError:\n"
            f"    print(f'{lib}: installed (no __version__)')\n"
            for lib in libraries
        ])
        return f"{imports}\nprint('All libraries imported successfully!')\n{versions}"
    
    # 其他语言的验证逻辑
    return f"# Verification for {language} not implemented"

@mcp.tool()
async def execute_code(
    code: str,
    session_id: str,
    libraries: list[str] | None = None,
    timeout: int = 30,
    auto_install: bool = True,
) -> list[ImageContent | TextContent]:
    """Execute code in a secure sandbox environment with session binding."""
    results: list[ImageContent | TextContent] = []

    try:
        # ✅ 检查 session_id
        if not session_id or not session_id.strip():
            error_result = {
                "status": "error",
                "error_type": "missing_session_id",
                "session_id": None,
                "message": "session_id is required. Please create a session first using create_session()."
            }
            return [TextContent(text=json.dumps(error_result, indent=2), type="text")]
        
        logger.info(f"[EXECUTE_CODE] Starting code execution in session: {session_id}")
        
        # ✅ 检查 session 是否存在
        async with _session_lock:
            if session_id not in _session_bindings:
                logger.error(f"[EXECUTE_CODE] Session not found: {session_id}")
                error_result = {
                    "status": "error",
                    "error_type": "session_not_found",
                    "session_id": session_id,
                    "message": f"Session '{session_id}' not found or has been released due to timeout ({SESSION_TIMEOUT}s inactivity)."
                }
                return [TextContent(text=json.dumps(error_result, indent=2), type="text")]
            
            language = _session_bindings[session_id]["language"]
            use_artifact = _session_bindings[session_id].get("use_artifact", False)
            logger.info(f"[EXECUTE_CODE] Session info - language: {language}, use_artifact: {use_artifact}")
        
        # ✅ 自动检测依赖
        detected_packages = []
        if auto_install:
            detected_packages = _extract_imports_from_code(code, language)
            logger.info(f"[EXECUTE_CODE] Detected packages from code: {detected_packages}")
        
        # ✅ 获取待安装的库
        pending_libs = []
        async with _session_lock:
            if session_id in _session_bindings:
                pending_libs = _session_bindings[session_id].pop("pending_libraries", [])
        
        if pending_libs:
            logger.info(f"[EXECUTE_CODE] Found pending libraries: {pending_libs}")
        
        # ✅ 合并所有需要安装的库
        all_libraries = list(set((libraries or []) + detected_packages + pending_libs))
        
        if all_libraries:
            logger.info(f"[EXECUTE_CODE] Total libraries to install: {all_libraries}")
        else:
            logger.info(f"[EXECUTE_CODE] No libraries to install")

        # ✅ 获取 session
        logger.info(f"[EXECUTE_CODE] Getting session object for: {session_id}")
        session = await _get_or_create_session(
            session_id=session_id,
            language="python",
            use_artifact=True
        )
        logger.info(f"[EXECUTE_CODE] Session object retrieved successfully")
        
        # ✅ 在执行前验证容器状态
        try:
            logger.info(f"[EXECUTE_CODE] Verifying container health...")
            health_check = await asyncio.to_thread(
                session.execute_command, 
                "echo 'Container is alive'"
            )
            logger.info(f"[EXECUTE_CODE] Container health check - exit_code: {health_check.exit_code}")
            logger.info(f"[EXECUTE_CODE] Container health check output: {health_check.stdout}")
        except Exception as e:
            logger.error(f"[EXECUTE_CODE] Container health check failed: {e}", exc_info=True)
        
        # ✅ 如果有库需要安装，先单独安装并验证
        if all_libraries:
            logger.info(f"[EXECUTE_CODE] Pre-installing libraries before code execution...")
            try:
                # ✅ 步骤1：先单独调用 install 方法
                logger.info(f"[EXECUTE_CODE] Step 1: Calling session.install({all_libraries})")
                await asyncio.to_thread(session.install, all_libraries)
                # session.install(all_libraries)
                logger.info(f"[EXECUTE_CODE] Step 1 completed: session.install() executed")
                
                # ✅ 步骤2：验证安装结果
                logger.info(f"[EXECUTE_CODE] Step 2: Verifying installation...")
                verification_code = _generate_verification_code(language, all_libraries)
                verify_result = await asyncio.to_thread(
                    session.run,
                    verification_code,
                    [],  # libraries
                    30   # timeout
                )
                logger.info(f"[EXECUTE_CODE] Verification result - exit_code: {verify_result.exit_code}")
                logger.info(f"[EXECUTE_CODE] Verification stdout:\n{verify_result.stdout}")
                
                if verify_result.exit_code != 0:
                    logger.error(f"[EXECUTE_CODE] ❌ Pre-installation verification FAILED!")
                    logger.error(f"[EXECUTE_CODE] Verification stderr:\n{verify_result.stderr}")
                    
                    # ✅ 尝试使用 session.run() 重新安装
                    logger.warning(f"[EXECUTE_CODE] Attempting reinstall via session.run()...")
                    import_statements = "\n".join([f"import {lib}" for lib in all_libraries])
                    reinstall_result = await asyncio.to_thread(
                        session.run,
                        import_statements,
                        all_libraries,
                        300
                    )
                    logger.info(f"[EXECUTE_CODE] Reinstall result - exit_code: {reinstall_result.exit_code}")
                    logger.info(f"[EXECUTE_CODE] Reinstall stdout:\n{reinstall_result.stdout}")
                    
                    if reinstall_result.exit_code != 0:
                        logger.error(f"[EXECUTE_CODE] ❌ Reinstall also FAILED!")
                        logger.error(f"[EXECUTE_CODE] Reinstall stderr:\n{reinstall_result.stderr}")
                        # 继续执行，让用户看到完整的错误信息
                    else:
                        logger.info(f"[EXECUTE_CODE] ✅ Reinstall succeeded!")
                        all_libraries = []  # 清空，避免重复安装
                else:
                    logger.info(f"[EXECUTE_CODE] ✅ All libraries pre-installed and verified successfully!")
                    all_libraries = []  # 清空，避免重复安装
                    
            except Exception as e:
                logger.error(f"[EXECUTE_CODE] Pre-installation failed with exception: {e}", exc_info=True)
                # 继续执行，让用户看到完整的错误信息
        
        # ✅ 新增：检查是否需要添加 mcp-server 路径到 sys.path
        # 方式1：检查 session 绑定中的标记
        has_mcp_server = False
        async with _session_lock:
            if session_id in _session_bindings:
                has_mcp_server = _session_bindings[session_id].get("has_mcp_server", False)
                logger.info(f"[EXECUTE_CODE] Session 标记 has_mcp_server: {has_mcp_server}")

        # 方式2：如果没有标记，再检查容器内目录
        if not has_mcp_server:
            logger.info(f"[EXECUTE_CODE] 检查容器内是否存在 mcp-server 目录...")
            try:
                check_mcp_dir = await asyncio.to_thread(
                    session.execute_command,
                    "test -d /sandbox/mcp-server && echo 'exists' || echo 'not_exists'"
                )
                logger.info(f"[EXECUTE_CODE] 目录检查结果: stdout='{check_mcp_dir.stdout}', stderr='{check_mcp_dir.stderr}', exit_code={check_mcp_dir.exit_code}")
                has_mcp_server = check_mcp_dir.stdout.strip() == 'exists'
                logger.info(f"[EXECUTE_CODE] 目录检查判断结果: has_mcp_server={has_mcp_server}")
            except Exception as e:
                logger.error(f"[EXECUTE_CODE] 目录检查失败: {e}")
                has_mcp_server = False

        if has_mcp_server:
            logger.info(f"[EXECUTE_CODE] ✅ 检测到 mcp-server 目录，将添加到 sys.path")
            # 在用户代码前添加 sys.path 设置
            path_setup_code = """import sys
import os
if '/sandbox/mcp-server/app' not in sys.path:
    sys.path.insert(0, '/sandbox/mcp-server/app')
if '/sandbox/mcp-server' not in sys.path:
    sys.path.insert(0, '/sandbox/mcp-server')
# ✅ 切换工作目录到 mcp-server，确保相对路径 'config/env_configs.yaml' 能正确解析
os.chdir('/sandbox/mcp-server')
"""
            # 将路径设置代码添加到用户代码前面
            code = path_setup_code + "\n" + code
            logger.info(f"[EXECUTE_CODE] ✅ 已添加 sys.path 设置到代码前")
        else:
            logger.warning(f"[EXECUTE_CODE] ⚠️  未检测到 mcp-server 目录，跳过 sys.path 设置")


        # ✅ 执行用户代码
        logger.info(f"[EXECUTE_CODE] Executing user code...")
        logger.info(f"[EXECUTE_CODE] Libraries parameter for session.run(): {all_libraries}")
        logger.debug(f"[EXECUTE_CODE] Code to execute:\n{code[:200]}...")  # 只记录前200字符
        
        result: ExecutionResult = await asyncio.to_thread(
            session.run,
            code,
            all_libraries,
            timeout
        )
        
        logger.info(f"[EXECUTE_CODE] Code execution completed - exit_code: {result.exit_code}")
        logger.info(f"[EXECUTE_CODE] Stdout length: {len(result.stdout)} bytes")
        logger.info(f"[EXECUTE_CODE] Stderr length: {len(result.stderr)} bytes")
        
        if result.exit_code != 0:
            logger.error(f"[EXECUTE_CODE] Execution failed with exit_code: {result.exit_code}")
            logger.error(f"[EXECUTE_CODE] Error output:\n{result.stderr}")
        
        # ... 处理可视化结果和输出（保持原有逻辑）
        if use_artifact and hasattr(result, "plots") and result.plots:
            plot = result.plots[0]
            results.append(
                ImageContent(
                    data=plot.content_base64,
                    mimeType=f"image/{plot.format.value}",
                    type="image",
                )
            )
        
        # ... 处理输出（保持原有逻辑）
        full_stdout = result.stdout
        system_output = full_stdout
        user_content = ""
        system_markers = [
            "Python plot detection setup complete",
            "Installing packages:",
            "Package installation complete",
        ]
        lines = full_stdout.split('\n')
        system_lines = []
        content_lines = []
        is_system_output = True
        for line in lines:
            if any(marker in line for marker in system_markers):
                system_lines.append(line)
            else:
                if line.strip():
                    is_system_output = False
                
                if is_system_output:
                    system_lines.append(line)
                else:
                    content_lines.append(line)
        system_output = '\n'.join(system_lines)
        user_content = '\n'.join(content_lines)

        result_dict = json.loads(result.to_json(include_plots=False))
        result_dict["stdout"] = system_output
        result_dict["content"] = user_content
        result_dict["session_id"] = session_id
        result_dict["status"] = "success"
        results.append(TextContent(text=json.dumps(result_dict, indent=2), type="text"))
        
        logger.info(f"[EXECUTE_CODE] Code execution completed successfully for session: {session_id}")

    except Exception as e:
        logger.exception(f"[EXECUTE_CODE] Error executing code in session {session_id}: {e}")
        error_result = {
            "status": "error",
            "error_type": "execution_error",
            "session_id": session_id,
            "error": str(e),
            "message": "Code execution failed"
        }
        return [TextContent(text=json.dumps(error_result, indent=2), type="text")]

    return results


# ✅ 新增：关闭 session
@mcp.tool()
async def close_session(session_id: str) -> TextContent:
    """Close a debugging session and release its container.
    
    Args:
        session_id: Session ID to close
        
    Returns:
        TextContent: Close result
    """
    global _session_bindings
    
    try:
        async with _session_lock:
            if session_id not in _session_bindings:
                return TextContent(
                    text=json.dumps({
                        "status": "not_found",
                        "session_id": session_id,
                        "message": "Session not found"
                    }),
                    type="text"
                )
            
            binding = _session_bindings[session_id]
            
            # 关闭 session（归还容器到池）
            if binding.get("session"):
                await asyncio.to_thread(binding["session"].close)
            
            # 删除绑定
            del _session_bindings[session_id]
            
            logger.info(f"Session closed: {session_id}")
            
            return TextContent(
                text=json.dumps({
                    "status": "success",
                    "session_id": session_id,
                    "message": "Session closed successfully"
                }),
                type="text"
            )
            
    except Exception as e:
        logger.exception(f"Error closing session {session_id}")
        return TextContent(
            text=json.dumps({"error": str(e), "status": "failed"}),
            type="text"
        )

# ✅ 新增：列出所有活跃 session
@mcp.tool()
async def list_sessions() -> TextContent:
    """List all active debugging sessions.
    
    Returns:
        TextContent: List of active sessions
    """
    try:
        async with _session_lock:
            sessions = []
            current_time = time.time()
            
            for session_id, binding in _session_bindings.items():
                sessions.append({
                    "session_id": session_id,
                    "language": binding["language"],
                    "created_at": binding["created_at"],
                    "last_access": binding["last_access"],
                    "idle_time": int(current_time - binding["last_access"]),
                    "visualization_support": binding.get("use_artifact", False),
                })
            
            result = {
                "total_sessions": len(sessions),
                "sessions": sessions
            }
            
            return TextContent(text=json.dumps(result, indent=2), type="text")
            
    except Exception as e:
        logger.exception("Error listing sessions")
        return TextContent(
            text=json.dumps({"error": str(e)}),
            type="text"
        )

@mcp.tool()
def get_supported_languages() -> TextContent:
    """Get the list of supported languages."""
    return TextContent(text=json.dumps([lang.value for lang in SupportedLanguage], indent=2), type="text")

@mcp.tool()
def get_language_details(language: str) -> TextContent:
    """Get the details of a language."""
    try:
        lang = SupportedLanguage(language)
        lang_details = LANGUAGE_RESOURCES.get(lang)
        return TextContent(
            text=json.dumps(lang_details, indent=2),
            type="text",
        )
    except ValueError:
        return TextContent(
            text=json.dumps({"error": f"Unsupported language: {language}"}),
            type="text",
        )

@mcp.tool()
def get_pool_status(language: str = "python") -> TextContent:
    """Get the status of the container pool for a language."""
    try:
        if language not in _pool_managers:
            return TextContent(
                text=json.dumps({"status": "not_initialized", "language": language}),
                type="text"
            )
        
        pool = _pool_managers[language]
        status = {
            "language": language,
            "status": "active",
            "total_containers": len(pool._containers) if hasattr(pool, '_containers') else 0,
            "available_containers": len(pool._available) if hasattr(pool, '_available') else 0,
            "in_use_containers": len(pool._in_use) if hasattr(pool, '_in_use') else 0,
        }
        
        return TextContent(text=json.dumps(status, indent=2), type="text")
    except Exception as e:
        return TextContent(
            text=json.dumps({"error": str(e)}),
            type="text"
        )

@mcp.resource("sandbox://languages")
def language_details() -> str:
    """Resource containing detailed information about supported languages."""
    return json.dumps(LANGUAGE_RESOURCES, indent=2)

def main() -> None:
    """Set up and run the server."""
    logger.info("Starting MCP server with backend: %s", os.environ.get("BACKEND", "docker"))
    logger.info("Pool configuration: max_size=%s, min_size=%s, idle_timeout=%s", 
                os.environ.get("POOL_MAX_SIZE", "10"),
                os.environ.get("POOL_MIN_SIZE", "0"),
                os.environ.get("POOL_IDLE_TIMEOUT", "7200.0"))
    logger.info("Session timeout: %s seconds (%.1f hours)", SESSION_TIMEOUT, SESSION_TIMEOUT/3600)
    
    # ✅ 修改：注册异步清理函数
    import atexit
    
    async def cleanup_async():
        """异步清理函数"""
        logger.info("Cleaning up sessions and pools...")
        
        # 关闭所有 session
        async with _session_lock:
            for session_id, binding in list(_session_bindings.items()):
                try:
                    if binding.get("session"):
                        # ✅ 修改：使用 asyncio.to_thread
                        await asyncio.to_thread(binding["session"].close)
                    logger.info(f"Closed session: {session_id}")
                except Exception as e:
                    logger.error(f"Error closing session {session_id}: {e}")
        
        # 关闭所有池
        for lang, pool in _pool_managers.items():
            try:
                # ✅ 修改：使用 asyncio.to_thread
                await asyncio.to_thread(pool.close)
                logger.info(f"Closed pool for {lang}")
            except Exception as e:
                logger.error(f"Error closing pool for {lang}: {e}")
    
    def cleanup_sync():
        """同步包装函数，用于 atexit"""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(cleanup_async())
        except RuntimeError:
            # 如果没有运行的事件循环，使用 asyncio.run
            asyncio.run(cleanup_async())
    
    atexit.register(cleanup_sync)
    
    mcp.run()

if __name__ == "__main__":
    main()