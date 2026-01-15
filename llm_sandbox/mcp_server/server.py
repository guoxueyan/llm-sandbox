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

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from llm_sandbox import SupportedLanguage
from llm_sandbox.data import ExecutionResult
from llm_sandbox.mcp_server.const import LANGUAGE_RESOURCES
from llm_sandbox.const import SandboxBackend
from llm_sandbox.session import _check_dependency

from llm_sandbox.pool import create_pool_manager, PooledSandboxSession, ArtifactPooledSandboxSession, PoolConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("llm-sandbox-mcp")

mcp = FastMCP("llm-sandbox")

# ✅ 全局容器池管理器（按语言分组）
_pool_managers: Dict[str, any] = {}

# ✅ Session 绑定管理器（session_id -> 容器绑定信息）
_session_bindings: Dict[str, Dict] = {}

# 线程锁
_pool_lock = threading.Lock()
_session_lock = threading.Lock()

# Session 配置
SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", "1800"))  # 30 分钟
SESSION_CLEANUP_INTERVAL = int(os.environ.get("SESSION_CLEANUP_INTERVAL", "300"))  # 5 分钟

def _get_backend() -> SandboxBackend:
    """Get the backend to use for the sandbox session."""
    backend = SandboxBackend(os.environ.get("BACKEND", "docker"))
    _check_dependency(backend)
    return backend

def _get_pool_config() -> PoolConfig:
    """Get pool configuration from environment variables."""
    return PoolConfig(
        max_pool_size=int(os.environ.get("POOL_MAX_SIZE", "10")),
        min_pool_size=int(os.environ.get("POOL_MIN_SIZE", "2")),
        max_idle_time=int(os.environ.get("POOL_MAX_IDLE_TIME", "600")),
        exhaustion_strategy=os.environ.get("POOL_EXHAUSTION_STRATEGY", "wait"),
    )

def _get_common_libraries(language: str) -> list[str]:
    """Get common libraries to pre-install for a language."""
    common_libs = {
        "python": ["numpy", "pandas", "matplotlib", "requests", "pydantic"],
        "javascript": ["lodash", "axios"],
    }
    return common_libs.get(language, [])

def _get_or_create_pool(language: str):
    """Get or create a container pool for the specified language."""
    global _pool_managers
    
    with _pool_lock:
        if language in _pool_managers:
            return _pool_managers[language]
        
        logger.info(f"Creating container pool for language: {language}")
        
        preinstall_libs_env = os.environ.get(f"PREINSTALL_LIBS_{language.upper()}", "")
        if preinstall_libs_env:
            preinstall_libs = [lib.strip() for lib in preinstall_libs_env.split(",")]
        else:
            preinstall_libs = _get_common_libraries(language)
        
        logger.info(f"Pre-installing libraries for {language}: {preinstall_libs}")
        
        pool = create_pool_manager(
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

def _cleanup_expired_sessions():
    """Clean up expired sessions (background task)."""
    global _session_bindings
    
    while True:
        try:
            time.sleep(SESSION_CLEANUP_INTERVAL)
            
            current_time = time.time()
            expired_sessions = []
            
            with _session_lock:
                for session_id, binding in _session_bindings.items():
                    if current_time - binding["last_access"] > SESSION_TIMEOUT:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    logger.info(f"Cleaning up expired session: {session_id}")
                    binding = _session_bindings[session_id]
                    
                    # 关闭 session（归还容器到池）
                    if binding.get("session"):
                        try:
                            binding["session"].close()
                        except Exception as e:
                            logger.error(f"Error closing session {session_id}: {e}")
                    
                    del _session_bindings[session_id]
                
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                    
        except Exception as e:
            logger.error(f"Error in session cleanup: {e}")

# ✅ 启动后台清理线程
_cleanup_thread = threading.Thread(target=_cleanup_expired_sessions, daemon=True)
_cleanup_thread.start()

def _get_or_create_session(session_id: str, language: str, use_artifact: bool = False) -> PooledSandboxSession:
    """Get or create a session bound to a specific session_id.
    
    Args:
        session_id: Unique session identifier
        language: Programming language
        use_artifact: Whether to use artifact session (for visualization)
        
    Returns:
        PooledSandboxSession bound to this session_id
    """
    global _session_bindings
    
    with _session_lock:
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
        
        pool = _get_or_create_pool(language)
        session_cls = ArtifactPooledSandboxSession if use_artifact else PooledSandboxSession
        
        # 创建并打开 session
        session = session_cls(pool_manager=pool)
        session.open()  # ✅ 手动打开，不使用 with 语句
        
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

# ✅ 新增：创建 session
@mcp.tool()
def create_session(language: str = "python") -> TextContent:
    """Create a new debugging session.
    
    Args:
        language: Programming language for the session
        
    Returns:
        TextContent: Session information including session_id
    """
    try:
        # 生成唯一的 session_id
        session_id = str(uuid.uuid4())
        
        # 判断是否需要可视化支持
        use_artifact = _supports_visualization(language)
        
        # 创建 session（会自动绑定容器）
        session = _get_or_create_session(session_id, language, use_artifact)
        
        result = {
            "session_id": session_id,
            "language": language,
            "visualization_support": use_artifact,
            "status": "created",
            "message": "Session created successfully. Use this session_id for subsequent execute_code calls."
        }
        
        return TextContent(text=json.dumps(result, indent=2), type="text")
        
    except Exception as e:
        logger.exception("Error creating session")
        return TextContent(
            text=json.dumps({"error": str(e), "status": "failed"}),
            type="text"
        )

@mcp.tool()
def execute_code(
    code: str,
    session_id: Optional[str] = None,  # ✅ 新增：session_id 参数
    language: str = "python",
    libraries: list[str] | None = None,
    timeout: int = 30,
) -> list[ImageContent | TextContent]:
    """Execute code in a secure sandbox environment with session support.

    Args:
        code: The code to execute
        session_id: Optional session ID to bind execution to a specific container.
                   If not provided, a temporary session will be used.
        language: Programming language (python, javascript, java, cpp, go, r, ruby)
        libraries: List of libraries/packages to install (if not pre-installed)
        timeout: Execution timeout in seconds (default: 30)

    Returns:
        List of content items including execution results and any generated visualizations

    """
    results: list[ImageContent | TextContent] = []

    try:
        # ✅ 如果提供了 session_id，使用绑定的 session
        if session_id:
            logger.info(f"Executing code in session: {session_id}")
            use_artifact = _supports_visualization(language)
            session = _get_or_create_session(session_id, language, use_artifact)
            
            # 执行代码（不使用 with，因为 session 需要保持打开）
            result = session.run(
                code=code,
                libraries=libraries or [],
                timeout=timeout,
                clear_plots=False,
            )
            
            # 处理结果
            if use_artifact and hasattr(result, "plots") and result.plots:
                plot = result.plots[0]
                results.append(
                    ImageContent(
                        data=plot.content_base64,
                        mimeType=f"image/{plot.format.value}",
                        type="image",
                    )
                )
            
            results.append(TextContent(text=result.to_json(include_plots=False), type="text"))
            
        else:
            # ✅ 没有 session_id，使用临时 session（原有逻辑）
            logger.info("Executing code in temporary session")
            pool = _get_or_create_pool(language)
            use_artifact = _supports_visualization(language)
            session_cls = ArtifactPooledSandboxSession if use_artifact else PooledSandboxSession
            
            with session_cls(pool_manager=pool) as session:
                result = session.run(
                    code=code,
                    libraries=libraries or [],
                    timeout=timeout,
                    clear_plots=False,
                )
                
                if use_artifact and hasattr(result, "plots") and result.plots:
                    plot = result.plots[0]
                    results.append(
                        ImageContent(
                            data=plot.content_base64,
                            mimeType=f"image/{plot.format.value}",
                            type="image",
                        )
                    )
                
                results.append(TextContent(text=result.to_json(include_plots=False), type="text"))

    except Exception as e:
        logger.exception("Error executing code")
        return [TextContent(text=ExecutionResult(exit_code=1, stderr=str(e)).to_json(), type="text")]

    else:
        return results

# ✅ 新增：关闭 session
@mcp.tool()
def close_session(session_id: str) -> TextContent:
    """Close a debugging session and release its container.
    
    Args:
        session_id: Session ID to close
        
    Returns:
        TextContent: Close result
    """
    global _session_bindings
    
    try:
        with _session_lock:
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
                binding["session"].close()
            
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
def list_sessions() -> TextContent:
    """List all active debugging sessions.
    
    Returns:
        TextContent: List of active sessions
    """
    try:
        with _session_lock:
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
    logger.info("Pool configuration: max_size=%s, min_size=%s", 
                os.environ.get("POOL_MAX_SIZE", "10"),
                os.environ.get("POOL_MIN_SIZE", "2"))
    logger.info("Session timeout: %s seconds", SESSION_TIMEOUT)
    
    # 注册清理函数
    import atexit
    def cleanup():
        logger.info("Cleaning up sessions and pools...")
        
        # 关闭所有 session
        with _session_lock:
            for session_id, binding in list(_session_bindings.items()):
                try:
                    if binding.get("session"):
                        binding["session"].close()
                    logger.info(f"Closed session: {session_id}")
                except Exception as e:
                    logger.error(f"Error closing session {session_id}: {e}")
        
        # 关闭所有池
        for lang, pool in _pool_managers.items():
            try:
                pool.close()
                logger.info(f"Closed pool for {lang}")
            except Exception as e:
                logger.error(f"Error closing pool for {lang}: {e}")
    
    atexit.register(cleanup)
    
    mcp.run()

if __name__ == "__main__":
    main()