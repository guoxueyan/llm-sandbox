"""FastAPI HTTP Server for LLM Sandbox MCP.

Provides HTTP endpoints to interact with the MCP server's session management
and code execution capabilities.
"""

import json
import logging
from typing import Optional, List
from contextlib import asynccontextmanager
import asyncio
import time
import os
import subprocess
from pathlib import Path
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# 导入 server.py 中的函数
from llm_sandbox.mcp_server.server import (
    create_session as mcp_create_session,
    execute_code as mcp_execute_code,
    close_session as mcp_close_session,
    list_sessions as mcp_list_sessions,
    get_supported_languages as mcp_get_supported_languages,
)

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
    logging.getLogger("llm-sandbox-http").setLevel(logging.INFO)
    logging.getLogger("llm-sandbox-mcp").setLevel(logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    
    return root_logger

# ✅ 初始化日志（在导入后立即执行）
setup_logging()
logger = logging.getLogger("llm-sandbox-http")

# ==================== Lifespan Event Handler ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    logger.info("Initializing container pools on startup...")
    
    # 从环境变量读取需要预热的语言列表
    prewarm_languages = os.environ.get("PREWARM_LANGUAGES", "python").split(",")
    prewarm_languages = [lang.strip() for lang in prewarm_languages]
    
    logger.info(f"Pre-warming container pools for languages: {prewarm_languages}")
    
    # 为每个语言创建容器池并等待预热完成
    for language in prewarm_languages:
        try:
            logger.info(f"Creating container pool for {language}...")
            # 调用 server.py 中的内部函数来创建池
            from llm_sandbox.mcp_server.server import _get_or_create_pool
            pool = _get_or_create_pool(language)
            logger.info(f"Container pool for {language} created, waiting for pre-warming...")
            
            # ✅ 等待容器池预热完成
            min_size = int(os.environ.get("POOL_MIN_SIZE", "10"))
            max_wait_time = int(os.environ.get("PREWARM_TIMEOUT", "300"))  # 默认最多等待 5 分钟
            start_time = time.time()
            
            while True:
                stats = pool.get_stats()
                idle_count = stats["state_counts"].get("idle", 0)
                total_count = stats["total_size"]
                
                logger.info(f"Pre-warming progress for {language}: {idle_count}/{min_size} idle containers ready (total: {total_count})")
                
                # 检查是否达到最小池大小
                if idle_count >= min_size:
                    logger.info(f"✅ Container pool for {language} pre-warming completed! {idle_count} containers ready.")
                    break
                
                # 检查超时
                elapsed = time.time() - start_time
                if elapsed > max_wait_time:
                    logger.warning(f"⚠️  Pre-warming timeout for {language} after {elapsed:.1f}s. Only {idle_count}/{min_size} containers ready. Continuing anyway...")
                    break
                
                # 等待一段时间后再检查
                await asyncio.sleep(2)  # 每 2 秒检查一次
            
        except Exception as e:
            logger.error(f"Failed to create pool for {language}: {e}")
    
    logger.info("🚀 All container pools pre-warmed and ready!")
    
    yield  # 服务运行期间
    
    # Shutdown
    logger.info("Shutting down container pools...")
    
    from llm_sandbox.mcp_server.server import _pool_managers
    
    for lang, pool in _pool_managers.items():
        try:
            pool.close()
            logger.info(f"Closed pool for {lang}")
        except Exception as e:
            logger.error(f"Error closing pool for {lang}: {e}")

# 创建 FastAPI 应用
app = FastAPI(
    title="LLM Sandbox HTTP API",
    description="HTTP API for secure code execution with session management",
    version="1.0.0",
    lifespan=lifespan,
)


# ==================== Request/Response Models ====================

class CreateSessionRequest(BaseModel):
    """Request model for creating a new session."""
    language: str = Field(default="python", description="Programming language for the session")
    libraries: Optional[List[str]] = Field(default=None, description="List of libraries to pre-install")  # ✅ 新增字段


class CreateSessionResponse(BaseModel):
    """Response model for session creation."""
    status: str
    session_id: str
    language: str
    visualization_support: bool
    timeout: int
    message: str


class ExecuteCodeRequest(BaseModel):
    """Request model for code execution."""
    code: str = Field(..., description="Code to execute")
    session_id: str = Field(..., description="Session ID (required)")
    libraries: Optional[List[str]] = Field(default=None, description="Libraries to install")
    operation: Optional[bool] = Field(default=False, description="Whether to perform Git operations (optional, default False)")
    timeout: int = Field(default=30, description="Execution timeout in seconds")


class ExecuteCodeResponse(BaseModel):
    """Response model for code execution."""
    status: str
    session_id: str
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None


class CloseSessionRequest(BaseModel):
    """Request model for closing a session."""
    session_id: str = Field(..., description="Session ID to close")


class ErrorResponse(BaseModel):
    """Standard error response."""
    status: str = "error"
    error: str
    message: str


# ==================== HTTP Endpoints ====================

@app.post(
    "/api/v1/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new session",
    description="Create a new debugging session with a dedicated container"
)
async def create_session(request: CreateSessionRequest):
    """Create a new session endpoint.
    
    Args:
        request: CreateSessionRequest containing language preference and optional libraries
        
    Returns:
        CreateSessionResponse with session details
        
    Raises:
        HTTPException: If session creation fails
    """
    try:
        logger.info(f"Creating session for language: {request.language}, libraries: {request.libraries}")
        
        # 调用 MCP server 的 create_session 函数
        result = mcp_create_session(language=request.language, libraries=request.libraries)  # ✅ 传递 libraries 参数
        
        # 解析返回的 TextContent
        result_data = json.loads(result.text)
        
        if result_data.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result_data
            )
        
        logger.info(f"Session created: {result_data.get('session_id')}")
        return JSONResponse(content=result_data, status_code=status.HTTP_201_CREATED)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error creating session")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "error": str(e),
                "message": "Failed to create session"
            }
        )

# ✅ 新增：Git 操作辅助函数
def is_git_error(returncode: int, stderr: str, stdout: str) -> bool:
    """判断 git 命令是否真的失败了"""
    if returncode == 0:
        return False
    
    error_keywords = ["error:", "fatal:", "ERROR", "FATAL", "failed"]
    combined_output = (stderr + stdout).lower()
    
    if any(keyword.lower() in combined_output for keyword in error_keywords):
        return True
    
    return False

def check_git_conflicts(output: str) -> bool:
    """检查 git pull 输出中是否有冲突"""
    conflict_keywords = ["CONFLICT", "conflict", "Merge conflict"]
    has_conflict = any(keyword in output for keyword in conflict_keywords)
    if has_conflict:
        logger.warning(f"检测到 Git 冲突: {output}")
    else:
        logger.info("未检测到 Git 冲突")
    return has_conflict

# ✅ 新增：处理 Git 操作的异步函数
async def _handle_git_operation(code: str, session_id: str, session):
    """在容器内处理 Git 操作：根据目录是否存在自动选择克隆或更新
    
    Args:
        code: 要写入的代码内容
        session_id: Session ID
        session: Session 对象（容器会话）
        
    Raises:
        Exception: 如果 Git 操作失败
    """
    tool_name = "sandbox_generated_tool"
    
    logger.info(f"[GIT_OPERATION] 开始在容器内处理 Git 操作: tool_name={tool_name}, session_id={session_id}")
    
    # 仓库配置
    REPO_URL = os.environ.get(
        "MCP_SERVER_REPO_URL", 
        "http://gxy01953892:qq2510725526.@gitlab.alibaba-inc.com/edu-quark/mcp-server.git"
    )
    REPO_BRANCH = "medical"
    
    # ✅ 容器内的路径
    REPO_DIR = "/sandbox/mcp-server"
    TOOLS_DIR = f"{REPO_DIR}/app/tools"
    
    logger.info(f"[GIT_OPERATION] 容器内工作目录: {REPO_DIR}")
    
    try:
        # ✅ 检查容器内目录是否存在
        check_dir_result = session.execute_command(f"test -d {REPO_DIR} && echo 'exists' || echo 'not_exists'")
        dir_exists = check_dir_result.stdout.strip() == 'exists'
        
        if not dir_exists:
            # ========== 目录不存在：执行克隆操作 ==========
            logger.info(f"[GIT_OPERATION] 容器内目录不存在，执行克隆操作...")
            
            # 在容器内执行 git clone
            clone_cmd = f"sh -c 'cd /sandbox && git clone -b {REPO_BRANCH} {REPO_URL} mcp-server'"
            logger.info(f"[GIT_OPERATION] 执行命令: {clone_cmd}")
            
            clone_result = session.execute_command(clone_cmd)
            
            logger.info(f"[GIT_OPERATION] git clone 返回码: {clone_result.exit_code}")
            logger.info(f"[GIT_OPERATION] git clone stdout: {clone_result.stdout}")
            
            if clone_result.exit_code != 0:
                error_msg = f"git clone 失败: {clone_result.stderr}"
                logger.error(f"[GIT_OPERATION] {error_msg}")
                raise Exception(error_msg)
            
            logger.info(f"[GIT_OPERATION] 容器内仓库克隆成功: {REPO_DIR}")
            
        else:
            # ========== 目录存在：执行更新操作 ==========
            logger.info(f"[GIT_OPERATION] 容器内目录已存在，执行更新操作...")
            
            # 步骤1: checkout 掉所有未提交的文件
            logger.info("[GIT_OPERATION] 步骤1: checkout 所有未提交文件...")
            checkout_all_cmd = f"cd {REPO_DIR} && git checkout ."
            checkout_all_result = session.execute_command(checkout_all_cmd)
            
            logger.info(f"[GIT_OPERATION] git checkout . 返回码: {checkout_all_result.exit_code}")
            
            if checkout_all_result.exit_code != 0:
                error_msg = f"git checkout . 失败: {checkout_all_result.stderr}"
                logger.error(f"[GIT_OPERATION] {error_msg}")
                raise Exception(error_msg)
            
            logger.info("[GIT_OPERATION] 成功 checkout 所有未提交文件")
            
            # 步骤2: 切换到 medical 分支
            logger.info(f"[GIT_OPERATION] 步骤2: 切换到 {REPO_BRANCH} 分支...")
            checkout_cmd = f"cd {REPO_DIR} && git checkout {REPO_BRANCH}"
            checkout_result = session.execute_command(checkout_cmd)
            
            logger.info(f"[GIT_OPERATION] git checkout {REPO_BRANCH} 返回码: {checkout_result.exit_code}")
            
            if checkout_result.exit_code != 0:
                error_msg = f"git checkout {REPO_BRANCH} 失败: {checkout_result.stderr}"
                logger.error(f"[GIT_OPERATION] {error_msg}")
                raise Exception(error_msg)
            
            logger.info(f"[GIT_OPERATION] 成功切换到 {REPO_BRANCH} 分支")
            
            # 步骤3: 拉取最新代码
            logger.info(f"[GIT_OPERATION] 步骤3: 拉取 {REPO_BRANCH} 分支最新代码...")
            pull_cmd = f"cd {REPO_DIR} && git pull origin {REPO_BRANCH}"
            pull_result = session.execute_command(pull_cmd)
            
            logger.info(f"[GIT_OPERATION] git pull 返回码: {pull_result.exit_code}")
            logger.info(f"[GIT_OPERATION] git pull stdout: {pull_result.stdout}")
            
            if is_git_error(pull_result.exit_code, pull_result.stderr, pull_result.stdout):
                error_msg = f"git pull 失败: {pull_result.stderr}"
                logger.error(f"[GIT_OPERATION] {error_msg}")
                raise Exception(error_msg)
            
            # 步骤4: 检查冲突
            logger.info("[GIT_OPERATION] 步骤4: 检查是否有冲突...")
            combined_output = pull_result.stdout + pull_result.stderr
            if check_git_conflicts(combined_output):
                error_msg = "检测到 Git 冲突，请手动解决冲突后再试"
                logger.error(f"[GIT_OPERATION] {error_msg}")
                raise Exception(error_msg)
            
            logger.info("[GIT_OPERATION] 仓库更新成功，无冲突")
        
        # ✅ 确保 tools 目录存在
        logger.info(f"[GIT_OPERATION] 确保工具目录存在: {TOOLS_DIR}")
        mkdir_result = session.execute_command(f"mkdir -p {TOOLS_DIR}")
        if mkdir_result.exit_code != 0:
            logger.error(f"[GIT_OPERATION] 创建目录失败: {mkdir_result.stderr}")
            raise Exception(f"创建目录失败: {mkdir_result.stderr}")
        
        # ✅ 写入或更新工具文件
        tool_file_path = f"{TOOLS_DIR}/{tool_name}.py"
        
        # 检查文件是否存在
        check_file_result = session.execute_command(f"test -f {tool_file_path} && echo 'exists' || echo 'not_exists'")
        file_exists = check_file_result.stdout.strip() == 'exists'
        
        if file_exists:
            logger.info(f"[GIT_OPERATION] 文件已存在，更新代码到文件: {tool_file_path}")
        else:
            logger.info(f"[GIT_OPERATION] 文件不存在，创建新文件: {tool_file_path}")
        
        # 使用 base64 编码写入文件（避免特殊字符问题）
        import base64
        content_b64 = base64.b64encode(code.encode('utf-8')).decode('ascii')
        write_cmd = f"echo '{content_b64}' | base64 -d > {tool_file_path}"
        
        write_result = session.execute_command(write_cmd)
        
        if write_result.exit_code != 0:
            error_msg = f"写入文件失败: {write_result.stderr}"
            logger.error(f"[GIT_OPERATION] {error_msg}")
            raise Exception(error_msg)
        
        # ✅ 新增：在 session 绑定中标记已有 mcp-server 代码（移到验证之前，确保一定执行）
        from llm_sandbox.mcp_server.server import _session_bindings, _session_lock

        with _session_lock:
            if session_id in _session_bindings:
                _session_bindings[session_id]["has_mcp_server"] = True
                logger.info(f"[GIT_OPERATION] ✅ 已标记 session {session_id} 拥有 mcp-server 代码")

        # ✅ 验证文件是否写入成功
        verify_result = session.execute_command(f"test -f {tool_file_path} && echo 'exists' || echo 'not_found'")
        logger.info(f"[GIT_OPERATION] 文件验证结果: {verify_result.stdout.strip()}")
        
    except Exception as e:
        logger.error(f"[GIT_OPERATION] 容器内 Git 操作失败: {e}")
        raise


@app.post(
    "/api/v1/execute",
    response_model=ExecuteCodeResponse,
    summary="Execute code in a session",
    description="Execute code in a secure sandbox environment with session binding"
)
async def execute_code(request: ExecuteCodeRequest):
    """Execute code endpoint.
    
    Args:
        request: ExecuteCodeRequest containing code and session details
        
    Returns:
        ExecuteCodeResponse with execution results
        
    Raises:
        HTTPException: If execution fails or session not found
    """
    try:
        logger.info(f"Executing code in session: {request.session_id}")
        
        # ✅ 修改：只检查 operation 是否为 True
        if request.operation is True:
            logger.info(f"检测到 Git 操作请求: operation=True")
            try:
                # ✅ 步骤1: 获取 session 对象
                from llm_sandbox.mcp_server.server import _get_or_create_session, _session_lock, _session_bindings
                
                with _session_lock:
                    if request.session_id not in _session_bindings:
                        logger.error(f"Session not found: {request.session_id}")
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail={
                                "status": "error",
                                "error_type": "session_not_found",
                                "message": f"Session '{request.session_id}' not found"
                            }
                        )
                    
                    language = _session_bindings[request.session_id]["language"]
                    use_artifact = _session_bindings[request.session_id].get("use_artifact", False)
                
                # 获取 session 对象
                session = _get_or_create_session(request.session_id, language, use_artifact)
                
                # ✅ 步骤2: 在容器内执行 Git 操作
                await _handle_git_operation(code=request.code, session_id=request.session_id, session=session)
                logger.info("容器内 Git 操作完成，继续执行代码...")
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"容器内 Git 操作失败: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "status": "error",
                        "error_type": "git_operation_error",
                        "error": str(e),
                        "message": "容器内 Git 操作失败"
                    }
                )
        else:
            logger.info(f"operation={request.operation}，跳过 Git 操作")

        # 调用 MCP server 的 execute_code 函数
        results = mcp_execute_code(
            code=request.code,
            session_id=request.session_id,
            libraries=request.libraries,
            timeout=request.timeout,
        )
        
        # 处理返回结果
        response_data = {}
        images = []
        
        for item in results:
            if item.type == "text":
                # 解析文本内容
                text_data = json.loads(item.text)
                response_data.update(text_data)
            elif item.type == "image":
                # 收集图片数据
                images.append({
                    "data": item.data,
                    "mime_type": item.mimeType
                })
        
        # 如果有图片，添加到响应中
        if images:
            response_data["images"] = images
        
        # 检查错误状态
        if response_data.get("status") == "error":
            error_type = response_data.get("error_type", "execution_error")
            
            if error_type == "session_not_found":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=response_data
                )
            elif error_type == "missing_session_id":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=response_data
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=response_data
                )
        
        logger.info(f"Code executed successfully in session: {request.session_id}")
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error executing code in session {request.session_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "error_type": "execution_error",
                "session_id": request.session_id,
                "error": str(e),
                "message": "Code execution failed"
            }
        )


@app.delete(
    "/api/v1/sessions/{session_id}",
    summary="Close a session",
    description="Close a debugging session and release its container"
)
async def close_session(session_id: str):
    """Close session endpoint.
    
    Args:
        session_id: Session ID to close
        
    Returns:
        Close result
        
    Raises:
        HTTPException: If session not found
    """
    try:
        logger.info(f"Closing session: {session_id}")
        
        # 调用 MCP server 的 close_session 函数
        result = mcp_close_session(session_id=session_id)
        result_data = json.loads(result.text)
        
        if result_data.get("status") == "not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result_data
            )
        
        logger.info(f"Session closed: {session_id}")
        return JSONResponse(content=result_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error closing session {session_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "error": str(e),
                "message": "Failed to close session"
            }
        )


@app.get(
    "/api/v1/sessions",
    summary="List all sessions",
    description="List all active debugging sessions"
)
async def list_sessions():
    """List sessions endpoint.
    
    Returns:
        List of active sessions
    """
    try:
        logger.info("Listing all sessions")
        
        # 调用 MCP server 的 list_sessions 函数
        result = mcp_list_sessions()
        result_data = json.loads(result.text)
        
        return JSONResponse(content=result_data)
        
    except Exception as e:
        logger.exception("Error listing sessions")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "error": str(e),
                "message": "Failed to list sessions"
            }
        )


@app.get(
    "/api/v1/languages",
    summary="Get supported languages",
    description="Get the list of supported programming languages"
)
async def get_supported_languages():
    """Get supported languages endpoint.
    
    Returns:
        List of supported languages
    """
    try:
        logger.info("Getting supported languages")
        
        # 调用 MCP server 的 get_supported_languages 函数
        result = mcp_get_supported_languages()
        languages = json.loads(result.text)
        
        return JSONResponse(content={"languages": languages})
        
    except Exception as e:
        logger.exception("Error getting supported languages")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "error",
                "error": str(e),
                "message": "Failed to get supported languages"
            }
        )


@app.get(
    "/health",
    summary="Health check",
    description="Check if the server is running"
)
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "llm-sandbox-http"}


@app.get(
    "/",
    summary="Root endpoint",
    description="API information"
)
async def root():
    """Root endpoint with API information."""
    return {
        "service": "LLM Sandbox HTTP API",
        "version": "1.0.0",
        "endpoints": {
            "create_session": "POST /api/v1/sessions",
            "execute_code": "POST /api/v1/execute",
            "close_session": "DELETE /api/v1/sessions/{session_id}",
            "list_sessions": "GET /api/v1/sessions",
            "supported_languages": "GET /api/v1/languages",
            "health": "GET /health"
        },
        "docs": "/docs"
    }


# ==================== Server Startup ====================

if __name__ == "__main__":
    import uvicorn
    
    # 从环境变量读取配置
    import os
    host = os.environ.get("HTTP_HOST", "0.0.0.0")
    port = int(os.environ.get("HTTP_PORT", "8000"))
    
    logger.info(f"Starting HTTP server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )