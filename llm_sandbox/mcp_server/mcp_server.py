"""FastAPI HTTP Server for LLM Sandbox MCP.

Provides HTTP endpoints to interact with the MCP server's session management
and code execution capabilities.
"""

import json
import logging
from typing import Optional, List

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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
)
logger = logging.getLogger("llm-sandbox-http")

# 创建 FastAPI 应用
app = FastAPI(
    title="LLM Sandbox HTTP API",
    description="HTTP API for secure code execution with session management",
    version="1.0.0",
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
    timeout: int = Field(default=30, description="Execution timeout in seconds")
    auto_install: bool = Field(default=True, description="Auto-detect and install dependencies") 

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
        
        # 调用 MCP server 的 execute_code 函数
        results = mcp_execute_code(
            code=request.code,
            session_id=request.session_id,
            libraries=request.libraries,
            timeout=request.timeout,
            auto_install=request.auto_install,
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