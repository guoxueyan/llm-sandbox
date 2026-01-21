#!/bin/bash

export POOL_MAX_SIZE=100
export POOL_MIN_SIZE=10
export PREWARM_LANGUAGES=python

sudo -E python3 -m llm_sandbox.mcp_server.mcp_server