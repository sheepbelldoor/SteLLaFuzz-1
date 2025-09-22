import asyncio
from datetime import datetime

import importlib, yaml, json
from pathlib import Path
from typing import Dict, Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

import argparse
import os
import traceback
import chromadb

from stellafuzz_mcp.client import MCPClient
from utils import RESULT_PATH, printer, format_assistant_responses

# Initialize
load_dotenv()
chroma_client = chromadb.Client()

async def main(target: str, seed_dir: str):

    client = MCPClient()
    try:
        # SteLLaFuzz MCP 서버에 연결
        await client.connect_to_python_server("stellafuzz_mcp/server.py", 
                                              {"SEED_DIR": seed_dir,
                                               "PATH_TO_DB": RESULT_PATH})
                                            #    "PATH_TO_DB": "agent_runs/2025-09-19_11-00-10"})
        # SteLLaFuzz 시작
        await client.stellafuzz(target, seed_dir)
        # await client.test_llm()
    finally:
        # 정리 작업 수행
        await client.cleanup()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True, help='Target for the agents to interact with')
    parser.add_argument('--seed_dir', type=str, default="", help='Directory for seed files')
    args = parser.parse_args()
    asyncio.run(main(args.target, args.seed_dir))
