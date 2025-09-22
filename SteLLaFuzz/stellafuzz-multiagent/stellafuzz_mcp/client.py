from email import utils
import os
import traceback
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from typing import Optional, Dict, Any
import asyncio
import json
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.shared_params.function_definition import FunctionDefinition
import chromadb
from dotenv import load_dotenv

from utils import RESULT_PATH, printer, format_assistant_responses
from agents.format_analyst import FORMAT_ANALYST
from agents.sequence_planner import SEQUENCE_PLANNER
from agents.field_designer import FIELD_DESIGNER
from agents.developer import DEVELOPER
from agents.tester import TESTER

load_dotenv()

class MCPClient:
    """
    MCP (Model Context Protocol) 클라이언트 클래스
    OpenAI API와 MCP 서버 간의 통신을 관리
    """
    
    def __init__(self):
        """MCP 클라이언트 초기화"""
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm = OpenAI()

    async def connect_to_server(self, command: str, args: list[str], env: dict = None):
        """
        커스텀 명령어와 인자로 MCP 서버에 연결
        Args:
            command: 실행할 명령어
            args: 명령어 인자들
            env: 환경 변수 (선택사항)
        """
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        # 사용 가능한 도구들 목록 출력
        response = await self.session.list_tools()
        tools = response.tools
        printer.print(f"\nConnected to server ({command} {' '.join(args)}) with tools:", [tool.name for tool in tools])

    async def connect_to_python_server(self, server_script_path: str, env: dict = None):
        """
        Python MCP 서버에 연결하는 헬퍼 메서드
        Args:
            server_script_path: 서버 스크립트 경로
            env: 환경 변수 (선택사항)
        """
        await self.connect_to_server("python", [server_script_path], env=env)

    async def cleanup(self):
        """리소스 정리"""
        await self.exit_stack.aclose()

    async def _available_tools(self) -> list[ChatCompletionToolParam]:
        """
        사용 가능한 도구들을 OpenAI 형식으로 변환
        Returns:
            OpenAI 도구 파라미터 리스트
        """
        response = await self.session.list_tools()
        return [
            ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=tool.name,
                    description=tool.description if tool.description else "",
                    parameters=tool.inputSchema
                )
            )
            for tool in response.tools
        ]

    async def process_tool_call(self, tool_call) -> ChatCompletionToolMessageParam:
        """
        도구 호출을 처리하고 결과를 반환
        Args:
            tool_call: 호출할 도구 정보
        Returns:
            도구 호출 결과 메시지
        """
        assert tool_call['type'] == "function"

        tool_name = tool_call['function']['name']
        tool_args = json.loads(tool_call['function']['arguments'] or "{}")

        # 최대 5번까지 재시도
        max_try = 5
        error_message = ""
        for t in range(max_try):
            call_tool_result = await self.session.call_tool(tool_name, tool_args)
            if call_tool_result.isError:
                if t == max_try - 1:
                    error_message = f"[ERROR] Tool call failed: {call_tool_result}"
            else:
                break

        results = []
        if call_tool_result.isError:
            results.append(error_message)
        else:
            for result in call_tool_result.content:
                if result.type == "text":
                    results.append(result.text[:256000])  # 텍스트 길이 제한
                else:
                    raise NotImplementedError(f"Unsupported result type: {result.type}")

        return ChatCompletionToolMessageParam(
            role="tool",
            content=json.dumps({
                **tool_args,
                tool_name: results
            }),
            tool_call_id=tool_call['id']
        )

    async def process_messages_streaming(self, messages: list[ChatCompletionMessageParam]):
        """
        메시지들을 스트리밍 방식으로 처리
        Args:
            messages: 처리할 메시지 리스트
        Returns:
            업데이트된 메시지 리스트
        """
        available_tools = await self._available_tools()

        # OpenAI 스트리밍 요청 생성
        stream = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=available_tools,
            tool_choice="auto",
            stream=True,
        )

        assistant_text_parts: list[str] = []
        tool_calls_acc: Dict[int, Dict[str, Any]] = {}
        finish_reason: Optional[str] = None

        printer.print("\nAgent: ", end="", flush=True)

        # 스트림 이벤트 처리
        for event in stream:
            choice = event.choices[0]
            delta = choice.delta

            # 텍스트 콘텐츠 처리
            if delta.content:
                printer.print(delta.content, end="", flush=True)
                assistant_text_parts.append(delta.content)

            # 도구 호출 처리
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    slot = tool_calls_acc.setdefault(idx, {"id": None, "type": tc.type, "function": {"name": "", "arguments": ""}})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["function"]["name"] = tc.function.name
                        if tc.function.arguments:
                            slot["function"]["arguments"] += tc.function.arguments

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        printer.print("", flush=True)

        # 완료 이유에 따른 처리
        if finish_reason == "stop":
            # 일반 텍스트 응답 완료
            messages.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content="".join(assistant_text_parts)
                )
            )
            return messages

        if finish_reason == "tool_calls":
            # 도구 호출 응답 처리
            assistant_tool_calls = []
            for idx in sorted(tool_calls_acc.keys()):
                slot = tool_calls_acc[idx]
                assistant_tool_calls.append(
                    ChatCompletionMessageToolCallParam(
                        id=slot["id"] or f"tool_{idx}",
                        type=slot["type"] or "function",
                        function=Function(
                            name=slot["function"]["name"],
                            arguments=slot["function"]["arguments"]
                        )
                    )
                )

            messages.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant",
                    tool_calls=assistant_tool_calls
                )
            )

            # 도구 호출들을 병렬로 실행
            tasks = [asyncio.create_task(self.process_tool_call(tc)) for tc in assistant_tool_calls]
            tool_outputs = await asyncio.gather(*tasks)
            printer.print(format_assistant_responses(tool_outputs))
            messages.extend(tool_outputs)

            # 재귀적으로 다음 응답 처리
            return await self.process_messages_streaming(messages)

        if finish_reason == "length":
            raise ValueError("[ERROR] Length limit reached while streaming. Try a shorter query.")

        if finish_reason == "content_filter":
            raise ValueError("[ERROR] Content filter triggered while streaming.")

        raise ValueError(f"[ERROR] Unknown finish reason during streaming: {finish_reason}")

    async def stellafuzz(self, target: str, seed_dir: str):
        """
        SteLLaFuzz Seed Generation Process
        Args:
            target: Target protocol or program
            seed_dir: Directory containing seed files
        """
        chroma_client = chromadb.Client()
        type_list = []
        seed_sequence_pairs = {}
        format_spec_DB = chroma_client.create_collection(name="format_spec_DB")
        sequence_DB = chroma_client.create_collection(name="sequence_DB")
        component_DB = chroma_client.create_collection(name="component_DB")
        coverage_DB = chroma_client.create_collection(name="coverage_DB")

        printer.print(f"Generate Seeds for: {target}")

        self.messages: list[ChatCompletionMessageParam] = []

        ## SteLLaFuzz Agent
        # FORMAT ANALYST
        format_analyst = FORMAT_ANALYST(target, 
                                        seed_dir=seed_dir, 
                                        seed_sequence_pairs=seed_sequence_pairs,
                                        format_spec_DB=format_spec_DB,
                                        sequence_DB=sequence_DB,
                                        type_list=type_list)
        spec_analyzing_result = await format_analyst.analyze_from_specification(self)
        print(type_list)
        printer.print('----------------------- Format Analysis Completed -----------------------')
        printer.print(f'* Format Analyst identified the format specification as:\n{spec_analyzing_result}')
        seed_analyzing_result = await format_analyst.analyze_from_inputs(self, input_dir=seed_dir)
        sequence_analyzing_result = await format_analyst.extract_sequence_from_inputs(self, input_dir=seed_dir)
        printer.print('----------------------- Input Analysis Completed -----------------------')
        printer.print(f'* Format Analyst identified the format specification as:\n{seed_analyzing_result}')
        printer.print(f'* Format Analyst identified the sequence as:\n{sequence_analyzing_result}')

        # printer.print(f'* Loaded format_spec_DB and sequence_DB from previous run.')
        # await self.load_memory(format_spec_DB, "format_spec_DB", "agent_runs/2025-09-18_16-40-00-dns")
        # await self.load_memory(sequence_DB, "sequence_DB", "agent_runs/2025-09-18_16-40-00-dns")

        # printer.print(f'* Current format_spec_DB has {len(format_spec_DB.get()["ids"])} entries.')
        # printer.print(f'* Current sequence_DB has {len(sequence_DB.get()["ids"])} entries.')
        
        # SEQUENCE PLANNER
        sequence_planner = SEQUENCE_PLANNER(target, 
                                           seed_dir=seed_dir, 
                                           format_spec_DB=format_spec_DB,
                                           sequence_DB=sequence_DB,
                                           type_list=type_list,
                                           id_counter=len(sequence_DB.get()["ids"]))
        structure_planning_result = await sequence_planner.plan_sequence(self, max_tries=3)
        printer.print('----------------------- Structure Planning Completed -----------------------')
        printer.print(f'* Sequence Planner proposed the structure as:\n{structure_planning_result}')

        # FIELD_DESIGNER
        field_designer = FIELD_DESIGNER(target, 
                                        seed_dir=seed_dir, 
                                        format_spec_DB=format_spec_DB,
                                        component_DB=component_DB,
                                        type_list=type_list)
        field_designing_result = await field_designer.design_field(self, max_tries=5)
        printer.print('----------------------- Field Design Completed -----------------------')
        printer.print(f'* Field Designer proposed the field designs as:\n{field_designing_result}')

        # DEVELOPER
        ## TESTING
        # type_list = ['Query', 'Update', 'Notify', 'Zone Transfer (AXFR)', 'Incremental Zone Transfer (IXFR)']
        # type_list = ['SETUP', 'DESCRIBE', 'SET_PARAMETER', 'GET_PARAMETER', 'ANNOUNCE', 'RECORD', 'TEARDOWN', 'PLAY', 'PAUSE', 'REDIRECT']
        # seed_sequence_pairs = {
        #     # "dns_queries.raw": "['Query', 'Query', 'Query', 'Query', 'Query', 'Query', 'Query', 'Query']"
        # }
        # printer.print(f'* Loaded format_spec_DB and sequence_DB from previous run.')
        # await self.load_memory(format_spec_DB, "format_spec_DB", "agent_runs/2025-09-18_18-45-50-DNS")
        # await self.load_memory(sequence_DB, "sequence_DB", "agent_runs/2025-09-18_18-45-50-DNS")
        # await self.load_memory(component_DB, "component_DB", "agent_runs/2025-09-18_18-45-50-DNS")
        # printer.print(f'* Current format_spec_DB has {len(format_spec_DB.get()["ids"])} entries.')
        # printer.print(f'* Current sequence_DB has {len(sequence_DB.get()["ids"])} entries.')
        # printer.print(f'* Current component_DB has {len(component_DB.get()["ids"])} entries.')
        ## TEST SETUP END

        developer = DEVELOPER(target, 
                              seed_dir=seed_dir, 
                              format_spec_DB=format_spec_DB,
                              sequence_DB=sequence_DB,
                              component_DB=component_DB,
                              seed_sequence_pairs=seed_sequence_pairs,
                              type_list=type_list)
        new_seed_result = await developer.develop_new_seed(self, sequence_id=len(sequence_DB.get()["ids"])-1, max_tries=3)
        printer.print('----------------------- New Seed Development Completed -----------------------')
        printer.print(f'* Developer generated the new seed as:\n{new_seed_result}')

        # TESTER
        

    async def load_memory(self, db, db_name, path_to_db):
        if db_name not in ["component_DB", "format_spec_DB", "sequence_DB"]:
            return f"[ERROR] Unsupported DB_name: {db_name}. Supported databases are: component_DB, format_spec_DB, sequence_DB."

        # Load the latest JSON file from the specified DB
        db_dir = os.path.join(path_to_db, db_name)
        json_files = [f for f in os.listdir(db_dir) if f.endswith('.json') and f[:-5].isdigit()]
        if not json_files:
            return f"[ERROR] No Data in {db_name}."    
        max_idx = max([int(f[:-5]) for f in json_files])
        max_json = f"{max_idx}.json"
        db_path = os.path.join(db_dir, max_json)

        # Add documents
        with open(db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            db.add(ids=data['ids'], documents=data['documents'])

    async def test_llm(self):
        tester = TESTER("test")
        result = await tester.test(self)
        print(result)
