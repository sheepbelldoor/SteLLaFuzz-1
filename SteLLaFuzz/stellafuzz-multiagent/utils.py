from datetime import datetime
import json
import os
import re

from openai.types.chat import (
    ChatCompletionMessageToolCallParam,
)

class DualPrinter:
    def __init__(self, file_path="output.log"):
        self.file = open(file_path, "a", encoding="utf-8")

    def print(self, *args, **kwargs):
        print(*args, **kwargs)

        kwargs_no_flush = {k: v for k, v in kwargs.items() if k != "flush"}
        print(*args, file=self.file, **kwargs_no_flush)

        self.file.flush()

    def close(self):
        self.file.close()


RESULT_PATH = 'agent_runs/{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(RESULT_PATH, exist_ok=True)
os.makedirs(os.path.join(RESULT_PATH, 'format_spec_DB'), exist_ok=True)
os.makedirs(os.path.join(RESULT_PATH, 'sequence_DB'), exist_ok=True)
os.makedirs(os.path.join(RESULT_PATH, 'component_DB'), exist_ok=True)
os.makedirs(os.path.join(RESULT_PATH, 'coverage_DB'), exist_ok=True)
os.makedirs(os.path.join(RESULT_PATH, 'seed_DB'), exist_ok=True)
printer = DualPrinter(file_path=os.path.join(RESULT_PATH, "output.log"))

def stringify_tool_call_results(tool_call_result: dict) -> str:
    if 'content' not in tool_call_result:
        return ""

    try:
        content_dict = json.loads(tool_call_result['content'])
    except json.JSONDecodeError:
        return tool_call_result['content']

    concat = ''
    for tool_name, result in content_dict.items():
        if isinstance(result, list) and len(result) == 1:
            result = result[0]
        concat += f"""\n* `{tool_name}`:
```
{result}
```"""
    return concat

def stringify_tool_call_requests(tool_call: ChatCompletionMessageToolCallParam):
    return f"[Assistant requested to call tool {tool_call['function']['name']} with arguments {tool_call['function']['arguments']}]"

def format_assistant_responses(messages, last_user_messages_index=-1):
    assistant_messages = messages[last_user_messages_index + 1:]
    formatted = []
    for msg in assistant_messages:
        if msg['role'] == "tool":
            formatted.append(stringify_tool_call_results(msg))
        elif msg['role'] == "assistant":
            if 'tool_calls' in msg and len(msg['tool_calls']) > 0:
                for tool_call in msg['tool_calls']:
                    formatted.append(stringify_tool_call_requests(tool_call))

            if 'content' in msg and len(msg['content']) > 0:
                formatted.append(msg['content'])
    return '\n'.join(formatted)

def message_to_json(message: str) -> dict:
    '''
    메시지에 ```json ... ``` 형식으로 JSON 데이터가 포함되어 있을 때, 해당 JSON 데이터를 파싱하여 딕셔너리로 반환합니다.
    '''
    pattern = r"```json\s*([\s\S]*?)```"
    match = re.search(pattern, message)
    if not match:
        return {}
    json_str = match.group(1).strip()
    try:
        return json.loads(json_str)
    except Exception:
        return {}
    