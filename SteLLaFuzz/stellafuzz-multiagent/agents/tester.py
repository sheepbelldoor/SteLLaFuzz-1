from openai.types.chat import (
    ChatCompletionUserMessageParam,
)

import json
import os
import traceback
import chromadb
from utils import message_to_json, RESULT_PATH, printer, format_assistant_responses

class TESTER:
    def __init__(self, target: str):
        self.target = target


#     ## Analyze from Inputs
#     async def test(self, mcp_client):
#         messages = [{
#             "role": "system",
#             "content": f"You are a helpful assistant."
#         }]
#         first_user_message = \
# f'''\
# I want to know about format and specification of RTSP OPTIONS message.
# \
# '''

#         messages.append(ChatCompletionUserMessageParam(role="user", content=first_user_message))

#         messages = await mcp_client.process_messages_streaming(messages)

#         response = messages[-1]['content']
#         print(response)
#         return "Success"
    
    # async def run()