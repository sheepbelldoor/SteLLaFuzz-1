import os
import json
import random
from typing import List
from pprint import pprint
import re

MODEL = "gpt-4o-mini"
LLM_RESULT_DIR = "llm_outputs"
TEST_MESSAGE_DIR = os.path.join(LLM_RESULT_DIR, "messages")
SEQUENCE_REPEAT = 1
LLM_RETRY = 3

def convert_message_to_binary(message: str) -> bytes:
    if not message:
        return b''
    
    parts = message.split(' ')
    processed_parts = []
    
    for part in parts:
        if part.startswith('0x'):
            try:
                binary_value = bytes([int(part[2:], 16)])
                processed_parts.append((binary_value, True))
            except ValueError:
                processed_parts.append((part.encode(), False))
        else:
            processed_parts.append((part.encode(), False))
    
    result = bytearray()
    for i in range(len(processed_parts)):
        current_data, current_is_binary = processed_parts[i]
        result.extend(current_data)
        
        if i < len(processed_parts) - 1:
            next_is_binary = processed_parts[i+1][1]
            if not current_is_binary and not next_is_binary:
                result.extend(b' ')

    return bytes(result)

def save_test_cases(test_cases: dict, output_dir: str, seed_file_name: str) -> None:
    concatnated_messages = bytearray()
    os.makedirs(output_dir, exist_ok=True)
    
    idx = 1
    for testcase in test_cases.values():
        for sequence in testcase["sequences"]:
            try:
                for message in sequence["messages"]:   
                    concatnated_messages += convert_message_to_binary(message["message"]) + b"\r\n"

                while True:
                    file_path = os.path.join(output_dir, f"new_{idx}.raw")
                    if not os.path.exists(file_path):
                        break
                    idx += 1
                
                with open(file_path, "wb") as f:
                    f.write(concatnated_messages)
                concatnated_messages = bytearray()
                idx += 1
            except Exception as e:
                print(f"Error: {e}")
            
def load_seed_messages(seed_messages_dir: str) -> List[str]:
    seed_messages = []
    file_names = []
    for file in os.listdir(seed_messages_dir):
        file_path = os.path.join(seed_messages_dir, file)
        file_names.append(file)
        with open(file_path, "rb") as f:
            binary_content = f.read()
        
        readable_content = ""
        for byte in binary_content:
            if byte in (9, 10, 13) or (32 <= byte <= 126):
                readable_content += chr(byte)
            else:
               readable_content += f" 0x{byte:02x} "
        
        seed_messages.append(readable_content)
    return file_names, seed_messages

