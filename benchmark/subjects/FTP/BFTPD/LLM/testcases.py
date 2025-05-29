import os
import json

from typing import Optional, List
from pydantic import BaseModel
from openai import OpenAI
from utility.utility import MODEL, LLM_RETRY, LLM_RESULT_DIR, SEQUENCE_REPEAT

TESTCASE_OUTPUT_DIR = "testcase_results"

class Message(BaseModel):
    message: str

class Sequence(BaseModel):
    sequenceId: str
    messages: List[Message]
    explanation: str

class TestCase(BaseModel):
    protocol: str
    sequences: List[Sequence]

MESSAGE_PROMPT = """\
You are a network protocol expert with deep understanding of [PROTOCOL].
Your task is to generate client-to-server message sequences for the [PROTOCOL] protocol based on the following inputs:

1. **Seed Message:**
   ```
   [SEED_MESSAGE]
   ```
   (This is the existing, valid message loaded using the provided load_seed_messages function. The function reads seed messages from files and converts non-ASCII characters to their hex representation. The valid parameters present in these seed messages must be preserved in all generated sequences.)

2. **Type Sequence:**  
   [SEQUENCE]

3. **Type Structure:**  
   [STRUCTURE]

4. **Number of Message Sequences to Generate:**  
   [NUMBER]

Please adhere to the following instructions:

1. **Generate Messages for the Sequence:**
   - MUST use the valid parameters from seed message as a baseline and preserve these values throughout the generated messages.
   - Generate messages according to the order specified in the type sequence.
   - Create [NUMBER] message sequences following the order specified in the type sequence.
   - To increase diversity and maximize coverage, vary the message type sequence (e.g., by rearranging the order, repeating specific message types, or introducing edge-case scenarios) while keeping the valid parameters from seed message intact.
   - If additional messages are needed, generate them according to the protocol specification using the preserved valid parameters.
   - For binary-based protocols, represent each message as a sequence of bytes in hex format separated by spaces (e.g., "0x1a 0x0b 0x34 0x00").
   - For text-based protocols, generate the message in plain ASCII text using spaces, newlines, or CRLF as needed according to the protocol specification.
   - For each message in a sequence, map the message type to its corresponding structure from the type structure and generate realistic, concrete values for each defined field using the valid parameters from seed message.
   - For each message, if is_binary is true, all messages MUST be written in a hex format separated by spaces.

   **Example:**  
   For SMTP, an acceptable output would be:
   ```json
   {
      "protocol": "SMTP",
      "sequences": [
          {
              "sequenceId": "1",
              "messages": [
                  {"message": "HELO localhost"},
                  {"message": "MAIL FROM:<ubuntu@ubuntu>"},
                  {"message": "RCPT TO:<ubuntu@ubuntu>"},
                  {"message": "DATA"},
                  {"message": "From: ubuntu <ubuntu@ubuntu>\\r\\nTo: ubuntu <ubuntu@ubuntu>\\r\\nSubject: Test Email\\r\\n\\r\\nThis is a test email body."},
                  {"message": "QUIT"}
              ],
              "explanation": "Explanation of the sequence generation process"
          }
      ]
   }
   ```
   For SSH, an acceptable output would be:
   ```json
   {
      "protocol": "SSH",
      "sequences": [
          {
              "sequenceId": "1",
              "messages": [
                  {"message": "SSH-2.0-OpenSSH_7.5"},
                  {"message": "0x00 0x00 0x9c 0x05 0x14 0x09 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x30 0x01 0x75 0x63 0x76 0x72 0x32 0x65 0x35 0x35 0x39 0x31 0x73 0x2d 0x61 0x68 0x35 0x32 0x2c 0x36 0x6c 0x7a 0x62 0x69 0x00 0x00 0x1a 0x00 0x6f 0x6e 0x65 0x6e 0x7a 0x2c 0x69 0x6c 0x40 0x62 0x70 0x6f 0x6e 0x65 0x73 0x73 0x2e 0x68 0x6f 0x63 0x2c 0x6d 0x6c 0x7a 0x62 0x69 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x2c 0x00 0x1e 0x06 0x00 0x00 0x20 0x00 0xe5 0x2f 0xa3 0x7d 0xcd 0x47 0x43 0x62 0x28 0x15 0xac 0xda 0xbb 0x5f 0x07 0x29 0xff 0x30 0x84 0xf6 0xc4 0xaf 0xc2 0xcf 0x90 0xed 0x5f 0x99 0xcb 0x58 0x74 0x3b 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x0c 0x00 0x00 0x0a"},
                  {"message": "0x00 0x15 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x00 0x06 0x18 0x00 0x05 0x00 0x00 0x73 0x0c 0x68 0x73 0x75 0x2d 0x65 0x73 0x61 0x72 0x74 0x75 0x00 0x68 0x00 0x00 0x00 0x00 0xb9 0x00 0x1a 0xac 0x9c 0xe0 0xc1 0xfa 0x00 0xd5 0x00 0x00 0x0a 0x30"},
                  {"message": "0x00 0x32 0x00 0x00 0x75 0x06 0x75 0x62 0x74 0x6e 0x00 0x75 0x00 0x00 0x73 0x0e 0x68 0x73 0x63 0x2d 0x6e 0x6f 0x65 0x6e 0x74 0x63 0x6f 0x69 0x00 0x6e 0x00 0x00 0x6e 0x04 0x6e 0x6f 0x00 0x65 0x00 0x00 0x00 0x00 0x00 0x00 0xf3 0x00 0x35 0xee 0xe3 0xb0 0x27 0x3a 0x00 0x5d 0x00 0x00 0x0a 0x48"}
              ],
              "explanation": "Explanation of the sequence generation process"
          }
      ]
   }
   ```

2. **Ensure Maximum Coverage:**
   - Design sequences to maximize coverage by including variations (e.g., repeated message types, edge-case values, error-triggering values) that exercise different protocol states and transitions.
   - Include variations that account for both normal and exceptional conditions in the protocol.

3. **Authoritative and Accurate:**
   - Base the actual values strictly on the provided type structure.
   - Use official documentation and RFC details from type structure to ensure correctness.
   - Avoid subjective assumptions; rely solely on the provided inputs.

4. **Step-by-Step Reasoning:**
   - In the "explanation" field, include a clear, step-by-step explanation of how the sequences were generated.
   - Describe the process of mapping each message type in sequence to its corresponding structure in type structure and how actual values were determined.
   - Note any differences in handling text-based versus binary-based protocols.
   - Explain how the valid parameters from seed message were preserved and utilized.
   - Note any differences in handling text-based versus binary-based protocols.

5. **Final Output Format:**
   - The final output must be a JSON object with the following structure:
     ```json
     {
       "protocol": "[PROTOCOL]",
       "sequences": [
         {
           "sequenceId": "A unique identifier for the sequence",
           "message_sequence": "Total messages in the sequence",
           "explanation": "A step-by-step explanation of how the sequences were generated and the rationale behind the actual values selected.",
           "is_binary": "True if the protocol is binary-based, False otherwise"
         }
         // ... additional sequence objects, up to [NUMBER] sequences
       ]
     }
     ```

Please generate multiple valid messages for [PROTOCOL] based on the above instructions.
"""


def using_llm(prompt: str) -> TestCase:
    client = OpenAI()
    try:
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a network protocol expert with deep understanding of [PROTOCOL]."},
                {"role": "user", "content": prompt}
            ],
            response_format=TestCase,
            timeout=30
        )
        response = completion.choices[0].message.parsed

        index = 0
        os.makedirs(os.path.join(LLM_RESULT_DIR, "6_testcases"), exist_ok=True)
        while os.path.exists(os.path.join(LLM_RESULT_DIR, "6_testcases", f"response_{index}.json")):
            index += 1
        protocol_file = os.path.join(LLM_RESULT_DIR, "6_testcases", f"response_{index}.json")
        with open(protocol_file, "w", encoding="utf-8") as f:
            json.dump(completion.model_dump(), f, indent=4, ensure_ascii=False)
        return response
    except Exception as e:
        print(f"Error processing protocol: {e}")
        return None

def get_test_case(protocol: str, type_sequence: List[str], specialized_structure: dict, seed_message: str) -> None:
    sequence = ""
    structure = ""
    for i, type in enumerate(type_sequence):
        sequence += f"{i+1}. {type}\n"
        structure += f"""\
{type}\n\
- Code: {specialized_structure[type]['code']}\n\
- Description: {specialized_structure[type]['type_description']}\n\
- Fields: {specialized_structure[type]['fields']}\n\n
"""
    sequence = sequence.strip()
    structure = structure.strip()

    if seed_message:
        seed_message = f"{seed_message}"
    else:
        seed_message = ""
    
    prompt = MESSAGE_PROMPT.replace("[PROTOCOL]", protocol)\
                           .replace("[SEQUENCE]", sequence)\
                           .replace("[STRUCTURE]", structure)\
                           .replace("[NUMBER]", str(SEQUENCE_REPEAT))\
                           .replace("[SEED_MESSAGE]", seed_message)

    
    for _ in range(LLM_RETRY):
        response = using_llm(prompt)
        if response is not None:
            break

    if response is None:
        raise Exception(f"Failed to generate message for {specialized_structure['message_type']} in {protocol}")

    return response.model_dump()

def get_test_cases(protocol: str, message_sequences: dict, specialized_structures: dict, seed_message: str) -> None:
    test_cases = {}
    for sequence in message_sequences["sequences"]:
        try:
            print(f"Processing message sequence: {sequence['sequenceId']}")
            test_cases[sequence["sequenceId"]] = get_test_case(protocol, sequence["type_sequence"], specialized_structures, seed_message)
        except Exception as e:
            print(f"Error processing message sequence {sequence['sequenceId']} in {protocol}: {e}")
    
    os.makedirs(TESTCASE_OUTPUT_DIR, exist_ok=True)
    idx = 1
    file_path = os.path.join(TESTCASE_OUTPUT_DIR, f"{protocol.lower()}_testcases_{idx}.json")
    while os.path.exists(file_path):
        idx += 1
        file_path = os.path.join(TESTCASE_OUTPUT_DIR, f"{protocol.lower()}_testcases_{idx}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=4, ensure_ascii=False)
    print(f"Saved results for {protocol} to {file_path}")

    os.makedirs(LLM_RESULT_DIR, exist_ok=True)
    idx = 1
    file_path = os.path.join(LLM_RESULT_DIR, f"4_{protocol.lower()}_testcases_{idx}.json")
    while os.path.exists(file_path):
        idx += 1
        file_path = os.path.join(LLM_RESULT_DIR, f"4_{protocol.lower()}_testcases_{idx}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=4, ensure_ascii=False)
    print(f"Saved results for {protocol} to {file_path}")

    return test_cases
