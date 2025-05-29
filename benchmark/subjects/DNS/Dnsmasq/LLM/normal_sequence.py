import os
import json

from typing import Optional, List
from pydantic import BaseModel
from openai import OpenAI
from utility.utility import MODEL, LLM_RETRY, LLM_RESULT_DIR

MESSAGE_SEQUENCE_OUTPUT_DIR = "message_sequence_results"

class Sequence(BaseModel):
    sequenceId: str
    type_sequence: List[str]

class ProtocolSequences(BaseModel):
    protocol: str
    sequences: List[Sequence]
    explanation: str

MESSAGE_PROMPT = """\
You are a network protocol expert with deep understanding of [PROTOCOL].
Your task is to generate a series of message sequences for client-to-server communications in the [PROTOCOL] protocol.
The objective is to generate **valid sequences** that maximize code coverage by exercising as many lines, states, and branches in the protocol implementation as possible.

You are provided with a complete list of client-to-server message types:
[TYPES]

Please adhere to the following instructions:

1. **Generate Valid Message Sequences:**
   - Create multiple message sequences that collectively include client-to-server message types from the provided list.
   - You MUST generate sequences of length [SEQ_LENGTH]. (for example, if sequence length is 1, you should not generate sequence SERVICE_REQUEST because there is no KEXINIT message before it.)
   - Each sequence should vary the order of messages and include conditional transitions to trigger different execution paths.
   - The goal is full exploration of states and alternative branches in the protocol's state machine to maximize coverage.
   - You *may* repeat message types within or across sequences if it helps to uncover additional states or branches, but it is *not required* if doing so does not add coverage value.

2. **Include Detailed Message Information (Optional):**
   - If needed, you may include a "details" or similar field for each message to specify parameters or edge conditions that could lead to different states or error scenarios.
   - However, ensure that any message type you specify strictly matches one of the provided client-to-server types.

3. **Provide a Coverage Rationale:**
   - In the "explanation" field, describe the step-by-step reasoning for constructing the sequences, focusing on how the chosen order, conditional paths, and potential repetition were used to achieve broad coverage.

4. **Final Output Requirements:**
   - Do not include any extraneous text; only provide the final JSON output.
   - The output must be valid JSON, strictly adhering to the structure below.

5. **Final Output Structure:**
   ```json
   {
     "protocol": "[PROTOCOL]",
     "sequences": [
       {
         "sequenceId": "A unique identifier for the sequence",
         "type_sequence": [
           "Type of message 1",
           "Type of message 2",
           "Type of message 3"
           // ...
         ]
       }
       // ... additional sequence objects
     ],
     "explanation": "A brief explanation of how these sequences were constructed to maximize coverage, including the rationale behind the order and selection of messages."
   }
   ```

Please generate the final message call sequences strictly following the above instructions.
"""

def using_llm(prompt: str) -> ProtocolSequences:
    client = OpenAI()
    try:
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a network protocol expert with deep understanding of [PROTOCOL]."},
                {"role": "user", "content": prompt}
            ],
            response_format=ProtocolSequences,
            timeout=90
        )
        response = completion.choices[0].message.parsed

        index = 0
        os.makedirs(os.path.join(LLM_RESULT_DIR, "3_message_sequences"), exist_ok=True)
        while os.path.exists(os.path.join(LLM_RESULT_DIR, "3_message_sequences", f"response_{index}.json")):
            index += 1
        protocol_file = os.path.join(LLM_RESULT_DIR, "3_message_sequences", f"response_{index}.json")
        with open(protocol_file, "w", encoding="utf-8") as f:
            json.dump(completion.model_dump(), f, indent=4, ensure_ascii=False)
        return response
    except Exception as e:
        print(f"Error processing protocol: {e}")
        return None

def get_message_sequences(protocol: str, message_types: dict, seq_length: int) -> dict:
    types_list = [type["name"] for type in message_types["client_to_server_messages"]]
    types = ""
    for type in types_list:
        types += f"- {type}\n"
    types = types.strip()

    prompt = MESSAGE_PROMPT.replace("[PROTOCOL]", protocol)\
                           .replace("[TYPES]", types)\
                           .replace("[SEQ_LENGTH]", str(seq_length))

    for _ in range(LLM_RETRY):
        response = using_llm(prompt)
        if response is not None:
            break

    if response is None:
        raise Exception(f"Failed to generate message sequence for {protocol}")


    # Filter out sequences that has length not equal to [SEQ_LENGTH]
    response.sequences = [seq for seq in response.sequences if len(seq.type_sequence) == seq_length]

    # Save the results to a JSON file
    os.makedirs(MESSAGE_SEQUENCE_OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(MESSAGE_SEQUENCE_OUTPUT_DIR, f"{protocol.lower()}_message_sequences_{seq_length}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(response.model_dump(), f, indent=4, ensure_ascii=False)    
    print(f"Saved results for {protocol} to {file_path}")

    # Save the prompt and response to a text file
    os.makedirs(LLM_RESULT_DIR, exist_ok=True)
    protocol_file = os.path.join(LLM_RESULT_DIR, f"3_{protocol.lower()}_message_sequences_{seq_length}.json")
    with open(protocol_file, "w", encoding="utf-8") as f:
        json.dump(response.model_dump(), f, indent=4, ensure_ascii=False)

    return response.model_dump()
