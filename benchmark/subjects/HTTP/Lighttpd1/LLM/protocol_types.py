import os
import json

from typing import Optional, List
from pydantic import BaseModel
from openai import OpenAI
from utility.utility import MODEL, LLM_RETRY, LLM_RESULT_DIR

PROTOCOL_TYPE_OUTPUT_DIR = "protocol_type_results"

class MessageType(BaseModel):
    name: str                               # Message type name (e.g., DISCONNECT, KEXINIT)
    code: Optional[str] = None              # Code of the message type
    description: str                        # Brief description of the message 

class ProtocolMessageTypes(BaseModel):
    protocol: str                                               # Protocol name (e.g., SSH, HTTP)
    client_to_server_messages: List[MessageType]                # List of all message types in the protocol
    potential_candidates: Optional[List[MessageType]] = None    # List of potential candidates for message types
    references: Optional[List[str]] = None                      # List of official documents or RFCs
    notes: Optional[str] = None                                 # Considerations for future extensibility or additional notes

PROTOCOL_TYPE_PROMPT = """\
You are a network protocol expert with deep understanding of [PROTOCOL].
Your task is to extract all defined client-to-server message types in the [PROTOCOL] protocol, including any extended or optional commands as defined in official documentation or recognized RFC extensions.

Please adhere to the following instructions:

1. **Identify All Client-to-Server Message Types (Including Extensions):**
   - List every client-to-server message type defined in the [PROTOCOL] protocol exactly as specified in the official documentation, RFCs, or other recognized authoritative sources.
   - Ensure that extended or optional commands (such as BDAT in the case of Exim's SMTP extensions) are also included if they are part of the protocol's official extensions.
   - If the protocol documentation provides message codes or numeric values alongside the message types, include them. This is not server response code.
   - Present your answer in a structured format (e.g., a JSON array or a table) to ensure clarity and completeness.
   - If applicable, sort the list in alphabetical order or according to the order specified in the official documentation.
   - Message types sent from the server to the client are not extracted.
   - **Example:**  
     For SSH, an acceptable output would be:  
     ```json
     {
       "protocol": "SSH",
       "client_to_server_messages": [
         {"name": "KEXINIT", "code": "20", "description": "Description of KEXINIT including its purpose and usage"},
         {"name": "SERVICE_REQUEST", "code": "5", "description": "Description of SERVICE_REQUEST including its purpose and usage"},
         {"name": "USERAUTH_REQUEST", "code": "50", "description": "Description of USERAUTH_REQUEST including its purpose and usage"}
         // ... include other message types as defined in the official documentation.
       ]
     }
     ```

2. **Authoritative and Accurate:**
   - Base your response strictly on official documentation, RFCs, or other recognized authoritative sources.
   - Provide references (e.g., document names, URLs) for the sources you consulted.
   - Avoid any subjective interpretation or hallucinated information.
   - **Example:**  
     In your response, include a section like:  
     ```plaintext
     Sources:
     - RFC 4253 (SSH Transport Layer Protocol): https://tools.ietf.org/html/rfc4253
     - Official [PROTOCOL] documentation: [Insert URL here]
     ```

3. **Step-by-Step Reasoning:**
   - Detail the process you used to derive the list of client-to-server message types.
   - Explain which official documents or RFCs you consulted and how you verified that the list is complete.
   - If there are ambiguous or unclear parts in the documentation, describe how you addressed them and note any potential uncertainties.
   - **Example:**  
     Include a reasoning section such as:  
     ```plaintext
     Reasoning Process:
     - Step 1: Reviewed the official [PROTOCOL] documentation to identify the message types.
     - Step 2: Cross-referenced with RFC [Number] to ensure all client-to-server message types were included.
     - Step 3: Noted that certain message types had ambiguous definitions; these are marked in the "Potential Candidates" section.
     ```

4. **Error Handling and Completeness:**
   - If certain message types (including any extensions like BDAT) are not clearly defined in the official sources, include a note on these uncertainties and list any potential candidates in a separate section (e.g., "Potential Candidates").
   - Cross-check multiple official sources to confirm the completeness of the list.
   - **Example:**  
     Add a section for ambiguous or uncertain message types, for example:  
     ```plaintext
     Potential Candidates:
     - [TYPE_X]: Defined in some unofficial documentation but not clearly specified in the official sources.
     - [TYPE_Y]: Might be part of an extended version of the protocol.
     ```

Please extract all client-to-server message types for [PROTOCOL] following the above instructions.
"""

def using_llm(prompt: str) -> ProtocolMessageTypes:
    client = OpenAI()
    try:
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": "You are a network protocol expert with deep understanding of [PROTOCOL]."},
                {"role": "user", "content": prompt}
            ],
            response_format=ProtocolMessageTypes,
            timeout=90
        )
        response = completion.choices[0].message.parsed

        index = 0
        os.makedirs(os.path.join(LLM_RESULT_DIR, "1_types"), exist_ok=True)
        while os.path.exists(os.path.join(LLM_RESULT_DIR, "1_types", f"response_{index}.json")):
            index += 1
        protocol_file = os.path.join(LLM_RESULT_DIR, "1_types", f"response_{index}.json")
        with open(protocol_file, "w", encoding="utf-8") as f:
            json.dump(completion.model_dump(), f, indent=4, ensure_ascii=False)
        return response
    except Exception as e:
        print(f"Error processing protocol: {e}")
        return None

def get_protocol_message_types(protocol: str) -> dict:
    prompt = PROTOCOL_TYPE_PROMPT.replace("[PROTOCOL]", protocol)

    for _ in range(LLM_RETRY):
        response = using_llm(prompt)
        if response is not None:
            break

    if response is None:
        raise Exception(f"Failed to generate message types for {protocol}")

    os.makedirs(PROTOCOL_TYPE_OUTPUT_DIR, exist_ok=True)    
    protocol_file = os.path.join(PROTOCOL_TYPE_OUTPUT_DIR, f"{protocol.lower()}_types.json")
    with open(protocol_file, "w", encoding="utf-8") as f:
        json.dump(response.model_dump(), f, indent=4, ensure_ascii=False)
    print(f"Saved results for {protocol} to {protocol_file}")

    os.makedirs(LLM_RESULT_DIR, exist_ok=True)    
    protocol_file = os.path.join(LLM_RESULT_DIR, f"1_{protocol.lower()}_types.json")
    with open(protocol_file, "w", encoding="utf-8") as f:
        json.dump(response.model_dump(), f, indent=4, ensure_ascii=False)

    return response.model_dump()
