import os
import json

from typing import Optional, List
from pydantic import BaseModel
from openai import OpenAI
from utility.utility import MODEL, LLM_RETRY, LLM_RESULT_DIR

PROTOCOL_SPECIALIZED_STRUCTURE_OUTPUT_DIR = "protocol_specialized_structure_results"

class StructuredField(BaseModel):
    name: str                                   # Field name (e.g., "field_name")
    fixed_byte_length: Optional[int] = None     # Fixed byte length
    data_type: str                              # Data type (e.g., "string", "int", "bytes", "boolean", etc.)
    description: str                            # Brief description of the field
    details: Optional[str] = None               # Additional information (length, encoding, constraints, etc.)

class StructuredOutput(BaseModel):
    protocol: str                       # Protocol name ([PROTOCOL])
    message_type: str                   # Message type ([TYPE])
    code: Optional[str] = None          # Message type code ([CODE])
    type_description: str               # Message type description ([DESCRIPTION])
    fields: List[StructuredField]       # List of fields that make up the message structure
    reasoning: str                      # Step-by-step explanation of the structure derivation and official sources used

PROTOCOL_SPECIALIZED_STRUCTURE_PROMPT = """\
You are a network protocol expert with deep understanding of [PROTOCOL].
Your task is to extract the detailed message structure for the client-to-server message type [TYPE] in the [PROTOCOL] protocol.

This message structure must include:
- A comprehensive list of all fields (including any common headers, body elements, and subfields) as defined in the official documentation, RFCs, or other recognized authoritative sources.
- For each field, include:
  - "name": The field name as specified in the documentation.
  - "fixed_byte_length": The fixed byte length of the field. if the field is variable, set this to null.
  - "data_type": The type of data (e.g., string, int, bytes, boolean, etc.).
  - "description": A brief description of the field and its purpose.
  - "details": Any additional information such as length, encoding, or constraints, if applicable.

In addition to the above, also include details about the message type itself:
- "code": The code of the message type (this value may be NULL). Represent this with [CODE].
- "typeDescription": A description of the message type. Represent this with [DESCRIPTION]. If it can be specified in the documentation, use it.

Please adhere to the following instructions:

1. **Extract the Message Structure:**
   - Identify every field of the [TYPE] message as specified in the [PROTOCOL] documentation.
   - Include any common header or shared fields if applicable, but focus primarily on the fields unique to [TYPE].

2. **Define the Output Format:**
   - Create the JSON output strictly following the given structure.
   - Ensure the JSON object includes "protocol", "message_type", "code", "type_description", "fields" (an array of field objects), and "reasoning".

3. **Specify Field Details:**
   - For each field in the "fields" array, provide:
     - "name": the exact field name.
     - "fixed_byte_length": The fixed byte length of the field. if the field is variable, set this to null.
     - "data_type": the field's data type (e.g., string, int, bytes, boolean, etc.).
     - "description": a brief explanation of the field.
     - "details": any additional details (such as length, encoding, constraints).

4. **Provide Structured Reasoning:**
   - In the "reasoning" field, include a clear, step-by-step explanation of how you derived the structure.
   - Mention the official sources (documentation, RFCs, etc.) that were referenced.
   - Note any assumptions or ambiguities encountered and how you resolved them.

5. **Final Output Structure:**
   - The final output must be a JSON object with the following structure:
   ```json
   {
    "protocol": "[PROTOCOL]",
    "message_type": "[TYPE]",
    "code": "[CODE]",
    "type_description": "[DESCRIPTION]",
    "fields": [
        {
        "name": "field_name",
        "fixed_byte_length": "fixed_byte_length",
        "data_type": "data_type",
        "description": "description",
        "details": "additional details if any"
        }
        // ... additional field objects
    ],
    "reasoning": "A step-by-step explanation of how the structure was derived, including the official sources (documentation, RFCs, etc.) used and any assumptions made."
    }
   ```

Please produce the final JSON output accordingly, strictly following the above instructions.
"""

def using_llm(prompt: str) -> StructuredOutput:
    client = OpenAI()
    try:
        completion = client.beta.chat.completions.parse(
            model=MODEL,
            temperature=0.1,
            messages=[
                {"role": "system", "content": "You are a network protocol expert with deep understanding of [PROTOCOL]."},
                {"role": "user", "content": prompt}
            ],
            response_format=StructuredOutput,
            timeout=90
        )
        response = completion.choices[0].message.parsed

        index = 0
        os.makedirs(os.path.join(LLM_RESULT_DIR, "2_specialized_structures"), exist_ok=True)
        while os.path.exists(os.path.join(LLM_RESULT_DIR, "2_specialized_structures", f"response_{index}.json")):
            index += 1
        protocol_file = os.path.join(LLM_RESULT_DIR, "2_specialized_structures", f"response_{index}.json")
        with open(protocol_file, "w", encoding="utf-8") as f:
            json.dump(completion.model_dump(), f, indent=4, ensure_ascii=False)
        return response
    except Exception as e:
        print(f"Error processing protocol: {e}")
        return None

def get_specialized_structure(protocol: str, message_type: dict) -> None:
    prompt = PROTOCOL_SPECIALIZED_STRUCTURE_PROMPT.replace("[PROTOCOL]", protocol)\
                                                  .replace("[TYPE]", message_type["name"])\
                                                  .replace("[CODE]", message_type["code"] if message_type["code"] else "NULL")\
                                                  .replace("[DESCRIPTION]", message_type["description"])
    
    for _ in range(LLM_RETRY):
        response = using_llm(prompt)
        if response is not None:
            break

    if response is None:
        raise Exception(f"Failed to generate specialized structure for {message_type['name']} in {protocol}")

    return response.model_dump()

def get_specialized_structures(protocol: str, message_types: dict) -> None:
    structures = {}

    for message_type in message_types["client_to_server_messages"]:
        try:
            structures[message_type["name"]] = get_specialized_structure(protocol, message_type)
        except Exception as e:
            print(f"Error processing message type {message_type['name']} in {protocol}: {e}")
    
    os.makedirs(PROTOCOL_SPECIALIZED_STRUCTURE_OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(PROTOCOL_SPECIALIZED_STRUCTURE_OUTPUT_DIR, f"{protocol.lower()}_specialized_structures.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(structures, f, indent=4, ensure_ascii=False)    
    print(f"Saved results for {protocol} to {file_path}")

    os.makedirs(LLM_RESULT_DIR, exist_ok=True)
    protocol_file = os.path.join(LLM_RESULT_DIR, f"2_{protocol.lower()}_specialized_structures.json")
    with open(protocol_file, "w", encoding="utf-8") as f:
        json.dump(structures, f, indent=4, ensure_ascii=False)

    return structures
