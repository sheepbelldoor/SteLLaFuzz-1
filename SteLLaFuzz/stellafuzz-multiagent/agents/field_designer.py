from openai.types.chat import (
    ChatCompletionUserMessageParam,
)

import json
import os
import traceback
import chromadb
from utils import message_to_json, RESULT_PATH, printer, format_assistant_responses

class FIELD_DESIGNER:
    def __init__(self, target: str, seed_dir: str, format_spec_DB: chromadb.api.Collection, component_DB: chromadb.api.Collection, type_list: list):
        self.target = target
        self.seed_dir = seed_dir
        self.format_spec_DB = format_spec_DB
        self.component_DB = component_DB
        self.type_list = type_list
        self.id_counter = 0

    def add_memory_entries(self, entries):
        ids = [str(self.id_counter + i) for i in range(len(entries))]
        self.id_counter += len(entries)
        self.component_DB.add(
            ids=ids,
            documents=entries
        )

    def dump_memory(self):
        return json.dumps(self.component_DB.get())

    ## Field Design
    async def design_field(self, mcp_client, max_tries=5):
        for type in self.type_list:
            for t in range(max_tries):
                messages = [{
                    "role": "system",
                    "content": f"You are the Field Designer. Your task is to generate new fields and confirm the constraints and features of a given message type."
                }]
                first_user_message = f'''\
You are a domain expert with deep understanding of {self.target}.

Your task is to analyze the format_spec_DB and confirm the constraints and structure of a given message type.  

1) Inputs
- Given Type: {type}
- seed_dir: {self.seed_dir}

2) What to Generate
Based on these constraints, you must design **new fields and message characteristics** that:
- strictly satisfy specification rules,
- are diverse and capable of reaching deeper code paths,
- and do NOT duplicate features of existing fields or messages of the same type.
- If additional details are required, you may extract information from {self.seed_dir}.  
- All generated fields must be acceptable to {self.target}.

3) Output Format:
Always follow the format below:
```json
{{
  "type": "FIELD_OR_MESSAGE_TYPE_X",
  "design": "Feature:\n1. ...\n2. ...\n3. ...\nConstraints:\n  1. Must not duplicate existing fields/features of the same type.\n  2. [Constraint from format_spec_DB or derived rule...]\n  3. [Constraint ensuring positional, length, or semantic validity...]\n  4. ..."
}}
```

4) Example Output:
```json
{{
  "type": "FIELD_OR_MESSAGE_TYPE_X",
  "design": "Feature:\n  1. Domain must embed \\"google.com\\".\n  2. Field encodes session token with random 16-byte hex.\n  3. Contains a nested structure representing protocol version.\nConstraints:\n  1. Must not duplicate any existing field of the same type.\n  2. A field must always have the same byte length as the length field.\n  3. B field must always appear immediately after C field.\n  4. D field must always precede E field.\n  5. F field must always use the literal name \\"USER\\".\n  6. G field must always carry the constant value \\"0x01\\"."
}}
```

5) Exit Criteria (**Important**):
- If no additional features or constraints can be proposed that would increase coverage for this type, IGNORE the Output Format and instead return:
```text
<<EXIT>>
```

6) Reasoning Process:
```text
- Step 1: Parsed the specification from format_spec_DB for {type}.
- Step 2: Identified existing fields and constraints to avoid duplication.
- Step 3: Derived structural and semantic constraints (length, order, constants, reserved keywords).
- Step 4: Checked {self.seed_dir} for additional information or examples.
- Step 5: Generated new field features consistent with constraints and diverse for deeper code coverage.
- Step 6: Validated that all constraints are satisfied and uniqueness is preserved.
- Step 7: Ensured the output follows the required format.
```
'''

                messages.append(ChatCompletionUserMessageParam(role="user", content=first_user_message))
                try:
                    messages = await mcp_client.process_messages_streaming(messages)
                except Exception as e:
                    printer.print(f"* * * [ERROR] Exception during message processing: {e}")
                    traceback.print_exc()
                    if t == max_tries - 1:
                        printer.print(f"* * * [WARNING] Maximum attempts reached ({max_tries}). Task is marked as incomplete.")
                        return "Failed"
                    continue
                response = messages[-1]['content']

                printer.print(f"* * * [INFO] Field Designer response (extract feature):\n{response}")
                response_json = {}
                try:
                    response_json = message_to_json(response)
                    if response_json == {}:
                        if t == max_tries - 1:
                            printer.print(f"* * * [WARNING] Maximum attempts reached ({max_tries}). Task is marked as incomplete.")
                            break
                        continue
                    self.add_memory_entries([json.dumps(response_json)])
                except json.JSONDecodeError:
                    printer.print(f"* * * [WARNING] Failed to parse JSON response. Retrying... ({t+1}/{max_tries})")
                    continue
                # Update memory
                # memory already updated above
                with open(os.path.join(RESULT_PATH, "component_DB", f"{self.id_counter}.json"), "w", encoding="utf-8") as f:
                    f.write(self.dump_memory())

                if "<<EXIT>>" in response:
                    printer.print(f"* * * [INFO] Field Designer chose to exit for type {type}.")
                    break  # Exit the loop for this type
                
                if t == max_tries - 1:
                    printer.print(f"* * * [WARNING] Maximum attempts reached ({max_tries}). Task is marked as incomplete.")
                    break

        return "Success"
