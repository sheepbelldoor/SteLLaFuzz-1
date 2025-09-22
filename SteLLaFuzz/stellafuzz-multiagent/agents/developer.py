import random
from openai.types.chat import (
    ChatCompletionUserMessageParam,
)

import json
import os
import traceback
import chromadb
from utils import message_to_json, RESULT_PATH, printer, format_assistant_responses

class DEVELOPER:
    def __init__(self, target: str, seed_dir: str, format_spec_DB: chromadb.api.Collection, sequence_DB: chromadb.api.Collection, component_DB: chromadb.api.Collection, type_list: list, seed_sequence_pairs: dict):
        self.target = target
        self.seed_dir = seed_dir
        self.format_spec_DB = format_spec_DB
        self.sequence_DB = sequence_DB
        self.component_DB = component_DB
        self.type_list = type_list
        self.seed_sequence_pairs = seed_sequence_pairs
        self.id_counter = 0

    def add_memory_entries(self, entries):
        ids = [str(self.id_counter + i) for i in range(len(entries))]
        self.id_counter += len(entries)
        self.sequence_DB.add(
            ids=ids,
            documents=entries
        )

    def retrieve_relevant_memory(self, db: str, query: str, n_results: int = 5) -> list[str]:
        """Retrieve relevant memory entries based on a query"""
        if db == "sequence":
            results = self.sequence_DB.query(query_texts=[query], n_results=n_results)
        elif db == "format_spec":
            results = self.format_spec_DB.query(query_texts=[query], n_results=n_results)
        elif db == "component":
            results = self.component_DB.query(query_texts=[query], n_results=n_results)
        else:
            raise ValueError(f"Unknown database: {db}")

        # semantically most similar memory entries to the current query
        return results['documents'][0]

    def dump_memory(self):
        return json.dumps(self.sequence_DB.get())

    ## develop new seed
    async def develop_new_seed(self, mcp_client, sequence_id: int, max_tries=3):
        for t in range(max_tries):
            messages = [{
                "role": "system",
                "content": f"You are a Developer whose role is to generate new seeds based on a given sequence."
            }]
            sequence_raw = self.sequence_DB.get(ids=[str(sequence_id)])['documents'][0]
            if isinstance(sequence_raw, str):
                try:
                    sequence = json.loads(sequence_raw)
                except Exception:
                    sequence = sequence_raw
            else:
                sequence = sequence_raw
            print(sequence)
            message_instructions = ""
            seq_len = len(sequence)
            for i in range(seq_len):
                result = self.retrieve_relevant_memory(db="component",
                                              query=f"What is the constraint and feature of {sequence[str(i+1)]}?")
                if result is not None and len(result) > 0:
                    random_number = random.randint(0, 4)
                    message_instructions += f"{sequence[str(i+1)]}:\n{result[random_number]}\n"

            first_user_message = f'''\
You are now acting as a Developer whose role is to generate new seeds based on a given sequence.

1) Instructions:
- Generate a new valid seed that {self.target} can accept.
- If unsure of the format or semantics, consult existing seeds in {self.seed_dir}.
- The new seed must be based on the provided sequence, and MUST include the fields or message types present in that sequence.

2) Inputs:
- target: {self.target}
- seed_dir: {self.seed_dir}
- sequence: {sequence}
- message_instructions:
{message_instructions}
- format_spec_DB: reference for constraints, byte-level structures
- component_DB: (optional) additional constraints or features for fields and message types

3) Development Rules:
1. You may use Python, C, C++, or Java to implement the seed generation.
2. If a suitable library exists for seed generation or mutation, use it actively.
3. If no such library exists, directly manipulate bytes or text to construct the seed.
4. The generated seed must:
  - respect the provided sequence order,
  - include the required fields or message types,
  - follow constraints from format_spec_DB (and component_DB if necessary),
  - be acceptable to {self.target}.
  - Binary-based protocols such as DNS, SSH, and TLS must be generated in binary form (e.g., 0x01 ...)
  - The generated seed must be saved as actual binary data, not as a string with escape sequences (e.g., use printf or equivalent methods to write true binary values in bash or shell).
5. Save the generated seed to {RESULT_PATH}/seed_DB.
  - File name can be arbitrary but must not duplicate existing names.

4) Failure / Success:
- If seed generation succeeds, return `"Success"` with the file name (including extension).
- If generation fails, return `"Failed"`.
- Output Format:
```json
{{
  "status": "Success",
  "seed_name": "your_generated_seed_name.raw"
}}
```
or
```json
{{
  "status": "Failed"
}}
```

5) Reasoning Process:
```text
- Step 1: Parsed the given sequence and identified required message types/fields.
- Step 2: Consulted format_spec_DB for structural and byte-level constraints.
- Step 3: Cross-checked component_DB for additional rules or unique field features.
- Step 4: Verified whether existing seeds in {self.seed_dir} provide useful patterns.
- Step 5: Selected an approach:
   - library-based generation if available,
   - otherwise manual byte/text manipulation.
- Step 6: Constructed the seed, embedding all mandatory sequence elements and respecting constraints.
- Step 7: Ensured the generated seed can be accepted by {self.target}.
- Step 8: Saved the seed under {RESULT_PATH}/seed_DB with a unique filename.
- Step 9: Returned "Success" + seed filename, or "Failed" if generation was not possible.
```\
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
            response_json = {}
            try:
                response_json = message_to_json(response)
                if response_json.get("status") == "Success" and "seed_name" in response_json:
                    if not os.path.exists(os.path.join(RESULT_PATH, "seed_DB", response_json["seed_name"])):
                        printer.print(f"* * * [ERROR] Generated seed file {response_json['seed_name']} does not exist in {RESULT_PATH}/seed_DB.")
                        if t == max_tries - 1:
                            printer.print(f"* * * [WARNING] Maximum attempts reached ({max_tries}). Task is marked as incomplete.")
                            return "Failed"
                        continue
                    self.seed_sequence_pairs[response_json["seed_name"]] = sequence
                    break
                else:
                    if t == max_tries - 1:
                        printer.print(f"* * * [WARNING] Maximum attempts reached ({max_tries}). Task is marked as incomplete.")
                        break
                    continue
            except json.JSONDecodeError:
                printer.print(f"* * * [WARNING] Failed to parse JSON response. Retrying... ({t+1}/{max_tries})")
                continue

        return "Success"
