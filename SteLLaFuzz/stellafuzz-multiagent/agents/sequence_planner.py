from openai.types.chat import (
    ChatCompletionUserMessageParam,
)

import json
import os
import traceback
import chromadb
from utils import message_to_json, RESULT_PATH, printer, format_assistant_responses

class SEQUENCE_PLANNER:
    def __init__(self, target: str, seed_dir: str, format_spec_DB: chromadb.api.Collection, sequence_DB: chromadb.api.Collection, type_list: list, id_counter=0):
        self.target = target
        self.seed_dir = seed_dir
        self.format_spec_DB = format_spec_DB
        self.sequence_DB = sequence_DB
        self.type_list = type_list
        self.id_counter = id_counter

    def add_memory_entries(self, entries):
        ids = [str(self.id_counter + i) for i in range(len(entries))]
        self.id_counter += len(entries)
        self.sequence_DB.add(
            ids=ids,
            documents=entries
        )

    def retrieve_relevant_memory(self, query: str) -> list[str]:
        """Retrieve relevant memory entries based on a query"""
        results = self.sequence_DB.query(query_texts=[query], n_results=5)

        # semantically most similar memory entries to the current query
        return '\n'.join(results['documents'][0])

    def dump_memory(self):
        return json.dumps(self.sequence_DB.get())

        ## Plan sequence
    async def plan_sequence(self, mcp_client, max_tries=3):
        for t in range(max_tries):
            messages = [{
                "role": "system",
                "content": f"You are a Sequence Planner that proposes new type-level sequences to increase coverage."
            }]
            first_user_message = \
f'''\
You are a domain expert with deep understanding of {self.target}.

You are a Sequence Planner that proposes new type-level sequences to increase coverage.
You must generate concise sequences using the allowed {self.type_list}, consult sequence_DB for similar/existing sequences, check coverage, and iterate until a clearly higher-coverage candidate is found.

1)Inputs:
- Allowed Types
```text
{self.type_list}
```

2) sequence_DB (read-only): query by (ordered type list) to retrieve
- existing sequences (exact/near matches)
- coverage metrics per sequence (e.g., line/branch/state/function)
- optional metadata (run count, failures)
- Use "get_coverage_data_of_sequence" to query coverage_DB for coverage data of a sequence.

3) Objective
- Maximize expected coverage (line/branch/state/function) with minimal length and redundancy.
- Avoid sequences already known to be low-yield from sequence_DB.

4) Reasoning Process:
```text
- Step 1: Propose an initial candidate sequence using {self.type_list}.
- Step 2: Query sequence_DB for identical or similar sequences.
- Step 3: Fetch coverage metrics of the nearest neighbors.
- Step 4: Mutate/improve the candidate (reorder/insert/delete/replace types) guided by neighbor coverage.
- Step 5: Repeat Steps 2-4. As soon as a candidate with clearly higher expected coverage emerges, stop and save it.

5) Constraints:
- Use only types from {self.type_list}.
- Keep it short but expressive (allow longer only if justified by coverage evidence).
- Do not output reasoning or coverage numbers—output only the final sequence.
- Stopping Rule:
    - Stop when the new candidate dominates nearest neighbors on >=2 coverage metrics (or by a preset improvement threshold if provided).
    - If no improvement is possible within 5 iterations, return the best candidate found.

6) Output Format:
Return a JSON object with integer keys (starting from 1) and type values only.
- Example:
{{
"1": "TYPE_A",
"2": "TYPE_B",
"3": "TYPE_C"
}}\
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

            printer.print(f"* * * [INFO] Sequence Planner response (extract sequence):\n{response}")
            response_json = {}
            try:
                response_json = message_to_json(response)
                self.add_memory_entries([json.dumps(response_json)])
            except json.JSONDecodeError:
                printer.print(f"* * * [WARNING] Failed to parse JSON response. Retrying... ({t+1}/{max_tries})")
                continue
            # Update memory
            # memory already updated above
            with open(os.path.join(RESULT_PATH, "sequence_DB", f"{self.id_counter}.json"), "w", encoding="utf-8") as f:
                f.write(self.dump_memory())

            if t == max_tries - 1:
                printer.print(f"* * * [WARNING] Maximum attempts reached ({max_tries}). Task is marked as incomplete.")
                return "Failed"
            break

        return "Success"
