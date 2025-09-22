from openai.types.chat import (
    ChatCompletionUserMessageParam,
)

import json
import os
import traceback
import chromadb
from utils import message_to_json, RESULT_PATH, printer, format_assistant_responses
import glob

class FORMAT_ANALYST:
    def __init__(self, target: str, seed_dir: str, seed_sequence_pairs: dict, format_spec_DB: chromadb.api.Collection, sequence_DB: chromadb.api.Collection, type_list: list):
        self.target = target
        self.seed_dir = seed_dir
        self.format_spec_DB = format_spec_DB
        self.sequence_DB = sequence_DB
        self.type_list = type_list
        self.seed_sequence_pairs = seed_sequence_pairs
        self.id_counter = 0
        self.id_counter_sequence = 0

    def add_memory_entries(self, entries):
        ids = [str(self.id_counter + i) for i in range(len(entries))]
        self.id_counter += len(entries)
        self.format_spec_DB.add(
            ids=ids,
            documents=entries
        )

    def add_sequence_memory_entries(self, entries):
        ids = [str(self.id_counter_sequence + i) for i in range(len(entries))]
        self.id_counter_sequence += len(entries)
        self.sequence_DB.add(
            ids=ids,
            documents=entries
        )

    def retrieve_relevant_memory(self, query: str) -> list[str]:
        """Retrieve relevant memory entries based on a query"""
        results = self.format_spec_DB.query(query_texts=[query], n_results=5)

        # semantically most similar memory entries to the current query
        return '\n'.join(results['documents'][0])

    def dump_memory(self):
        return json.dumps(self.format_spec_DB.get())
    
    def dump_sequence_memory(self):
        return json.dumps(self.sequence_DB.get())

    async def reflect_on_previous_attempt(self, messages, task, mcp_client):
        reflection_request = f'''The following are the trajectory of your attempt to perform the following task: 
{task}

on the website url {self.target_website_url}.

Message trajectory:
{str(messages)}

Generate brief one-line reflections and takeaways that you can refer to in future attempts. Your answer should only contain one reflection per a line, and no other text.'''
        messages = [{
            "role": "user",
            "content": reflection_request
        }]
        messages = await mcp_client.process_messages_streaming(messages) # TODO: disable tool calls
        response = messages[-1]['content']
        reflections = response.splitlines()

        self.add_memory_entries(reflections)

    ## Analyze from Specification
    async def analyze_from_specification(self, mcp_client, max_tries=3):
        ## Retrieve all format types (e.g., file extensions, message types) that the target can accept as input
        for t in range(max_tries):
            messages = [{
                "role": "system",
                "content": f"The Format Analyst agent extracts type definitions, structural hierarchies, and constraints of a format through both specification-based and input-based analysis, storing the results in a database for subsequent agents to utilize."
            }]
            first_user_message = \
f'''\
You are a domain expert with deep understanding of {self.target}.
[TARGET_TYPE] is one of: protocol, file_library, application.

Your task is to extract authoritative, complete information as follows:

1) What to Extract
If [TARGET_TYPE] = protocol
Extract all client-to-server message types ONLY defined in {self.target} (including extended/optional messages defined in official specifications or recognized standards).
Exclude server-to-client messages.
Otherwise if [TARGET_TYPE] = file_library or application
Extract all accepted input file formats supported by {self.target}.

2) Output Format
Return a single JSON object with one of the following shapes:

3) Examples
- Example (Protocol: OpenSSH):
```json
{{
  "target": "OpenSSH",
  "target_type": "protocol",
  "types": [
    {{"name": "KEXINIT"}},
    {{"name": "SERVICE_REQUEST"}},
    {{"name": "USERAUTH_REQUEST"}},
    ...
  ]
}}
```
- Example (File Library: libpng):
```json
{{
  "target": "libpng",
  "target_type": "file_library",
  "types": [
    {{"name": ".png"}},
    ...
  ]
}}
```

4) Sources (Authoritative Only)
- Base the extraction strictly on official docs, RFCs/standards, or recognized authoritative references.
- Provide a Sources list with titles and URLs.
- Avoid speculation.
```text
Sources:
- RFC 4253: SSH Transport Layer Protocol - https://www.rfc-editor.org/rfc/rfc4253
- Official libpng manual - https://libpng.sourceforge.io/
```

5) Reasoning Process (Step-by-Step)
Explain how completeness was ensured:
```text
Reasoning Process:
- Step 1: Reviewed [document A] to enumerate [messages|formats].
- Step 2: Cross-checked with [document B] for extensions/optional items.
- Step 3: Verified identifiers (codes, magic numbers, MIME types) against [document C].
- Step 4: Noted build-time/conditional support where applicable.
```

6) Error Handling & Ambiguities
If anything is unclear or unofficial:
```text
Potential Candidates:
- ITEM_X: Mentioned in [source]; not confirmed in official spec.
- ITEM_Y: Requires vendor patch or compile-time option; unclear default.
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
            printer.print(f"* * * [INFO] Format Analyst response (extract types):\n{response}")
            response_json = {}
            try:
                response_json = message_to_json(response)
                # self.add_memory_entries([json.dumps(entry) for entry in response_json])
            except json.JSONDecodeError:
                printer.print(f"* * * [WARNING] Failed to parse JSON response. Retrying... ({t+1}/{max_tries})")
                continue

            for format in response_json.get('types', []):
                printer.print(f"* * * [INFO] Detected format type: {format}")
                self.type_list.append(format['name'])

            break
            
        ## Analyze format specification
        for type in self.type_list:
            for t in range(max_tries):
                messages = [{
                    "role": "system",
                    "content": f"The Format Analyst agent extracts type definitions, structural hierarchies, and constraints of a format through both specification-based and input-based analysis, storing the results in a database for subsequent agents to utilize."
                }]
                first_user_message = \
f'''\
You are a domain expert with deep understanding of {self.target}.
[TARGET_TYPE] is one of: protocol, file_library, application.

Your task is to extract authoritative, complete information as follows:

1) What to Extract
Extract comprehensive, documentation-backed details for the {type} of {self.target}, focusing on *what the structure contains* and *how it is used* (exclude endianness/framing-level encoding such as "big endian"). Concretely, extract:
- Overview: purpose/role of this {type} within {self.target}; when/why it appears.
- Fields: for each field, provide
  - name, description (what it means/function), required (true/false), multiplicity (e.g., 1, 0..1, 0..N), order/position (if specified), location/path (e.g., header.payload.field),
  - constraints: constants, enums, ranges, length rules, regex/patterns, and cross-field/relational constraints as human-readable predicates (e.g., "if color_type==3 then bit_depth in {1,2,4,8}"),
  - dependencies: references to other fields or previously negotiated/session values,
  - default/sentinel values (if any), and whether a field may be ignored/unknown.
- Magic numbers / type codes: their exact values, where they occur, and their semantic meaning.
- Functional semantics: the effect or behavior this {type} triggers or represents (e.g., handshake initiation, capability advertisement, configuration header).
- Relationships: ordering/containment/state relationships with other types (e.g., must_precede, allowed_parent, allowed_children), and any preconditions/postconditions (protocol state assumptions).
- Error/handling rules: validation conditions, error codes, or recovery behavior mentioned by the docs.
- Security notes (only if directly stated by authoritative sources).
- Extras: any additional attributes explicitly described by authoritative sources that are important but not listed above.

2) Output Format
Return a single JSON object with one of the following shapes:

3) Examples
- Example:
```json
{{
  "target": "{self.target}",
  "type_name": "<name>",
  "artifact_kind": "message|record|chunk|section",
  "description": "<purpose/role>",

  "fields": [
    {{
      "name": "<field>",
      "description": "<meaning>",
      "required": true,
      "constraints": {{
        "const": "<value if fixed>",
        "enum": ["<v1>", "<v2>"],
        "range": {{"min": "<num>", "max": "<num>"}},
        "relational": ["<cross-field rule>"]
      }},
      "dependencies": ["<other field>"]
    }}
  ],

  "magic": [
    {{"value": "<literal>", "meaning": "<semantics>"}}
  ],

  "relations": [
    {{"type": "must_precede|must_follow|allowed_child", "with": "<other type>"}}
  ],

  "functional_semantics": "<behavior/effect>",
  "errors": ["<validation rules>"],
  "extras": {{"<key>": "<value>"}},
  "provenance": [
    {{"source": "RFC/Manual", "uri": "<URL>"}}
  ]
}}
```

4) Sources (Authoritative Only)
- Base the extraction strictly on official docs, RFCs/standards, or recognized authoritative references.
- Provide a Sources list with titles and URLs.
- Avoid speculation.
```text
Sources:
- RFC 4253: SSH Transport Layer Protocol - https://www.rfc-editor.org/rfc/rfc4253
- Official libpng manual - https://libpng.sourceforge.io/

5) Reasoning Process (Step-by-Step)
Explain how completeness was ensured and how each constraint/dependency was grounded in sources:
```text
- Step 1: Enumerated all fields and mandatory/optional status from the primary spec sections.
- Step 2: Collected constraints (const/enum/range/length/pattern) and cross-field predicates from normative language (MUST/SHALL).
- Step 3: Mapped magic numbers/type codes and their semantics; linked ordering/relations to neighboring types or states.
- Step 4: Added functional semantics, error handling, and security notes exactly as stated by sources.
- Step 5: Reviewed for missing but important attributes explicitly present in the docs and placed them under "extras".
- Step 6: Verified each item with citations and version provenance.
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
                    self.add_memory_entries([json.dumps(response_json)])
                except json.JSONDecodeError:
                    printer.print(f"* * * [WARNING] Failed to parse JSON response. Retrying... ({t+1}/{max_tries})")
                    continue

                # Update memory
                # memory already updated above
                with open(os.path.join(RESULT_PATH, "format_spec_DB", f"{self.id_counter}.json"), "w", encoding="utf-8") as f:
                    f.write(self.dump_memory())

                if t == max_tries - 1:
                    printer.print(f"* * * [WARNING] Maximum attempts reached ({max_tries}). Task is marked as incomplete.")
                    return "Failed"
                
                break
        return "Success"

    ## Analyze from Inputs
    async def analyze_from_inputs(self, mcp_client, input_dir, max_tries=3):
        for t in range(max_tries):
            messages = [{
                "role": "system",
                "content": f"The Format Analyst agent extracts type definitions, structural hierarchies, and constraints of a format through both specification-based and input-based analysis, storing the results in a database for subsequent agents to utilize."
            }]
            first_user_message = \
f'''\
You are a domain expert with deep understanding of {self.target}.

Your task is to analyze the given seed input and extract structured, authoritative information.  
Focus on capturing **exact field values**, **dependencies**, and **sequence-level context** to support generation of valid future seeds.

1) Input
You are given a seed input of {self.target}:
```text
{input_dir}
```
2) What to Extract
- Identify the type(s) this seed corresponds to (message, record, chunk, section, etc.).
- Extract every field present in the seed:
  - field name
  - semantic description (purpose/function)
  - exact value from the seed
  - constraints (constants, enums, ranges, relational rules)
  - dependencies (references to other fields or sequence context)
  - required level
    - "mandatory": fields strictly required for valid execution (e.g., username, authentication data, protocol identifiers).
    - "optional": fields not strictly required, but may affect behavior, metadata, or optional features.
- Detect and record magic numbers, type codes, or identifiers.
- If the seed is part of a sequence, extract:
  - ordering of messages/records
  - cross-message dependencies (e.g., field values negotiated in one step reused later)
  - state transitions or pre/postconditions
- Functional semantics: what does this seed or sequence achieve in {self.target}?
- Errors or anomalies if the seed violates any documented rule.
- Direct parsing code can be written to separate and analyze in unit-by-unit basis.
- Extras: any additional information of the specification in "format_spec_DB" that can help analyze the seed.

3) Output Format
- Return the result as a single JSON object.
- The JSON does not need to follow a fixed schema; instead, organize the information into logical keys (e.g., "type", "fields", "sequence", "relations", "magic", "semantics", etc.).
- Always include exact field values where available.
- Include lists (e.g., "fields": [...], "sequence": [...]) when multiple items exist.

4) Reasoning Process (Step-by-Step)
Explain how completeness was ensured:
```text
Reasoning Process:
- Step 1: Parsed the raw seed input into structural units (messages/records).
- Step 2: Mapped each unit to its documented type in {self.target}.
- Step 3: Extracted exact field values and validated them against specification rules.
- Step 4: Captured constraints and dependencies between fields and across sequence steps.
- Step 5: Identified magic numbers and type codes.
- Step 6: Checked for anomalies or violations.
- Step 7: Verified correctness using authoritative sources.
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
                self.add_memory_entries([json.dumps(response_json)])
            except json.JSONDecodeError:
                printer.print(f"* * * [WARNING] Failed to parse JSON response. Retrying... ({t+1}/{max_tries})")
                continue

            # Update memory
            with open(os.path.join(RESULT_PATH, "format_spec_DB", f"{self.id_counter}.json"), "w", encoding="utf-8") as f:
                f.write(self.dump_memory())

            if t == max_tries - 1:
                printer.print(f"* * * [WARNING] Maximum attempts reached ({max_tries}). Task is marked as incomplete.")
                return "Failed"
            break

        return "Success"

    ## Analyze from inputs of sequence
    async def extract_sequence_from_inputs(self, mcp_client, input_dir, max_tries=3):
        files = glob.glob(os.path.join(input_dir, '**'), recursive=True)
        files = [f for f in files if os.path.isfile(f)]
        for file in files:
            printer.print(f"* * * [INFO] Extracting sequence from input file: {file}")
            
            for t in range(max_tries):
                messages = [{
                    "role": "system",
                    "content": f"The Format Analyst agent extracts sequence from the given seed input."
                }]
                first_user_message = \
f'''\
You are a domain expert with deep understanding of {self.target}.

Your task is to analyze the given seed input and extract ONLY the SEQUENCE ORDER using {self.type_list}.  
Do not extract full field details. Output must only be a mapping of step index to type.

1) Input
```text
{file}
```

2) Allowed Types
```
{self.type_list}
```

3) Output Format
Return a JSON object with integer keys (starting from 1) and type values.
```json
{{
    "1": "TYPE_A",
    "2": "TYPE_B",
    "3": "TYPE_C"
}}
```

4) Reasoning Process:
```text
- Step 1: Segment the seed into ordered units.
- Step 2: Map each unit to one of {self.type_list}.
- Step 3: Assign each unit an index (1-based).
- Step 4: Output as {{index: "TYPE"}} pairs only.
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
                    self.add_sequence_memory_entries([json.dumps(response_json)])
                    self.seed_sequence_pairs[file] = response_json
                except json.JSONDecodeError:
                    printer.print(f"* * * [WARNING] Failed to parse JSON response. Retrying... ({t+1}/{max_tries})")
                    continue

                # Update memory
                # memory already updated above
                with open(os.path.join(RESULT_PATH, "sequence_DB", f"{self.id_counter_sequence}.json"), "w", encoding="utf-8") as f:
                    f.write(self.dump_sequence_memory())

                if t == max_tries - 1:
                    printer.print(f"* * * [WARNING] Maximum attempts reached ({max_tries}). Task is marked as incomplete.")
                    return "Failed"
                break
        return "Success"
