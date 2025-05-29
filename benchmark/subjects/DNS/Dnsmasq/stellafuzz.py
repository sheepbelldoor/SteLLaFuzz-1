import os
import json
import argparse

from LLM.protocol_types import get_protocol_message_types
from LLM.specialized_structures import get_specialized_structures
from LLM.normal_sequence import get_message_sequences
from LLM.repeated_sequence import get_repeated_message_sequences
from LLM.testcases import get_test_cases
from LLM.structured_seed_message import get_structured_seed_message
from utility.utility import save_test_cases, load_seed_messages

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", "-p", type=str, required=True)
    parser.add_argument("--output_dir", "-o", type=str, required=False, default="results")
    parser.add_argument("--seed_messages", "-s", type=str, required=False, default=None, help="Path to initial seed messages")
    args = parser.parse_args()

    protocol = args.protocol
    output_dir = args.output_dir
    seed_messages_dir = args.seed_messages
    
    try:
        result = load_seed_messages(seed_messages_dir) if seed_messages_dir else (None, None)
        file_names, seed_messages = result
        # 1. Extract message types
        message_types: dict = get_protocol_message_types(protocol)

        # 2. Extract specialized structure
        specialized_structures: dict = get_specialized_structures(protocol, message_types)
        
        # 3. Generate message sequences
        message_sequences = {}
        message_sequences[1] = get_message_sequences(protocol, message_types, 1)
        message_sequences[3] = get_message_sequences(protocol, message_types, 3)
        message_sequences[5] = get_message_sequences(protocol, message_types, 5)
        repeated_message_sequences: dict = get_repeated_message_sequences(protocol, message_types)

        # 4. Generate test cases
        seed_index = 0
        if seed_messages:
            test_cases = {}
            for file_name, seed_message in zip(file_names, seed_messages):
                structured_seed_message = get_structured_seed_message(protocol, seed_message)
                test_cases[seed_index] = get_test_cases(protocol, message_sequences[1], specialized_structures, structured_seed_message)
                seed_index += 1
                test_cases[seed_index] = get_test_cases(protocol, message_sequences[3], specialized_structures, structured_seed_message)
                seed_index += 1
                test_cases[seed_index] = get_test_cases(protocol, message_sequences[5], specialized_structures, structured_seed_message)
                seed_index += 1
                if repeated_message_sequences:
                    test_cases[seed_index] = get_test_cases(protocol, repeated_message_sequences, specialized_structures, structured_seed_message)
                    seed_index += 1
                for seed_index, test_case in test_cases.items():
                    save_test_cases(test_case, output_dir, file_name)
        else:
            test_cases = {}
            test_cases[0] = get_test_cases(protocol, message_sequences, specialized_structures, None)
            if repeated_message_sequences:
                test_cases[1] = get_test_cases(protocol, repeated_message_sequences, specialized_structures, None)
            for seed_index, test_case in test_cases.items():
                save_test_cases(test_case, output_dir, "default")

    except Exception as e:
        print(f"Error processing protocol {protocol}: {e}")

if __name__ == "__main__":
    main()
