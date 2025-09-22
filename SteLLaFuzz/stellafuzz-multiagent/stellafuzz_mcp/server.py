import json
import logging

import os
import time
import shutil
import glob
import subprocess

import chromadb
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("stellafuzz")

@mcp.tool()
def list_files() -> str:
    """
    List all files in the seed directory for fuzzing.
    """
    seed_dir = os.getenv("SEED_DIR", ".")
    files = glob.glob(os.path.join(seed_dir, '**'), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    return "\n".join(files) if files else "No files found."

@mcp.tool()
def read_seed_file_as_hex_format_and_ascii_text(file_path: str) -> str:
    """
    For each byte, if it can be converted to an ASCII character, output the ASCII character;
    otherwise, output the byte as a two-digit hexadecimal string. Returns the file contents in this mixed format.
    Args:
        file_path (str): The path of the file to read.
    """
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        result = []
        prev_type = None  # 'ascii' or 'hex'
        for b in content:
            is_ascii = (b == 9 or b == 10 or b == 13 or 32 <= b <= 126)
            curr_type = 'ascii' if is_ascii else 'hex'
            if prev_type and curr_type != prev_type:
                result.append(' ')
            if is_ascii:
                result.append(chr(b))
            else:
                result.append(f"{b:02x} ")
            prev_type = curr_type
        return ''.join(result)
    except Exception as e:
        return f"[ERROR] Could not read file: {e}"

@mcp.tool()
def read_seed_file_as_hex_format(file_path: str) -> str:
    """
    Reads the file at the given file_path and returns its contents in hex format.
    Useful for handling files that cannot be read as ASCII text.
    Args:
        file_path (str): The path of the file to read.
    """
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        # Split by lines (preserve line breaks)
        lines = content.split(b'\n')
        hex_lines = []
        for i, line in enumerate(lines):
            # Each line: convert to hex, group by 2 chars
            hex_line = ' '.join([f"{b:02x}" for b in line])
            hex_lines.append(hex_line)
        # Re-add newlines to match original
        return '\n'.join(hex_lines)
    except Exception as e:
        return f"[ERROR] Could not read file as hex: {e}"

@mcp.tool()
def read_seed_file_as_ascii_text(file_path: str) -> str:
    """
    Reads the file at the given file_path and returns its contents as ASCII text.
    For binary files, not all bytes can be converted to ASCII characters, so reading as hex format may be more appropriate.
    Args:
        file_path (str): The path of the file to read.
    """
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        # Decode as ASCII, ignore non-ASCII bytes (they will be skipped)
        return content.decode('ascii', errors='ignore')
    except Exception as e:
        return f"[ERROR] Could not read file as ASCII text: {e}"

@mcp.tool()
def run_python_code(python_code: str) -> str:
    """
    Run python on the given Python code.
    Args:
        python_code (str): The Python test to run. Input format is ```python ... ```.
    """
    import re
    try:
        # Extract code from markdown code block if present
        match = re.search(r"```python(.*?)```", python_code, re.DOTALL | re.IGNORECASE)
        code = match.group(1).strip() if match else python_code.strip()
        # Write the extracted code to a temporary file
        temp_test_file = "temp_test_file.py"
        with open(temp_test_file, 'w') as f:
            f.write(code)
        # Run the code using python
        result = subprocess.run(['python3', temp_test_file], capture_output=True, text=True)
        # Clean up the temporary file
        os.remove(temp_test_file)
        output = result.stdout
        error = result.stderr
        if result.returncode == 0:
            return f"Execution succeeded.\nOutput:\n{output}"
        else:
            return f"Execution failed (exit code {result.returncode}).\nOutput:\n{output}\nError:\n{error}"
    except Exception as e:
        return f"[ERROR] Could not run python code: {e}"

@mcp.tool()
def run_c_code(c_code: str) -> str:
    """
    Run gcc on the given C code.
    Args:
        c_code (str): The C code to run. Input format is ```c ... ```.
    """
    import re
    try:
        # Extract code from markdown code block if present
        match = re.search(r"```c(.*?)```", c_code, re.DOTALL | re.IGNORECASE)
        code = match.group(1).strip() if match else c_code.strip()
        # Write the extracted code to a temporary file
        temp_c_file = "temp_test_file.c"
        with open(temp_c_file, 'w') as f:
            f.write(code)
        # Compile the C code using gcc
        executable_file = "temp_executable"
        compile_result = subprocess.run(['gcc', temp_c_file, '-o', executable_file], capture_output=True, text=True)
        if compile_result.returncode != 0:
            os.remove(temp_c_file)
            return f"Compilation failed (exit code {compile_result.returncode}).\nError:\n{compile_result.stderr}"
        # Run the compiled executable
        run_result = subprocess.run([f'./{executable_file}'], capture_output=True, text=True)
        # Clean up temporary files
        os.remove(temp_c_file)
        os.remove(executable_file)
        output = run_result.stdout
        error = run_result.stderr
        if run_result.returncode == 0:
            return f"Execution succeeded.\nOutput:\n{output}"
        else:
            return f"Execution failed (exit code {run_result.returncode}).\nOutput:\n{output}\nError:\n{error}"
    except Exception as e:
        return f"[ERROR] Could not run C code: {e}"

@mcp.tool()
def run_cpp_code(cpp_code: str) -> str:
    """
    Run g++ on the given C++ code.
    Args:
        cpp_code (str): The C++ code to run. Input format is ```cpp ... ```.
    """
    import re
    try:
        # Extract code from markdown code block if present
        match = re.search(r"```cpp(.*?)```", cpp_code, re.DOTALL | re.IGNORECASE)
        code = match.group(1).strip() if match else cpp_code.strip()
        # Write the extracted code to a temporary file
        temp_cpp_file = "temp_test_file.cpp"
        with open(temp_cpp_file, 'w') as f:
            f.write(code)
        # Compile the C++ code using g++
        executable_file = "temp_executable"
        compile_result = subprocess.run(['g++', temp_cpp_file, '-o', executable_file], capture_output=True, text=True)
        if compile_result.returncode != 0:
            os.remove(temp_cpp_file)
            return f"Compilation failed (exit code {compile_result.returncode}).\nError:\n{compile_result.stderr}"
        # Run the compiled executable
        run_result = subprocess.run([f'./{executable_file}'], capture_output=True, text=True)
        # Clean up temporary files
        os.remove(temp_cpp_file)
        os.remove(executable_file)
        output = run_result.stdout
        error = run_result.stderr
        if run_result.returncode == 0:
            return f"Execution succeeded.\nOutput:\n{output}"
        else:
            return f"Execution failed (exit code {run_result.returncode}).\nOutput:\n{output}\nError:\n{error}"
    except Exception as e:
        return f"[ERROR] Could not run C++ code: {e}"

@mcp.tool()
def run_java_code(java_code: str) -> str:
    """
    Run javac and java on the given Java code.
    Args:
        java_code (str): The Java code to run. Input format is ```java ... ```.
    """
    import re
    try:
        # Extract code from markdown code block if present
        match = re.search(r"```java(.*?)```", java_code, re.DOTALL | re.IGNORECASE)
        code = match.group(1).strip() if match else java_code.strip()
        # Write the extracted code to a temporary file
        temp_java_file = "TempTestFile.java"
        class_name_match = re.search(r'public\s+class\s+(\w+)', code)
        class_name = class_name_match.group(1) if class_name_match else "TempTestFile"
        with open(temp_java_file, 'w') as f:
            f.write(code)
        # Compile the Java code using javac
        compile_result = subprocess.run(['javac', temp_java_file], capture_output=True, text=True)
        if compile_result.returncode != 0:
            os.remove(temp_java_file)
            return f"Compilation failed (exit code {compile_result.returncode}).\nError:\n{compile_result.stderr}"
        # Run the compiled Java class
        run_result = subprocess.run(['java', class_name], capture_output=True, text=True)
        # Clean up temporary files
        os.remove(temp_java_file)
        os.remove(f"{class_name}.class")
        output = run_result.stdout
        error = run_result.stderr
        if run_result.returncode == 0:
            return f"Execution succeeded.\nOutput:\n{output}"
        else:
            return f"Execution failed (exit code {run_result.returncode}).\nOutput:\n{output}\nError:\n{error}"
    except Exception as e:
        return f"[ERROR] Could not run Java code: {e}"

@mcp.tool()
def run_command(command: str) -> str:
    """
    Run an arbitrary linux shell command.
    Args:
        command (str): The shell command to run.
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        output = result.stdout
        error = result.stderr
        if result.returncode == 0:
            return f"Command succeeded.\nOutput:\n{output}"
        else:
            return f"Command failed (exit code {result.returncode}).\nOutput:\n{output}\nError:\n{error}"
    except Exception as e:
        return f"[ERROR] Could not run command: {e}"

@mcp.tool()
def get_data_from_DB_using_RAG(query: str, DB_name: str, n_results: int) -> str:
    """
    Retrieve relevant data from a database using Retrieval-Augmented Generation (RAG) techniques.
    Args:
        query (str): The query to search in the database. The query must be written as a complete sentence. (e.g., "What is the structure of DESCRIBE message?")
        DB_name (str): The name of the database. Supported databases: "component_DB", "format_spec_DB", "sequence_DB".
        n_results (int): The number of top results to retrieve.
    """
    chromadb_client = chromadb.Client()
    if DB_name not in ["component_DB", "format_spec_DB", "sequence_DB"]:
        return f"[ERROR] Unsupported DB_name: {DB_name}. Supported databases are: component_DB, format_spec_DB, sequence_DB."

    db = chromadb_client.get_or_create_collection(name=DB_name)
    db_dir = os.path.join(os.getenv("PATH_TO_DB"), DB_name)
    json_files = [f for f in os.listdir(db_dir) if f.endswith('.json') and f[:-5].isdigit()]
    if not json_files:
        return f"[ERROR] No Data in {DB_name}."    
    max_idx = max([int(f[:-5]) for f in json_files])
    max_json = f"{max_idx}.json"
    db_path = os.path.join(db_dir, max_json)

    # Add documents and RAG Retrieval
    with open(db_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        db = chromadb_client.get_or_create_collection(name=DB_name)
        db.add(ids=data['ids'], documents=data['documents'])
        results = db.query(query_texts=[query], n_results=n_results)

    # Result Formatting
    pretty_results = []
    ids = results.get('ids', [[]])[0]
    docs = results.get('documents', [[]])[0]
    for i in range(len(ids)):
        pretty_results.append({
            'id': ids[i],
            'document': docs[i]
        })
    return json.dumps(pretty_results, ensure_ascii=False, indent=2)

@mcp.tool()
def get_coverage_data_of_sequence(sequence: str) -> str:
    """
    Get the coverage data of the sequence with the given sequence from coverage_DB.
    It returns the top 3 most relevant coverage data entries.
    Args:
        sequence (str): The sequence to get coverage data for. (example: "[MESSAGE1, MESSAGE2, ...]")
    """
    chromadb_client = chromadb.Client()
    DB_name = "coverage_DB"
    db = chromadb_client.get_or_create_collection(name=DB_name)
    db_dir = os.path.join(os.getenv("PATH_TO_DB"), DB_name)
    json_files = [f for f in os.listdir(db_dir) if f.endswith('.json') and f[:-5].isdigit()]
    if not json_files:
        return f"[ERROR] No Data in {DB_name}."    
    max_idx = max([int(f[:-5]) for f in json_files])
    max_json = f"{max_idx}.json"
    db_path = os.path.join(db_dir, max_json)

    # Add documents and RAG Retrieval
    with open(db_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        db = chromadb_client.get_or_create_collection(name=DB_name)
        db.add(ids=data['ids'], documents=data['documents'])
        results = db.query(query_texts=[sequence], n_results=3)

    # Result Formatting
    pretty_results = []
    ids = results.get('ids', [[]])[0]
    docs = results.get('documents', [[]])[0]
    for i in range(len(ids)):
        pretty_results.append({
            'id': ids[i],
            'document': docs[i]
        })
    return json.dumps(pretty_results, ensure_ascii=False, indent=2)


@mcp.tool()
def measure_coverage(test_file_path: str) -> str:
    """
    Measure code coverage for the generated test files in the target directory.
    Args:
        test_file_path (str): The path of the Python test file to run.
    """
    pass


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
