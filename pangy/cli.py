import argparse
import os
import subprocess
from .parser_lexer import Lexer, Parser, ProgramNode
from .compiler import Compiler

# Helper function to resolve include paths relative to the directory of the current file
def resolve_include_path(current_file_path: str, include_path_str: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(current_file_path))
    return os.path.normpath(os.path.join(current_dir, include_path_str))

# Function to parse a single file and return its classes and further includes
def parse_single_file(file_path: str) -> tuple[list, list, bool, str | None, str | None]:
    # Returns (classes, includes_to_process_next, success, error_type, error_message)
    if not os.path.exists(file_path):
        return [], [], False, "File Not Found", f"Error: File '{file_path}' not found."
    
    # Main loop in main() handles uniqueness to avoid re-parsing the same file.

    with open(file_path, 'r') as f:
        source_code = f.read()

    lexer = Lexer(source_code)
    try:
        tokens = lexer.tokenize()
    except Exception as e:
        return [], [], False, "Lexer Error", f"Lexer Error in '{file_path}': {e}"

    parser_instance = Parser(tokens)
    try:
        ast_program_node = parser_instance.parse() # This is a ProgramNode
        further_includes_to_process = []
        for include_node in ast_program_node.includes:
            resolved_path = resolve_include_path(file_path, include_node.path)
            further_includes_to_process.append(resolved_path)
        
        return ast_program_node.classes, further_includes_to_process, True, None, None
    except Exception as e:
        return [], [], False, "Parser Error", f"Parser Error in '{file_path}': {e}"

def main():
    arg_parser = argparse.ArgumentParser(description="Pangy Compiler")
    arg_parser.add_argument("input_file", help="Pangy source file to compile (.pgy)")
    arg_parser.add_argument("-o", "--output", help="Output file name for the executable (default: a.out or input_file base name)")
    arg_parser.add_argument("-S", "--assembly", action="store_true", help="Output assembly code (.s file) instead of an executable")
    arg_parser.add_argument("--ast", action="store_true", help="Print the Abstract Syntax Tree and exit")
    arg_parser.add_argument("--tokens", action="store_true", help="Print the tokens from the main input file and exit")

    args = arg_parser.parse_args()

    initial_input_file_path = os.path.abspath(args.input_file)
    if not initial_input_file_path.endswith(".pgy"):
        print(f"Error: Input file must be a .pgy file. Got: {initial_input_file_path}")
        return

    # Initial check for the main file moved here before queuing
    if not os.path.exists(initial_input_file_path):
        print(f"Error: Input file '{initial_input_file_path}' not found.")
        return

    if args.tokens:
        # Token processing remains simple, for the main file only
        # No include processing for --tokens for now
        with open(initial_input_file_path, 'r') as f:
            source = f.read()
        lexer = Lexer(source)
        try:
            tokens_list = lexer.tokenize()
            print("Tokens:")
            for token in tokens_list:
                print(token)
        except Exception as e:
            print(f"Lexer Error (main file for tokens): {e}")
        return

    all_classes = []
    files_to_process_queue = [initial_input_file_path]
    # This set tracks files that have been added to the queue to prevent cycles and redundant processing.
    files_added_to_queue = {initial_input_file_path}

    while files_to_process_queue:
        current_file_to_parse = files_to_process_queue.pop(0)
        
        classes_from_file, next_includes, success, err_type, err_msg = parse_single_file(current_file_to_parse)
        
        if not success:
            print(f"{err_type}: {err_msg}") # Error message already includes file path
            return
        
        all_classes.extend(classes_from_file)
        
        for include_path in next_includes: # these are already resolved absolute paths
            if include_path not in files_added_to_queue:
                files_to_process_queue.append(include_path)
                files_added_to_queue.add(include_path) # Add here to prevent adding to queue multiple times

    master_ast = ProgramNode(classes=all_classes, includes=[])

    if args.ast:
        print("Combined Abstract Syntax Tree (AST) from all included files:")
        import pprint
        pprint.pprint(master_ast)
        return

    compiler = Compiler()
    try:
        assembly_code = compiler.compile(master_ast)
    except Exception as e:
        print(f"Compiler Error: {e}")
        return

    base_name = os.path.splitext(os.path.basename(initial_input_file_path))[0]
    output_s_file = f"{base_name}.s"
    
    if args.output:
        output_executable_file = args.output
    else:
        output_executable_file = "a.out" if not args.assembly else base_name

    if args.assembly:
        s_output_target = args.output if args.output and args.output.endswith(".s") else output_s_file
        if args.output and not args.output.endswith(".s"):
             print(f"Warning: -S specified, output file '{args.output}' does not end with .s. Saving assembly to '{s_output_target}'")

        with open(s_output_target, 'w') as f:
            f.write(assembly_code)
        print(f"Assembly code saved to {s_output_target}")
        return

    temp_s_file = f"{base_name}_temp.s"
    with open(temp_s_file, 'w') as f:
        f.write(assembly_code)

    try:
        compile_command = ["gcc", "-no-pie", temp_s_file, "-o", output_executable_file]
        print(f"Running: {' '.join(compile_command)}")
        subprocess.run(compile_command, check=True)
        print(f"Executable saved to {output_executable_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during assembly/linking with gcc: {e}")
    except FileNotFoundError:
        print("Error: gcc command not found. Please ensure gcc is installed and in your PATH.")
    finally:
        if os.path.exists(temp_s_file):
            os.remove(temp_s_file)

if __name__ == '__main__':
    main()
