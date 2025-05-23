import argparse
import os
import subprocess
from .parser_lexer import Lexer, Parser, ProgramNode
from .compiler import Compiler

# Helper function to resolve include paths relative to the directory of the current file
def resolve_include_path(current_file_path: str, include_path_str: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(current_file_path))
    
    # Split the include path into parts (filename.Class.InnerClass)
    parts = include_path_str.split('.')
    
    # First part is the filename (without extension)
    filename = parts[0] + ".pgy"
    
    # Return the resolved file path
    return os.path.normpath(os.path.join(current_dir, filename))

# Helper function to extract import details from include path
def extract_import_details(include_path_str: str) -> tuple:
    parts = include_path_str.split('.')
    
    # Default: import the entire file
    filename = parts[0] + ".pgy"
    import_type = "file"
    import_name = None
    
    if len(parts) > 1:
        # Importing a specific item (class, inner class, macro)
        if parts[1].startswith('@'):
            # Macro import
            import_type = "macro"
            import_name = parts[1][1:]  # Remove '@' prefix
        else:
            # Class import
            import_type = "class"
            import_name = '.'.join(parts[1:])  # Join all remaining parts for nested classes
    
    return (filename, import_type, import_name)

# Function to parse a single file and return its classes, macros and further includes
def parse_single_file(file_path: str) -> tuple[list, list, list, bool, str | None, str | None]:
    # Returns (classes, macros, includes_to_process_next, success, error_type, error_message)
    if not os.path.exists(file_path):
        return [], [], [], False, "File Not Found", f"Error: File '{file_path}' not found."
    
    # Main loop in main() handles uniqueness to avoid re-parsing the same file.

    with open(file_path, 'r') as f:
        source_code = f.read()

    lexer = Lexer(source_code)
    try:
        tokens = lexer.tokenize()
    except Exception as e:
        return [], [], [], False, "Lexer Error", f"Lexer Error in '{file_path}': {e}"

    parser_instance = Parser(tokens)
    try:
        ast_program_node = parser_instance.parse() # This is a ProgramNode
        further_includes_to_process = []
        
        for include_node in ast_program_node.includes:
            # Get the file path without the class/inner class part
            file_path_to_include = resolve_include_path(file_path, include_node.path)
            
            # Include the whole file for processing
            further_includes_to_process.append((file_path_to_include, include_node.path))
        
        return ast_program_node.classes, ast_program_node.macros, further_includes_to_process, True, None, None
    except Exception as e:
        return [], [], [], False, "Parser Error", f"Parser Error in '{file_path}': {e}"

# Helper to filter classes and macros based on import path
def filter_imports(classes, macros, import_path):
    filename, import_type, import_name = extract_import_details(import_path)
    
    filtered_classes = []
    filtered_macros = []
    
    if import_type == "file":
        # Import everything from the file
        return classes, macros
    elif import_type == "class":
        # Import specific class(es)
        for cls in classes:
            if cls.name == import_name or cls.name.startswith(import_name + '.'):
                filtered_classes.append(cls)
    elif import_type == "macro":
        # Import specific macro
        for macro in macros:
            if macro.name == import_name:
                filtered_macros.append(macro)
    
    return filtered_classes, filtered_macros

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
    all_macros = []
    # Queue now contains tuples of (file_path, import_path)
    files_to_process_queue = [(initial_input_file_path, None)]  # Main file has no import path
    # This set tracks files that have been added to the queue to prevent cycles and redundant processing.
    files_added_to_queue = {initial_input_file_path}

    while files_to_process_queue:
        current_file_to_parse, import_path = files_to_process_queue.pop(0)
        
        classes_from_file, macros_from_file, next_includes, success, err_type, err_msg = parse_single_file(current_file_to_parse)
        
        if not success:
            print(f"{err_type}: {err_msg}") # Error message already includes file path
            return
        
        # If this is an import (not the main file), filter based on import path
        if import_path:
            classes_from_file, macros_from_file = filter_imports(classes_from_file, macros_from_file, import_path)
        
        all_classes.extend(classes_from_file)
        all_macros.extend(macros_from_file)
        
        for file_path, full_import_path in next_includes:
            if file_path not in files_added_to_queue:
                files_to_process_queue.append((file_path, full_import_path))
                files_added_to_queue.add(file_path) # Add here to prevent adding to queue multiple times

    master_ast = ProgramNode(classes=all_classes, includes=[], macros=all_macros)

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
