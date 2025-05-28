import argparse
import os
import subprocess
import shutil # Added for library installation
from .lexer import Lexer
from .parser import Parser, ProgramNode
from .compiler import Compiler

# Directory for globally installed Pangy libraries
PANGYLIBS_DIR = os.path.expanduser("~/.pangylibs")

# Helper function to extract module base path, import type, and specific item name
def extract_import_details(full_include_str: str) -> tuple[str, str, str | None]:
    """
    Parses a full include string and separates it into module path for file resolution,
    import type (file, class, macro), and the specific name of the item to import.

    Examples:
    "mylib.utils.MyClass" -> ("mylib.utils", "class", "MyClass")
    "mylib.utils.actualfile" -> ("mylib.utils.actualfile", "file", None)
    "mylib.utils.@mymacro" -> ("mylib.utils", "macro", "mymacro")
    "justfile" -> ("justfile", "file", None)
    "MyFileWithCaps" -> ("MyFileWithCaps", "file", None)
    """
    parts = full_include_str.split('.')
    if not parts:
        raise ValueError(f"Empty include string provided.")

    module_path_parts = []
    specific_import_name = None
    import_type = "file"  # Default

    # Determine the split point between module path and class/macro
    split_idx = len(parts) # Assume all parts form the module path initially

    for i, part in enumerate(parts):
        if part.startswith('@'):
            if i > 0: # Macro must be preceded by a file path part
                split_idx = i
                import_type = "macro"
                specific_import_name = part[1:] # Remove '@'
            else: # Macro at the start of the path (e.g., "@mymacro") is invalid
                raise ValueError(f"Invalid include: Macro '{part}' must be part of a file import (e.g., 'filename.@macro').")
            break
        # A part starting with an uppercase letter signals a class name,
        # provided it's not the first part of the include string (e.g. "MyLib.my_module")
        # and it's not the only part (e.g. "MyFile", which is a file).
        # Heuristic: If 'Lib.MyClass', then Lib.pgy, import MyClass.
        # If 'Lib.module.MyClass', then Lib/module.pgy, import MyClass.
        # If 'MyFileAlone', then MyFileAlone.pgy.
        if part[0].isupper():
            if i > 0: # e.g. mylib.MyClass
                split_idx = i
                import_type = "class"
            # If i == 0 and len(parts) > 1, e.g. MyLib.common.Helper
            # The loop will continue, and if common/Helper are not uppercase/@, they become part of module_path_parts
            # If 'MyLib.MyClass', split_idx becomes 1. module_path_parts = ['MyLib']
            # specific_import_name for class will be set after loop.
            # This means the first capitalized segment can be a directory/file part.
            # The split happens *before* the first recognized class/macro that isn't part of the file path itself.

    module_path_parts = parts[:split_idx]

    if import_type == "class":
        if split_idx < len(parts):
            specific_import_name = '.'.join(parts[split_idx:])
        else: # e.g. include MyFile (where MyFile.pgy contains class MyFile) - treat as whole file import
            import_type = "file" # No specific class name after module path

    if not module_path_parts:
        # This can happen if the path was invalid, e.g., starts with only a class/macro
        # or extract_import_details was called with an empty string.
        raise ValueError(f"Invalid include string: '{full_include_str}'. Could not determine module path.")

    module_base_str_for_resolution = '.'.join(module_path_parts)
    
    return module_base_str_for_resolution, import_type, specific_import_name


# Helper function to resolve include paths
def resolve_include_path(current_file_path: str, module_base_str: str) -> str | None:
    """
    Resolves the absolute path to a .pgy file based on a module base string.
    Searches locally first, then in PANGYLIBS_DIR.
    module_base_str examples: "mylib.utils", "singlefile"
    """
    file_path_parts = module_base_str.split('.')
    if not file_path_parts:
        return None

    # The last part is the filename stem, previous parts are directories.
    filename_stem = file_path_parts[-1]
    directory_parts = file_path_parts[:-1]
    
    # Path structure, e.g., "mylib/utils.pgy" or "singlefile.pgy"
    relative_pgy_file_path = os.path.join(*directory_parts, filename_stem + ".pgy")

    # 1. Local Search (relative to the current file's directory)
    current_dir = os.path.dirname(os.path.abspath(current_file_path))
    local_file_path = os.path.normpath(os.path.join(current_dir, relative_pgy_file_path))
    if os.path.exists(local_file_path):
        return local_file_path

    # 2. PangyLibs Search
    # PANGYLIBS_DIR is already an absolute path (expanded with expanduser)
    lib_file_path = os.path.normpath(os.path.join(PANGYLIBS_DIR, relative_pgy_file_path))
    if os.path.exists(lib_file_path):
        return lib_file_path
        
    return None


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
            full_include_str = include_node.path # e.g., "mylib.utils.MyClass"
            
            try:
                module_base_str, _, _ = extract_import_details(full_include_str)
            except ValueError as e:
                return [], [], [], False, "Include Error", f"Error parsing include statement '{full_include_str}' in '{file_path}': {e}"

            file_path_to_include = resolve_include_path(file_path, module_base_str)

            if file_path_to_include is None:
                return [], [], [], False, "File Not Found", f"Error in '{file_path}': Include '{module_base_str}.pgy' (from '{full_include_str}') not found locally or in pangylibs."
            
            # Pass the full_include_str for filter_imports later
            further_includes_to_process.append((file_path_to_include, full_include_str))
        
        return ast_program_node.classes, ast_program_node.macros, further_includes_to_process, True, None, None
    except Exception as e:
        return [], [], [], False, "Parser Error", f"Parser Error in '{file_path}': {e}"

# Helper to filter classes and macros based on import path
def filter_imports(classes, macros, import_path_full_str): # import_path_full_str is the original full include string
    # The module_base_str (first part of tuple) isn't needed here, as the file is already resolved and parsed.
    # We only need the import_type and specific import_name.
    _, import_type, import_name = extract_import_details(import_path_full_str)
    
    filtered_classes = []
    filtered_macros = []
    
    if import_type == "file":
        # Import everything from the file
        return classes, macros
    elif import_type == "class":
        # Import specific class(es)
        for cls in classes:
            # Handles "ParentClass.InnerClass" style import_name
            if cls.name == import_name or cls.name.startswith(import_name + '.'):
                filtered_classes.append(cls)
    elif import_type == "macro":
        # Import specific macro
        for macro_def in macros: # Assuming macros is a list of MacroDefinitionNode (or similar)
            if macro_def.name == import_name:
                filtered_macros.append(macro_def)
    
    return filtered_classes, filtered_macros

def handle_install_command(args):
    library_source_path = os.path.abspath(args.library_path)
    
    if not os.path.isdir(library_source_path):
        print(f"Error: Library path '{library_source_path}' is not a valid directory or does not exist.")
        return

    lib_name = os.path.basename(library_source_path)
    if not lib_name: # Handles cases like path ending with '/'
        lib_name = os.path.basename(os.path.dirname(library_source_path))

    if not lib_name:
        print(f"Error: Could not determine library name from path '{library_source_path}'.")
        return

    destination_path = os.path.join(PANGYLIBS_DIR, lib_name)

    if not os.path.exists(PANGYLIBS_DIR):
        try:
            os.makedirs(PANGYLIBS_DIR)
            print(f"Created library directory: {PANGYLIBS_DIR}")
        except OSError as e:
            print(f"Error creating library directory '{PANGYLIBS_DIR}': {e}")
            return
            
    if os.path.exists(destination_path):
        print(f"Warning: Library '{lib_name}' already exists in '{PANGYLIBS_DIR}'. Overwriting.")
        try:
            shutil.rmtree(destination_path)
        except OSError as e:
            print(f"Error removing existing library '{destination_path}': {e}")
            return
            
    try:
        shutil.copytree(library_source_path, destination_path, dirs_exist_ok=True) # dirs_exist_ok for robustness
        print(f"Library '{lib_name}' installed successfully to '{destination_path}'.")
    except Exception as e:
        print(f"Error installing library '{lib_name}': {e}")

def handle_compile_command(args):
    initial_input_file_path = os.path.abspath(args.input_file)
    if not initial_input_file_path.endswith(".pgy"):
        print(f"Error: Input file must be a .pgy file. Got: {initial_input_file_path}")
        return

    if not os.path.exists(initial_input_file_path):
        print(f"Error: Input file '{initial_input_file_path}' not found.")
        return

    if args.tokens:
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
    files_to_process_queue = [(initial_input_file_path, None)] 
    files_added_to_queue = {initial_input_file_path}

    while files_to_process_queue:
        current_file_to_parse, import_path_full = files_to_process_queue.pop(0)
        
        classes_from_file, macros_from_file, next_includes, success, err_type, err_msg = parse_single_file(current_file_to_parse)
        
        if not success:
            print(f"{err_type}: {err_msg}")
            return
        
        if import_path_full: # If this file was an import
            classes_from_file, macros_from_file = filter_imports(classes_from_file, macros_from_file, import_path_full)
        
        all_classes.extend(classes_from_file)
        all_macros.extend(macros_from_file)
        
        for file_path_resolved, full_include_path_str in next_includes:
            if file_path_resolved not in files_added_to_queue:
                files_to_process_queue.append((file_path_resolved, full_include_path_str))
                files_added_to_queue.add(file_path_resolved)

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
        # For debugging, you might want to print more details or re-raise with traceback
        # import traceback
        # traceback.print_exc()
        return

    base_name = os.path.splitext(os.path.basename(initial_input_file_path))[0]
    output_s_file = f"{base_name}.s"
    
    if args.output:
        output_executable_file = args.output
    else:
        # Adjust default output name based on whether assembly or executable is requested
        if args.assembly:
            output_executable_file = base_name # Will be forced to .s later if not specified
        else:
            output_executable_file = "a.out"


    if args.assembly:
        s_output_target = args.output if args.output and args.output.endswith(".s") else output_s_file
        # If -o was given but not ending in .s, warn and use default .s naming convention based on input
        if args.output and not args.output.endswith(".s"):
             s_output_target = f"{os.path.splitext(args.output)[0]}.s"
             print(f"Warning: -S specified, output file '{args.output}' does not end with .s. Saving assembly to '{s_output_target}'")
        
        with open(s_output_target, 'w') as f:
            f.write(assembly_code)
        print(f"Assembly code saved to {s_output_target}")
        return

    # Default behavior: compile to executable
    temp_s_file = f"{base_name}_temp.s" # Use a clearly temporary name
    with open(temp_s_file, 'w') as f:
        f.write(assembly_code)

    # Determine final executable name (respecting -o if provided)
    final_exe_name = output_executable_file
    if args.output:
        final_exe_name = args.output
        # If -o name ends with .s, but we are not in --assembly mode, it's confusing.
        # However, gcc can take foo.s as output name if told -o foo.s for an executable.
        # For simplicity, we'll let it be. User might intend a specific name.
    else: # No -o provided, not --assembly
        final_exe_name = base_name if os.path.exists(base_name) and os.path.isdir(base_name) else "a.out"
        if base_name == "a.out" : final_exe_name = "a.out.run" # Avoid overwriting a.out if input is a.out.pgy


    try:
        compile_command = ["gcc", "-no-pie", temp_s_file, "-o", final_exe_name]
        print(f"Running: {' '.join(compile_command)}")
        subprocess.run(compile_command, check=True)
        print(f"Executable saved to {final_exe_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error during assembly/linking with gcc: {e}")
    except FileNotFoundError:
        print("Error: gcc command not found. Please ensure gcc is installed and in your PATH.")
    finally:
        if os.path.exists(temp_s_file):
            os.remove(temp_s_file)

def main():
    arg_parser = argparse.ArgumentParser(description="Pangy Compiler & Tools")
    subparsers = arg_parser.add_subparsers(dest="command", help="Sub-commands")
    subparsers.required = False # Make subcommands optional for default behavior

    # Compile command parser
    compile_parser = subparsers.add_parser("compile", help="Compile a Pangy source file (.pgy)", aliases=['c'])
    compile_parser.add_argument("input_file", nargs='?', default=None, help="Pangy source file to compile (.pgy)")
    compile_parser.add_argument("-o", "--output", help="Output file name for the executable or assembly file")
    compile_parser.add_argument("-S", "--assembly", action="store_true", help="Output assembly code (.s file) instead of an executable")
    compile_parser.add_argument("--ast", action="store_true", help="Print the Abstract Syntax Tree and exit")
    compile_parser.add_argument("--tokens", action="store_true", help="Print the tokens from the main input file and exit")

    # Install command parser
    install_parser = subparsers.add_parser("install", help="Install a Pangy library to ~/.pangylibs", aliases=['i'])
    install_parser.add_argument("library_path", help="Path to the library folder to install")
    
    args = arg_parser.parse_args()

    if args.command == "install":
        handle_install_command(args)
    elif args.command == "compile":
        if not args.input_file:
            compile_parser.error("argument input_file: Can't be empty when 'compile' command is used.")
            return
        handle_compile_command(args)
    else:
        # Default behavior if no command is given or if input_file is provided at top level
        # This allows `pangy myfile.pgy` to work as `pangy compile myfile.pgy`
        # We need to check if first positional arg looks like a file for the default compile action
        # For simplicity, let's assume if no command, print help. User should be explicit.
        # Or, we can try to intelligently route `pangy file.pgy` to compile.
        # For now, making subcommands explicit is clearer.
        # If we want `pangy file.pgy` to work, we'd need a top-level input_file arg
        # and then dispatch based on its presence if no command is given.
        
        # Check if there are any arguments that might imply a default command
        # This is a bit tricky with argparse. A simpler approach:
        # If no command specified, print help.
        if hasattr(args, 'input_file') and args.input_file and not args.command: # Heuristic for `pangy file.pgy`
             # Manually set command to compile if input_file is present and no other command
             args.command = "compile"
             handle_compile_command(args)
        else:
            arg_parser.print_help()


if __name__ == '__main__':
    main()
