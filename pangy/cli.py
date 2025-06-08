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
    # includes_to_process_next is a list of tuples: (resolved_path_to_pgy_file, (import_type, specific_name_to_import))
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
            full_include_str = include_node.path # e.g., "mylib.utils.MyClass", "mylib.utils.actualfile", "mylib.utils.@mymacro"
            
            resolved_file_to_include = None
            determined_import_type = "file" # Default: assume importing the whole file
            determined_specific_name = None

            # Attempt 1: Treat the full include string as a potential direct file path
            # e.g., "mylib.utils.actualfile" -> try to load "mylib/utils/actualfile.pgy"
            potential_direct_file_path = resolve_include_path(file_path, full_include_str)
            if potential_direct_file_path and os.path.exists(potential_direct_file_path):
                resolved_file_to_include = potential_direct_file_path
                determined_import_type = "file"
                determined_specific_name = None
            else:
                # Attempt 2: Parse the include string for module, type, and specific name
                try:
                    module_base_str, import_type_from_extract, specific_name_from_extract = extract_import_details(full_include_str)
                except ValueError as e:
                    return [], [], [], False, "Include Error", f"Error parsing include statement '{full_include_str}' in '{file_path}': {e}"
                
                resolved_path_for_module = resolve_include_path(file_path, module_base_str)

                if resolved_path_for_module and os.path.exists(resolved_path_for_module):
                    resolved_file_to_include = resolved_path_for_module
                    determined_import_type = import_type_from_extract
                    determined_specific_name = specific_name_from_extract
                    # If extract_import_details said "file" but specific_name is None, it means it was like "justafile".
                    # If import_type_from_extract is "class" but specific_name_from_extract is None
                    # (e.g. "MyLib" where MyLib.pgy exists but extract_import_details thought MyLib was a class of itself),
                    # we should ensure it's treated as a "file" import for MyLib.pgy.
                    # This is subtle: extract_import_details might identify "MyLib" in "include MyLib" as a potential class
                    # if there's no further path. But if resolve_include_path then finds "MyLib.pgy", it's a file import.
                    if import_type_from_extract == "class" and specific_name_from_extract is None:
                         # This case indicates that extract_import_details might have thought "MyModule" itself was a class
                         # but we are resolving "MyModule.pgy". So, it's a full file import.
                         determined_import_type = "file" 
                else:
                    # Failed to resolve the module path even after parsing
                    # Construct a more informative error message
                    err_msg_detail = f"Could not resolve '{module_base_str}.pgy' for include '{full_include_str}'."
                    if potential_direct_file_path:
                        err_msg_detail += f" Also tried '{full_include_str}.pgy' which resolved to '{potential_direct_file_path}' but was not found."
                    else:
                        err_msg_detail += f" An attempt to treat '{full_include_str}' as a direct file path also failed to yield an existing file."
                    return [], [], [], False, "File Not Found", f"Error in '{file_path}': {err_msg_detail} Searched locally and in pangylibs."

            if resolved_file_to_include is None:
                # This should ideally be caught by the logic above, but as a safeguard:
                return [], [], [], False, "File Not Found", f"Error in '{file_path}': Include '{full_include_str}' could not be resolved to an existing .pgy file."
            
            further_includes_to_process.append((resolved_file_to_include, (determined_import_type, determined_specific_name)))
        
        return ast_program_node.classes, ast_program_node.macros, further_includes_to_process, True, None, None
    except Exception as e:
        return [], [], [], False, "Parser Error", f"Parser Error in '{file_path}': {e}"

# Helper to filter classes and macros based on import path
def filter_imports(classes, macros, import_type: str, import_name: str | None): # import_path_full_str is the original full include string
    # The module_base_str (first part of tuple) isn't needed here, as the file is already resolved and parsed.
    # We only need the import_type and specific import_name.
    # _, import_type, import_name = extract_import_details(import_path_full_str) # No longer calling this
    
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

    # Master lists for the entire program AST
    all_classes_for_master_ast = []
    all_macros_for_master_ast = []
    
    # Queue for processing include directives: (absolute_path_to_pgy_file, (import_type, specific_name_to_import))
    # The specific_name_to_import is used to filter which parts of the file are *explicitly* imported,
    # but the whole file is parsed and its contents (all classes/macros) are made available.
    import_directives_queue = [(initial_input_file_path, ("file", None))]
    
    # Set to track *specific import directives* that have been queued to avoid redundant processing of the *same directive*.
    # (e.g., if "include mylib.MyClass" appears twice, we only add it to the queue once).
    # The key is (resolved_pgy_file_path, import_type, specific_name).
    processed_import_directives = set()
    processed_import_directives.add((initial_input_file_path, "file", None))

    # Cache to store analysis results of a .pgy file to avoid re-parsing.
    # Key: absolute_path_to_pgy_file
    # Value: (list_of_ClassNodes, list_of_MacroDefNodes, list_of_further_includes_from_that_file)
    file_analysis_cache = {}

    # Flag to track if math library is used
    uses_math_library = False

    while import_directives_queue:
        current_pgy_file_to_analyze, specific_import_details = import_directives_queue.pop(0)
        # specific_import_details: (import_type, specific_name) for *this particular import directive*

        # Check cache first
        if current_pgy_file_to_analyze in file_analysis_cache:
            classes_from_this_file, macros_from_this_file, includes_from_this_file = file_analysis_cache[current_pgy_file_to_analyze]
        else:
            # Parse the file
            parsed_classes, parsed_macros, further_includes_from_parse, success, err_type, err_msg = parse_single_file(current_pgy_file_to_analyze)
            if not success:
                print(f"{err_type}: {err_msg}")
                return
            
            # Check for math library usage
            for cls in parsed_classes:
                for method in cls.methods:
                    # Check if any method contains a math library call
                    def check_for_math_library(node):
                        nonlocal uses_math_library
                        if hasattr(node, 'statements'):
                            for stmt in node.statements:
                                check_for_math_library(stmt)
                        elif hasattr(node, 'expression'):
                            check_for_math_library(node.expression)
                        elif hasattr(node, 'condition'):
                            check_for_math_library(node.condition)
                            check_for_math_library(node.then_block)
                            if node.else_block:
                                check_for_math_library(node.else_block)
                        elif hasattr(node, 'body'):
                            check_for_math_library(node.body)
                        elif hasattr(node, 'left') and hasattr(node, 'right'):
                            check_for_math_library(node.left)
                            check_for_math_library(node.right)
                        elif hasattr(node, 'operand_node'):
                            check_for_math_library(node.operand_node)
                        elif hasattr(node, 'arguments'):
                            for arg in node.arguments:
                                check_for_math_library(arg)
                        # Check if this is a CLibraryCallNode with library="m"
                        if hasattr(node, 'library') and node.library == "m":
                            uses_math_library = True
                            
                    if method.body:
                        check_for_math_library(method.body)
            
            classes_from_this_file = parsed_classes
            macros_from_this_file = parsed_macros
            includes_from_this_file = further_includes_from_parse # These are (resolved_path, (type, name))

            # Cache the results of parsing this file
            file_analysis_cache[current_pgy_file_to_analyze] = (classes_from_this_file, macros_from_this_file, includes_from_this_file)

        # Add ALL classes and macros from this parsed file to our master lists.
        # Deduplication will happen later.
        all_classes_for_master_ast.extend(classes_from_this_file)
        all_macros_for_master_ast.extend(macros_from_this_file)
        
        # Now, process the include directives found *within* current_pgy_file_to_analyze
        for resolved_next_pgy_file, next_import_details_tuple in includes_from_this_file:
            # next_import_details_tuple is (determined_import_type, determined_specific_name)
            # This represents an include directive like "include some.other.Module" found inside current_pgy_file_to_analyze.
            
            import_key_for_directive = (resolved_next_pgy_file, next_import_details_tuple[0], next_import_details_tuple[1])
            
            if import_key_for_directive not in processed_import_directives:
                import_directives_queue.append((resolved_next_pgy_file, next_import_details_tuple))
                processed_import_directives.add(import_key_for_directive)

    # Deduplicate classes and macros by name (first encountered wins)
    final_classes = []
    seen_class_names = set()
    for cls_node in all_classes_for_master_ast:
        if cls_node.name not in seen_class_names:
            final_classes.append(cls_node)
            seen_class_names.add(cls_node.name)

    final_macros = []
    seen_macro_names = set() # Consider macro scope (class vs global) if macros can have same name in diff classes
    for macro_node in all_macros_for_master_ast:
        # Assuming macro names are unique globally for now, or qualified if class members
        # If macros can be part of classes, a more complex key is needed (e.g., classname_macroname)
        macro_key = macro_node.name 
        if hasattr(macro_node, 'class_name') and macro_node.class_name: # For class macros
            macro_key = f"{macro_node.class_name}_{macro_node.name}"

        if macro_key not in seen_macro_names:
            final_macros.append(macro_node)
            seen_macro_names.add(macro_key)
            
    master_ast = ProgramNode(classes=final_classes, includes=[], macros=final_macros)

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
        # import traceback # For debugging
        # traceback.print_exc() # For debugging
        return

    base_name = os.path.splitext(os.path.basename(initial_input_file_path))[0]
    output_s_file = f"{base_name}.s"
    
    if args.output:
        output_executable_file = args.output
    else:
        if args.assembly:
            output_executable_file = base_name 
        else:
            output_executable_file = "a.out"


    if args.assembly:
        s_output_target = args.output if args.output and args.output.endswith(".s") else output_s_file
        if args.output and not args.output.endswith(".s"):
             s_output_target = f"{os.path.splitext(args.output)[0]}.s"
             print(f"Warning: -S specified, output file '{args.output}' does not end with .s. Saving assembly to '{s_output_target}'")
        
        with open(s_output_target, 'w') as f:
            f.write(assembly_code)
        print(f"Assembly code saved to {s_output_target}")
        return

    temp_s_file = f"{base_name}_temp.s" 
    with open(temp_s_file, 'w') as f:
        f.write(assembly_code)

    final_exe_name = output_executable_file
    if args.output:
        final_exe_name = args.output
    else: 
        final_exe_name = base_name if os.path.exists(base_name) and os.path.isdir(base_name) else "a.out"
        if base_name == "a.out" : final_exe_name = "a.out.run" 

    try:
        # Add -lm flag if math library is used
        compile_command = ["gcc", "-no-pie", temp_s_file, "-o", final_exe_name]
        if uses_math_library:
            compile_command.append("-lm")
            
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

def handle_run_command(args):
    # First, compile the file
    # We need to temporarily modify args to ensure handle_compile_command
    # produces an executable and doesn't just print tokens or AST.
    original_assembly_arg = args.assembly if hasattr(args, 'assembly') else False
    original_ast_arg = args.ast if hasattr(args, 'ast') else False
    original_tokens_arg = args.tokens if hasattr(args, 'tokens') else False

    args.assembly = False # Ensure executable is produced
    args.ast = False      # Ensure executable is produced
    args.tokens = False   # Ensure executable is produced

    handle_compile_command(args)

    # Restore original args state if needed for future commands (though run is usually final)
    args.assembly = original_assembly_arg
    args.ast = original_ast_arg
    args.tokens = original_tokens_arg

    # Determine the output executable name based on handle_compile_command logic
    # This logic is replicated from lines 444-477 of handle_compile_command
    base_name = os.path.splitext(os.path.basename(os.path.abspath(args.input_file)))[0]
    if args.output:
        final_exe_name = args.output
    else:
        # This part of the logic is slightly simplified for the run command,
        # assuming we always want an executable.
        final_exe_name = base_name if os.path.exists(base_name) and os.path.isdir(base_name) else "a.out"
        if base_name == "a.out" : final_exe_name = "a.out.run"

    # Check if the executable was successfully created
    if not os.path.exists(final_exe_name):
        print(f"Error: Compilation failed. Could not find executable '{final_exe_name}'.")
        return

    # Run the compiled executable
    print(f"Running executable: ./{final_exe_name}")
    try:
        # Use subprocess.run to execute the compiled binary
        # capture_output=False allows the executable's output to go directly to the terminal
        subprocess.run([f"./{final_exe_name}"], check=True, capture_output=False)
    except FileNotFoundError:
        print(f"Error: Executable '{final_exe_name}' not found. Compilation might have failed or output path is incorrect.")
    except PermissionError:
        print(f"Error: Permission denied to execute '{final_exe_name}'. Make sure it's executable.")
    except subprocess.CalledProcessError as e:
        print(f"Execution failed with error code {e.returncode}")


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

    # Run command parser
    run_parser = subparsers.add_parser("run", help="Compile and run a Pangy source file (.pgy)", aliases=['r'])
    run_parser.add_argument("input_file", help="Pangy source file to compile and run (.pgy)")
    run_parser.add_argument("-o", "--output", help="Output file name for the executable (optional)") # Allow -o for run too
    
    args = arg_parser.parse_args()

    if args.command == "install":
        handle_install_command(args)
    elif args.command == "compile":
        if not args.input_file:
            compile_parser.error("argument input_file: Can't be empty when 'compile' command is used.")
            return
        handle_compile_command(args)
    elif args.command == "run":
        if not args.input_file:
             run_parser.error("argument input_file: Can't be empty when 'run' command is used.")
             return
        handle_run_command(args)
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
