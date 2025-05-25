from .parser_lexer import (
    ASTNode, ProgramNode, ClassNode, MethodNode, BlockNode, PrintNode, Token,
    ParamNode, IntegerLiteralNode, IdentifierNode, ThisNode,
    MethodCallNode, FunctionCallNode, BinaryOpNode, UnaryOpNode, ReturnNode, StringLiteralNode, IfNode, VarDeclNode, LoopNode, StopNode, AssignmentNode,
    MacroDefNode, MacroInvokeNode, ListLiteralNode, ArrayAccessNode,
    TT_LESS_THAN, TT_GREATER_THAN, TT_EQUAL, TT_NOT_EQUAL, TT_LESS_EQUAL, TT_GREATER_EQUAL,
    TT_PLUS, TT_MINUS, TT_STAR, TT_SLASH, TT_PERCENT,
    TT_PLUSPLUS, TT_MINUSMINUS,
    TT_AMPERSAND, TT_PIPE, TT_CARET, TT_TILDE,
    TT_LSHIFT, TT_RSHIFT, TT_URSHIFT
)
# import itertools # Not strictly needed for current logic
from types import SimpleNamespace # For getattr default

class CompilerError(Exception):
    pass

class Compiler:
    def __init__(self):
        self.assembly_code = "" # Final combined code
        self.data_section_code = "" # Accumulates .rodata content
        self.text_section_code = ""   # Accumulates .text content
        self.string_literals = {} 
        self.next_string_label_id = 0
        self.next_printf_format_label_id = 0
        self.next_if_label_id = 0 # For if/else labels
        self.next_loop_id = 0 # For loop labels
        self.next_list_header_label_id = 0 # For list static data
        self.externs = ["printf", "exit", "scanf", "atoi", "malloc", "fopen", "fclose", "fwrite", "fread", "fgets", "realloc", "memcpy", "strlen"] # Added realloc, memcpy, strlen
        
        self.current_method_params = {}
        self.current_method_locals = {} # For local variables
        self.current_method_stack_offset = 0 # Tracks current available stack slot relative to RBP for new locals
        self.current_method_total_stack_needed = 0 # For params + locals in prologue
        self.current_method_context = None
        # For loop support
        self.loop_end_labels = []
        # For macro support
        self.macros = {} # Maps macro names to MacroDefNode objects
        self.array_access_counter = 0

    def new_string_label(self, value):
        if value in self.string_literals:
            return self.string_literals[value]
        label = f".LC{self.next_string_label_id}"
        self.string_literals[value] = label
        self.next_string_label_id += 1
        # Escape special characters for assembly
        escaped_value = value.replace('\\', '\\\\')  # Must be first to avoid double-escaping
        escaped_value = escaped_value.replace('"', '\\"')
        escaped_value = escaped_value.replace('\n', '\\n')
        escaped_value = escaped_value.replace('\t', '\\t')
        escaped_value = escaped_value.replace('\r', '\\r')
        escaped_value = escaped_value.replace('\0', '\\0')
        escaped_value = escaped_value.replace('\b', '\\b')
        escaped_value = escaped_value.replace('\f', '\\f')
        escaped_value = escaped_value.replace('\v', '\\v')
        escaped_value = escaped_value.replace('\'', '\\\'')
        self.data_section_code += f'{label}:\n  .string "{escaped_value}"\n'
        return label

    def new_printf_format_label(self, fmt_string_value):
        label = f".LCPF{self.next_printf_format_label_id}"
        self.next_printf_format_label_id += 1
        
        # Escape special characters for assembly
        escaped_for_asciz = fmt_string_value.replace('\\', '\\\\')  # Must be first to avoid double-escaping
        escaped_for_asciz = escaped_for_asciz.replace('"', '\\"')
        escaped_for_asciz = escaped_for_asciz.replace('\n', '\\n')
        escaped_for_asciz = escaped_for_asciz.replace('\t', '\\t')
        escaped_for_asciz = escaped_for_asciz.replace('\r', '\\r')
        escaped_for_asciz = escaped_for_asciz.replace('\0', '\\0')
        escaped_for_asciz = escaped_for_asciz.replace('\b', '\\b')
        escaped_for_asciz = escaped_for_asciz.replace('\f', '\\f')
        escaped_for_asciz = escaped_for_asciz.replace('\v', '\\v')
        escaped_for_asciz = escaped_for_asciz.replace('\'', '\\\'')
        
        self.data_section_code += f'{label}:\n  .asciz "{escaped_for_asciz}"\n'
        return label

    def new_if_labels(self):
        idx = self.next_if_label_id
        self.next_if_label_id += 1
        return f".L_else_{idx}", f".L_end_if_{idx}"

    def new_list_header_label(self): # For potential static list data, though lists are dynamic
        idx = self.next_list_header_label_id
        self.next_list_header_label_id += 1
        return f".LH{idx}"

    def visit(self, node, context_override=None):
        effective_context = context_override if context_override is not None else self.current_method_context
        method_name = f"visit_{node.__class__.__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node, effective_context)

    def generic_visit(self, node, context):
        lineno_attr = getattr(node, 'lineno', None)
        if lineno_attr is None and hasattr(node, 'token'):
            lineno_attr = getattr(node.token, 'lineno', 'unknown')
        elif lineno_attr is None and hasattr(node, 'op_token'): # For BinaryOpNode potentially
             lineno_attr = getattr(node.op_token, 'lineno', 'unknown')
        elif lineno_attr is None:
            lineno_attr = 'unknown'
        raise CompilerError(f"No visit_{node.__class__.__name__} method for node {type(node)} (op: {getattr(node, 'op', 'N/A')}) at L{lineno_attr}")

    def visit_ProgramNode(self, node: ProgramNode, context=None):
        self.data_section_code = ".section .rodata\n"
        self.text_section_code = ".section .text\n"
        
        # Store the program node for reference by other visitors
        self.current_program_node = node
        
        # Declare externs for GAS
        temp_externs = list(self.externs) # Use a copy in case self.externs is modified
        # Ensure essential externs like malloc, realloc, memcpy, strlen are present
        for essential_extern in ["malloc", "realloc", "memcpy", "strlen", "printf", "exit"]:
            if essential_extern not in temp_externs:
                temp_externs.append(essential_extern)
        
        for ext_symbol in temp_externs:
            self.text_section_code += f".extern {ext_symbol}\n"
        self.text_section_code += "\n" # Blank line after externs

        self.string_literals = {}
        self.next_string_label_id = 0; self.next_printf_format_label_id = 0; self.next_if_label_id = 0; self.next_loop_id = 0; self.next_list_header_label_id = 0
        self.current_method_params = {}; self.current_method_stack_offset = 0
        self.current_method_total_stack_needed = 0; self.current_method_context = None
        
        # Register all macros (both global and class-level)
        self.macros = {}
        for macro_def in node.macros:
            # For class macros, use qualified name as the key
            if hasattr(macro_def, 'class_name') and macro_def.class_name:
                key = f"{macro_def.class_name}_{macro_def.name}"
                self.macros[key] = macro_def
            else:
                self.macros[macro_def.name] = macro_def
            # No code generation needed for macro definitions

        for class_node in node.classes:
            self.text_section_code += self.visit(class_node)
        
        self.assembly_code = (
            ".intel_syntax noprefix\n\n" + 
            self.data_section_code.strip() + "\n\n" + 
            self.text_section_code.strip() + "\n"
        )
        return self.assembly_code

    def visit_ClassNode(self, node: ClassNode, context=None):
        class_assembly = ""
        # Use safe class label (replace dots for nested classes)
        class_label = node.name.replace('.', '_')
        for method_node in node.methods:
            self.current_method_context = (class_label, method_node.name)
            class_assembly += self.visit(method_node)
        self.current_method_context = None
        return class_assembly

    def visit_MethodNode(self, node: MethodNode, context=None):
        class_name, method_name = self.current_method_context
        method_label = f"{class_name}_{method_name}"
        epilogue_label = f".L_epilogue_{class_name}_{method_name}"
        self.current_method_params = {} 
        self.current_method_locals = {}
        # Store method return type for later use in ReturnNode
        self.current_method_return_type = node.return_type
        method_text_assembly = ""
        
        # Determine if method is static (based on AST node flag)
        # Main.main is special: it's an entry point but we need to decide its static/instance nature for 'this'
        # For Math.main() from example, it's explicitly static.
        is_method_static = node.is_static
        is_entry_point_main = node.name == "main" and class_name == "Main" and not node.is_static # Default Main.main is instance

        # Check if this is main() with argc/argv parameters
        has_argc_argv = False
        if (is_entry_point_main or (node.is_static and node.name == "main" and class_name == "Main")) and len(node.params) == 2:
            # Check for main(argc int, argv string[]) pattern
            if (node.params[0].name == "argc" and node.params[0].type == "int" and
                node.params[1].name == "argv" and node.params[1].type == "string[]"):
                has_argc_argv = True

        if is_entry_point_main or (node.is_static and node.name == "main" and class_name == "Main"):
            # This covers instance Main.main and static Main.main if we ever allow that as entry
            method_text_assembly += f".global main\nmain:\n"
        else:
            method_text_assembly += f"{method_label}:\n"
        
        method_text_assembly += "  push rbp\n  mov rbp, rsp\n"
        
        local_stack_needed_for_params = 0
        current_rbp_offset_for_params = 0 
        
        # If method is static, params start at RDI. Otherwise (instance), RDI is 'this', params start at RSI.
        if is_method_static:
            explicit_param_arg_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
            method_text_assembly += f"  # Static method {class_name}.{method_name}. No 'this' pointer.\n"
        else: # Instance method
            explicit_param_arg_regs = ["rsi", "rdx", "rcx", "r8", "r9"]
            method_text_assembly += f"  # Instance method {class_name}.{method_name}. 'this' in RDI.\n"
        
        for i, p_node in enumerate(node.params):
            local_stack_needed_for_params += 8 
            current_rbp_offset_for_params -= 8
            param_data = {'offset_rbp': current_rbp_offset_for_params, 'type': p_node.type, 'index': i}
            if i < len(explicit_param_arg_regs):
                param_data['reg'] = explicit_param_arg_regs[i]
            self.current_method_params[p_node.name] = param_data

        # Helper function to recursively find all variable declarations in a block
        def find_var_declarations_in_block(block_node, variables_dict, current_rbp_offset):
            if not isinstance(block_node, BlockNode):
                return current_rbp_offset
            
            # First pass: collect all variable declarations
            for stmt in block_node.statements:
                if isinstance(stmt, VarDeclNode):
                    var_size = 8  # Assuming all variables (including list pointers) need 8 bytes
                    current_rbp_offset -= var_size
                    if stmt.name not in variables_dict:  # Only add if not already declared in outer scope
                        variables_dict[stmt.name] = {
                            'offset_rbp': current_rbp_offset, 
                            'type': stmt.type,
                            'size': var_size,
                            'is_initialized': False
                        }
                
                # Recursively search in nested blocks
                elif isinstance(stmt, IfNode):
                    current_rbp_offset = find_var_declarations_in_block(stmt.then_block, variables_dict, current_rbp_offset)
                    if stmt.else_block:
                        current_rbp_offset = find_var_declarations_in_block(stmt.else_block, variables_dict, current_rbp_offset)
                elif isinstance(stmt, LoopNode):
                    current_rbp_offset = find_var_declarations_in_block(stmt.body, variables_dict, current_rbp_offset)
            
            return current_rbp_offset

        # Start from the current offset after parameters
        current_rbp_offset_for_locals = current_rbp_offset_for_params
        current_rbp_offset_for_locals = find_var_declarations_in_block(node.body, self.current_method_locals, current_rbp_offset_for_locals)
        
        # Calculate total stack space needed
        local_stack_needed_for_vars = abs(current_rbp_offset_for_locals - current_rbp_offset_for_params)
        self.current_method_total_stack_needed = local_stack_needed_for_params + local_stack_needed_for_vars
        actual_stack_to_allocate = self.current_method_total_stack_needed
        
        # Ensure stack alignment (16 bytes)
        if actual_stack_to_allocate % 16 != 0:
            actual_stack_to_allocate = ((actual_stack_to_allocate // 16) + 1) * 16
        
        if actual_stack_to_allocate > 0:
            method_text_assembly += f"  sub rsp, {actual_stack_to_allocate} # Allocate stack ({actual_stack_to_allocate})\n"
        
        # Special handling for main with argc/argv
        if has_argc_argv:
            method_text_assembly += "  # Setting up argc and argv parameters from command line\n"
            # argc is already in RDI, argv is already in RSI
            argc_offset = self.current_method_params['argc']['offset_rbp']
            argv_offset = self.current_method_params['argv']['offset_rbp']
            
            # Store argc directly
            method_text_assembly += f"  mov QWORD PTR [rbp {argc_offset}], rdi # Store argc\n"
            
            # We need to convert the C-style argv (char**) to our string[] representation
            # 1. Create a new list to hold the argv strings
            method_text_assembly += "  # Convert C argv to Pangy string[]\n"
            method_text_assembly += "  mov r12, rdi # Save argc to r12\n"
            method_text_assembly += "  mov r13, rsi # Save argv to r13\n"
            
            # Allocate list with initial capacity = argc
            method_text_assembly += "  # Allocate list for argv with capacity = argc\n"
            method_text_assembly += "  imul rdi, 8  # Calculate size for elements (argc * 8)\n"
            method_text_assembly += "  add rdi, 16  # Add space for header (capacity and length)\n"
            method_text_assembly += "  call malloc  # Allocate memory for string[] list\n"
            method_text_assembly += "  mov r14, rax # Save list pointer to r14\n"
            
            # Initialize list header (capacity and length)
            method_text_assembly += "  # Initialize list header\n"
            method_text_assembly += "  mov QWORD PTR [r14], r12 # Set capacity = argc\n"
            method_text_assembly += "  mov QWORD PTR [r14 + 8], r12 # Set length = argc\n"
            
            # Loop through argv and copy pointers to our list
            method_text_assembly += "  # Copy argv strings to our list\n"
            method_text_assembly += "  xor rcx, rcx # Initialize counter\n"
            method_text_assembly += ".L_argv_loop_start:\n"
            method_text_assembly += "  cmp rcx, r12 # Compare counter with argc\n"
            method_text_assembly += "  je .L_argv_loop_end # Exit loop if done\n"
            
            # Get argv[i] (string pointer)
            method_text_assembly += "  mov rax, QWORD PTR [r13 + rcx*8] # Get argv[i]\n"
            
            # Store string pointer in our list
            method_text_assembly += "  mov QWORD PTR [r14 + 16 + rcx*8], rax # Store in our list\n"
            
            # Increment counter and continue loop
            method_text_assembly += "  inc rcx\n"
            method_text_assembly += "  jmp .L_argv_loop_start\n"
            method_text_assembly += ".L_argv_loop_end:\n"
            
            # Store our list pointer as the argv parameter
            method_text_assembly += f"  mov QWORD PTR [rbp {argv_offset}], r14 # Store argv list pointer\n"
        else:
            # Spill parameters from registers to their stack slots
            # (No 'this' to spill for static methods from RDI unless it was a param)
            for p_name, p_data in self.current_method_params.items():
                if 'reg' in p_data:
                    method_text_assembly += f"  mov QWORD PTR [rbp {p_data['offset_rbp']}], {p_data['reg']} # Spill param {p_name}\n"

        method_text_assembly += self.visit(node.body)
        method_text_assembly += f"{epilogue_label}:\n"

        # Add special handling for list/matrix return types
        if node.return_type.endswith('[]') or node.return_type.endswith('[][]'):
            method_text_assembly += f"  # Method returns {node.return_type}, pointer already in RAX\n"
        elif (is_entry_point_main or (node.is_static and node.name == "main")) and node.return_type == "void":
            method_text_assembly += "  mov rax, 0 # Default return 0 for main\n"

        method_text_assembly += "  mov rsp, rbp \n  pop rbp\n  ret\n\n"
        return method_text_assembly

    def visit_BlockNode(self, node: BlockNode, context):
        return "".join([self.visit(stmt, context) for stmt in node.statements])

    def visit_PrintNode(self, node: PrintNode, context):
        print_assembly = f"  # Print statement at L{node.lineno}\n"
        format_string_parts = []
        arg_expr_nodes = [] # For expressions that are not string literals themselves
        format_types = []  # Track format type for each argument

        for i, expr in enumerate(node.expressions):
            # Check for type conversion method calls (.as_string(), .as_int(), etc.)
            if isinstance(expr, MethodCallNode):
                # Handle methods like index(str, 0).as_string() or variable.as_string()
                if expr.method_name == "as_string":
                    format_string_parts.append("%c")  # Use %c for character display
                    arg_expr_nodes.append(expr.object_expr)
                    format_types.append("char")  # Mark as character format
                elif expr.method_name == "as_int":
                    format_string_parts.append("%d")  # Use %d for integer display
                    arg_expr_nodes.append(expr.object_expr)
                    format_types.append("int")
                else:
                    # Default handling for other method calls
                    node_type_for_print = self.get_expr_type(expr, context)
                    if node_type_for_print == 'string':
                        format_string_parts.append("%s")
                        format_types.append("string")
                    else:
                        format_string_parts.append("%d")
                        format_types.append("int")
                    arg_expr_nodes.append(expr)
            elif isinstance(expr, StringLiteralNode):
                format_string_parts.append(expr.value.replace("%", "%%"))
                # No need to add to arg_expr_nodes or format_types for string literals
            elif isinstance(expr, IntegerLiteralNode):
                format_string_parts.append("%d")
                arg_expr_nodes.append(expr)
                format_types.append("int")
            elif isinstance(expr, IdentifierNode):
                # Determine type of identifier to use %s or %d
                var_name = expr.value
                var_type = "unknown" # Default if not found, though it should be
                if var_name in self.current_method_locals:
                    var_type = self.current_method_locals[var_name]['type']
                elif var_name in self.current_method_params:
                    var_type = self.current_method_params[var_name]['type']
                
                if var_type == 'string':
                    format_string_parts.append("%s")
                    format_types.append("string")
                elif var_type == 'file':
                    # For file pointers, print as pointer value
                    format_string_parts.append("FILE*@%p")
                    format_types.append("pointer")
                # Add check for list types - printing a list variable directly prints its address (pointer)
                elif var_type.endswith("[]"):
                    format_string_parts.append("%p") # Print list pointer address
                    format_types.append("pointer")
                else: # Default to %d for int or unknown/other types for now
                    format_string_parts.append("%d")
                    format_types.append("int")
                arg_expr_nodes.append(expr)
            elif isinstance(expr, ArrayAccessNode):
                # Determine element type for format specifier using our new helper method
                element_type_for_print = self.get_array_element_type(expr.array_expr, context)
                
                if element_type_for_print == 'string':
                    format_string_parts.append("%s")
                    format_types.append("string")
                elif element_type_for_print == 'file': # If we ever have file[] and print file[i]
                    format_string_parts.append("FILE*@%p")
                    format_types.append("pointer")
                else: # Default to %d for int[], int, or other/unknown element types
                    format_string_parts.append("%d")
                    format_types.append("int")
                arg_expr_nodes.append(expr)
            elif isinstance(expr, FunctionCallNode):
                # Determine the function return type to use %s or %d
                func_name = expr.name
                if func_name == "input":  # input() returns string
                    format_string_parts.append("%s")
                    format_types.append("string")
                elif func_name == "to_int":  # to_int() returns int
                    format_string_parts.append("%d")
                    format_types.append("int")
                elif func_name == "to_string":  # to_string() returns string
                    format_string_parts.append("%s")
                    format_types.append("string")
                elif func_name == "open":  # open() returns file
                    format_string_parts.append("FILE*@%p")
                    format_types.append("pointer")
                elif func_name == "index":  # index() returns int (ASCII value)
                    format_string_parts.append("%d")
                    format_types.append("int")
                else:  # Default to %d for unknown functions
                    format_string_parts.append("%d")
                    format_types.append("int")
                arg_expr_nodes.append(expr)
            elif isinstance(expr, (BinaryOpNode, ThisNode)):
                # Check the expression type using our method
                node_type_for_print = self.get_expr_type(expr, context)
                if node_type_for_print == 'string':
                    format_string_parts.append("%s")
                    format_types.append("string")
                else:
                    format_string_parts.append("%d")
                    format_types.append("int")
                arg_expr_nodes.append(expr)
            elif isinstance(expr, MacroInvokeNode):
                # For macro invocations, we'll assume they'll resolve to integers for now
                format_string_parts.append("%d")
                format_types.append("int")
                arg_expr_nodes.append(expr)
            elif isinstance(expr, UnaryOpNode):
                # Unary operations like ~a always result in integers
                format_string_parts.append("%d")
                format_types.append("int")
                arg_expr_nodes.append(expr)
            else: 
                raise CompilerError(f"Cannot print expression of type {type(expr)} at L{node.lineno}")
        
        format_string = "".join(format_string_parts)
        format_string += "\n" # Add newline to all prints
        format_label = self.new_printf_format_label(format_string)
        printf_arg_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
        num_val_args = len(arg_expr_nodes)
        if num_val_args > (len(printf_arg_regs) -1) :
            for i in range(num_val_args - 1, (len(printf_arg_regs) -1) - 1, -1):
                print_assembly += self.visit(arg_expr_nodes[i], context)
                print_assembly += "  push rax # Push printf stack argument\n"
        for i in range(min(num_val_args, len(printf_arg_regs)-1) - 1, -1, -1):
            print_assembly += self.visit(arg_expr_nodes[i], context)
            target_reg = printf_arg_regs[i+1]
            
            # Special handling for char format (%c)
            if i < len(format_types) and format_types[i] == "char":
                # For %c, we need to ensure only the low byte is used
                print_assembly += f"  movzx {target_reg}, al # Load character (zero-extended) into {target_reg}\n"
            else:
                print_assembly += f"  mov {target_reg}, rax # Load printf arg into {target_reg}\n"
                
        print_assembly += f"  lea {printf_arg_regs[0]}, {format_label}[rip] # Format string for printf\n"
        print_assembly += "  mov rax, 0 # Num SSE registers for variadic call\n"
        print_assembly += "  call printf\n"
        if num_val_args > (len(printf_arg_regs) -1):
            bytes_pushed = (num_val_args - (len(printf_arg_regs) -1)) * 8
            print_assembly += f"  add rsp, {bytes_pushed} # Clean up printf stack arguments\n"
        return print_assembly

    def visit_ReturnNode(self, node: ReturnNode, context):
        class_name, method_name = context
        epilogue_label = f".L_epilogue_{class_name}_{method_name}"
        ret_assembly = f"  # Return statement at L{node.lineno}\n"
        
        # Get return type from current method context
        return_type = None
        for cls in getattr(self, 'current_program_node', ProgramNode([])).classes:
            if cls.name.replace('.', '_') == class_name:
                for method in cls.methods:
                    if method.name == method_name:
                        return_type = method.return_type
                        break
                if return_type:
                    break
        
        # Generate code to evaluate the expression
        if node.expression:
            ret_assembly += self.visit(node.expression, context)
            
            # If we're returning a list or matrix (any array type), we need to ensure
            # the pointer stays valid after returning from the function
            if return_type and (return_type.endswith('[]') or return_type.endswith('[][]')):
                ret_assembly += f"  # Returning {return_type} pointer in RAX\n"
        
        ret_assembly += f"  jmp {epilogue_label}\n"
        return ret_assembly

    def visit_IntegerLiteralNode(self, node: IntegerLiteralNode, context):
        return f"  mov rax, {node.value} # Integer literal {node.value}\n"
    def visit_StringLiteralNode(self, node: StringLiteralNode, context):
        label = self.new_string_label(node.value)
        return f"  lea rax, {label}[rip] # Address of string literal\n"
    def visit_IdentifierNode(self, node: IdentifierNode, context):
        var_name = node.value
        if var_name in self.current_method_params:
            param_info = self.current_method_params[var_name]
            offset = param_info['offset_rbp']
            return f"  mov rax, QWORD PTR [rbp {offset}] # Param {var_name} from [rbp{offset}]\n"
        elif var_name in self.current_method_locals:
            local_info = self.current_method_locals[var_name]
            # Assuming type checking/initialization is handled/guaranteed by VarDeclNode or semantic analysis.
            # If read-before-write was a concern at runtime for uninitialized vars, a check could be added.
            # For now, we trust it's initialized if it exists in locals map and is being read.
            offset = local_info['offset_rbp']
            return f"  mov rax, QWORD PTR [rbp {offset}] # Local var {var_name} from [rbp{offset}]\n"
        # Add check for 'this' as an identifier if we ever support it as a standalone variable rather than keyword
        # else if var_name == 'this': # This would be if 'this' was a normal variable
        #     return self.visit_ThisNode(ThisNode(node.token), context) # Hacky way to reuse ThisNode logic

        raise CompilerError(f"Undefined identifier '{var_name}' at L{node.token.lineno}. Only params and declared local vars supported.")

    def visit_ThisNode(self, node: ThisNode, context):
        class_name, method_name = context
        # To correctly use 'this', we must be in an instance method.
        # Need to find the MethodNode in AST to check is_static. This is complex here.
        # For now, assume if ThisNode is used, it implies an instance context.
        # The check for 'this' in static main was removed, but a general check for 'this' in ANY static method is needed.
        # This check should ideally be in the parser or a semantic analysis phase.
        # If current method is static, `mov rax, rdi` is wrong as RDI is not 'this'.
        # This implies compiler needs to know if current_method_context is static.
        # Let's assume self.current_method_node is set by visit_ClassNode/visit_MethodNode
        # if hasattr(self, 'current_method_node_is_static') and self.current_method_node_is_static:
        #     raise CompilerError(f"'this' keyword cannot be used in a static method '{method_name}'. L{node.token.lineno}")
        return "  mov rax, rdi # 'this' is in rdi for instance methods\n"

    def visit_VarDeclNode(self, node: VarDeclNode, context):
        var_name = node.name
        # The stack space and offset for this variable should have been pre-calculated 
        # and stored in self.current_method_locals by visit_MethodNode.
        # The type string (e.g., "int", "int[]") is in local_info['type']
        if var_name not in self.current_method_locals:
            raise CompilerError(f"Internal Compiler Error: Local variable '{var_name}' was not pre-allocated space. L{node.lineno}")

        local_info = self.current_method_locals[var_name]
        var_offset_rbp = local_info['offset_rbp']
        var_type_str = local_info['type'] # e.g., "int", "int[]"

        decl_assembly = f"  # Variable Declaration: {var_name} ({var_type_str}) at L{node.lineno}\n"
        
        # 1. Evaluate the right-hand side expression (initialization value)
        #    This will handle ListLiteralNode if used for initialization.
        decl_assembly += self.visit(node.expression, context) # Result will be in RAX
        
        # 2. Store the result from RAX into the variable's pre-allocated stack slot
        decl_assembly += f"  mov QWORD PTR [rbp {var_offset_rbp}], rax # Initialize {var_name} = RAX (pointer for lists) at [rbp{var_offset_rbp}]\n"
        
        self.current_method_locals[var_name]['is_initialized'] = True # Mark as initialized
        return decl_assembly

    def visit_MethodCallNode(self, node: MethodCallNode, context):
        # Special handling for .as_string() and .as_int() methods
        if node.method_name in ["as_string", "as_int"]:
            # These are just type annotations for print statements, not actual method calls
            # Just evaluate the object expression and return its value
            return self.visit(node.object_expr, context)
            
        current_class_name, current_method_name = context 
        call_assembly = f"  # Method Call to {node.method_name} on {type(node.object_expr)} at L{node.method_name_token.lineno}\n"
        target_method_label = ""
        
        is_call_on_this = isinstance(node.object_expr, ThisNode)
        is_static_call = False
        is_new_call = False # Flag for ClassName.new()
        object_address_in_rdi_setup = ""

        if is_call_on_this:
            target_method_label = f"{current_class_name}_{node.method_name}"
            object_address_in_rdi_setup = "  # 'this' (current object) is already in RDI for instance call\n"
        elif isinstance(node.object_expr, IdentifierNode):
            obj_name_or_class_name = node.object_expr.value
            # For nested classes, create a safe name for labels
            safe_name = obj_name_or_class_name.replace('.', '_')
            
            if node.method_name == "new":
                is_new_call = True
                # ClassName.new() call
                class_name_for_new = obj_name_or_class_name
                call_assembly += f"  # Allocation for new {class_name_for_new} object\n"
                # TODO: Determine actual size based on class fields later.
                # For now, assume a minimal fixed size (e.g., 8 bytes for a pointer or placeholder).
                object_size = 8 
                call_assembly += f"  mov rdi, {object_size}  # Size for malloc\n"
                call_assembly += "  call malloc          # Allocate memory, result in RAX\n"
                # Optional: call an initializer/constructor ClassName_ClassName if it exists
                # call_assembly += f"  mov rdi, rax         # Pass new object ptr to constructor in RDI\n"
                # call_assembly += f"  call {class_name_for_new}_{class_name_for_new} # Call constructor {ClassName}_init or {ClassName}_{ClassName}\n"
                return call_assembly # RAX contains the new object pointer, this is the result of .new()
            
            # Not a .new() call, so it's obj.method() or Class.static_method()
            var_type = None
            is_obj_var_known = False
            if obj_name_or_class_name in self.current_method_locals:
                var_type = self.current_method_locals[obj_name_or_class_name]['type']
                is_obj_var_known = True
            elif obj_name_or_class_name in self.current_method_params:
                var_type = self.current_method_params[obj_name_or_class_name]['type']
                is_obj_var_known = True

            if is_obj_var_known and var_type not in ['int', 'string']:
                # Instance call on an object variable: math.add()
                # var_type is the ClassName of the object (e.g., "Math" or nested class)
                is_static_call = False  # It's an instance call
                # Use safe class type for nested classes
                safe_var_type = var_type.replace('.', '_')
                target_method_label = f"{safe_var_type}_{node.method_name}" 
                object_address_in_rdi_setup = self.visit(node.object_expr, context) # Get var's address in RAX
                object_address_in_rdi_setup += "  mov rdi, rax # Load object address into RDI for the call\n"
            else:
                # Assume it's a class name for a static call: Math.main() or other Class.static_method()
                # Or, it could be an error if obj_name_or_class_name is an int/string var used like an object.
                # For now, this path leads to a static call interpretation.
                is_static_call = True
                # Use safe name for static nested classes
                target_method_label = f"{safe_name}_{node.method_name}" 
                object_address_in_rdi_setup = f"  # Static call to {obj_name_or_class_name}.{node.method_name}. No 'this' in RDI.\n"
        else:
            raise CompilerError(f"Unsupported object expression for method call: {type(node.object_expr)} L{node.method_name_token.lineno}")

        # If is_new_call was true, we would have returned already.
        # Argument registers depend on whether it's a static or instance call
        if is_static_call:
            explicit_arg_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
        else: # Instance call (either on 'this' or on an object variable)
            explicit_arg_regs = ["rsi", "rdx", "rcx", "r8", "r9"]
        
        num_explicit_args_to_pass = len(node.arguments)
        stack_arg_count = 0
        if num_explicit_args_to_pass > len(explicit_arg_regs):
            stack_arg_count = num_explicit_args_to_pass - len(explicit_arg_regs)

        # Evaluate arguments & push to stack (RTL for registers, then RTL for stack args)
        # Push stack arguments (if any), from right to left
        for i in range(num_explicit_args_to_pass - 1, len(explicit_arg_regs) - 1, -1):
            call_assembly += self.visit(node.arguments[i], context) 
            call_assembly += "  push rax # Push stack argument\n"

        # Evaluate register arguments and push them temporarily to stack (right to left)
        temp_pushes_for_reg_args = 0
        for i in range(min(num_explicit_args_to_pass, len(explicit_arg_regs)) - 1, -1, -1):
            call_assembly += self.visit(node.arguments[i], context) 
            call_assembly += "  push rax # Temporarily store reg argument\n"
            temp_pushes_for_reg_args += 1

        # Setup RDI if it's an instance call on an object variable (already handled for 'this' calls conceptually)
        # For static calls, RDI is the first explicit argument, so no special setup here.
        if not is_static_call and not is_call_on_this:
             call_assembly += object_address_in_rdi_setup # This loads object address into RDI
        elif is_call_on_this:
             call_assembly += object_address_in_rdi_setup # This is just a comment for 'this' calls
        # For static calls, RDI will be populated by the first argument pop if any, or is not used if no args.

        # Pop arguments into their designated registers.
        for i in range(min(num_explicit_args_to_pass, len(explicit_arg_regs))):
            target_reg = explicit_arg_regs[i]
            call_assembly += f"  pop {target_reg} # Load arg into {target_reg}\n"
        
        # If it's a static call and RDI was not used by an argument, it remains as is (not 'this')
        # If it's an instance call on 'this', RDI was already 'this'.
        # If it's an instance call on object var, RDI was set by object_address_in_rdi_setup.

        alignment_padding = 0
        if stack_arg_count > 0 and (stack_arg_count % 2 != 0):
            call_assembly += "  sub rsp, 8 # Align stack for odd stack args\n"
            alignment_padding = 8

        call_assembly += f"  call {target_method_label}\n"

        if alignment_padding > 0:
            call_assembly += f"  add rsp, {alignment_padding} # Restore stack alignment\n"

        if stack_arg_count > 0:
            call_assembly += f"  add rsp, {stack_arg_count * 8} # Clean up stack arguments\n"
            
        return call_assembly

    def visit_FunctionCallNode(self, node: FunctionCallNode, context):
        # Built-in free function call support (e.g., exit(code))
        if node.name == "exit":
            # exit expects one argument
            if len(node.arguments) != 1:
                raise CompilerError(f"exit() expects exactly one argument, got {len(node.arguments)} L{node.name_token.lineno}")
            code = f"  # exit({node.arguments[0]}) call at L{node.name_token.lineno}\n"
            # Evaluate argument to RAX
            code += self.visit(node.arguments[0], context)
            code += "  mov rdi, rax # exit code\n"
            code += "  call exit\n"
            return code
        elif node.name == "length":
            if len(node.arguments) != 1:
                raise CompilerError(f"length() expects one argument (list or string), got {len(node.arguments)} at L{node.name_token.lineno}")
            
            # Add a unique ID to avoid duplicate labels
            if not hasattr(self, 'next_length_label_id'):
                self.next_length_label_id = 0
            length_label_id = self.next_length_label_id
            self.next_length_label_id += 1
            
            code = f"  # length(list|string) call at L{node.name_token.lineno}\n"
            code += self.visit(node.arguments[0], context) # List/string pointer in RAX
            
            # Determine if the argument is a string or a list by checking its type
            arg_type = self.get_expr_type(node.arguments[0], context)
            
            code += "  mov rdi, rax # Pointer for length check\n"
            code += "  cmp rdi, 0 # Check for null pointer \n"
            code += f"  jne .L_length_not_null_{node.name_token.lineno}_{length_label_id}\n"
            code += "  mov rax, 0 # Length of null is 0\n"
            code += f"  jmp .L_length_end_{node.name_token.lineno}_{length_label_id}\n"
            code += f".L_length_not_null_{node.name_token.lineno}_{length_label_id}:\n"
            
            if arg_type == "string":
                # For strings, use strlen
                if "strlen" not in self.externs:
                    self.externs.append("strlen")
                code += "  call strlen # Get string length\n"
            else:
                # For lists, get length from list header
                code += "  mov rax, QWORD PTR [rdi + 8] # Load length from list_ptr + 8 bytes offset\n"
                
            code += f".L_length_end_{node.name_token.lineno}_{length_label_id}:\n"
            return code
        elif node.name == "index":
            if len(node.arguments) != 2:
                raise CompilerError(f"index() expects two arguments (string, position), got {len(node.arguments)} at L{node.name_token.lineno}")
            
            # Add a unique ID to avoid duplicate labels
            if not hasattr(self, 'next_index_label_id'):
                self.next_index_label_id = 0
            index_label_id = self.next_index_label_id
            self.next_index_label_id += 1
            
            code = f"  # index(string, position) call at L{node.name_token.lineno}\n"
            
            # Get the string pointer
            code += self.visit(node.arguments[0], context) # String pointer in RAX
            code += "  mov r12, rax # Save string pointer to r12\n"
            
            # Get the index position
            code += self.visit(node.arguments[1], context) # Index in RAX
            code += "  mov r13, rax # Save index to r13\n"
            
            # Check for null string
            code += "  cmp r12, 0 # Check for null string\n"
            code += f"  jne .L_index_not_null_{node.name_token.lineno}_{index_label_id}\n"
            
            # Handle null string error
            error_msg = self.new_string_label("Error: Cannot index a null string.\n")
            code += f"  lea rdi, {error_msg}[rip]\n"
            code += "  call printf\n"
            code += "  mov rdi, 1\n"
            code += "  call exit\n"
            
            code += f".L_index_not_null_{node.name_token.lineno}_{index_label_id}:\n"
            
            # Get string length using strlen for bounds check
            if "strlen" not in self.externs:
                self.externs.append("strlen")
            code += "  mov rdi, r12 # String pointer for strlen\n"
            code += "  call strlen # Get string length\n"
            code += "  mov r14, rax # Save string length to r14\n"
            
            # Bounds check
            code += "  cmp r13, 0 # Check if index < 0\n"
            code += f"  jl .L_index_out_of_bounds_{node.name_token.lineno}_{index_label_id}\n"
            code += "  cmp r13, r14 # Check if index >= length\n"
            code += f"  jge .L_index_out_of_bounds_{node.name_token.lineno}_{index_label_id}\n"
            code += f"  jmp .L_index_in_bounds_{node.name_token.lineno}_{index_label_id}\n"
            
            # Handle out of bounds error
            code += f".L_index_out_of_bounds_{node.name_token.lineno}_{index_label_id}:\n"
            error_msg = self.new_string_label("Error: String index out of bounds.\n")
            code += f"  lea rdi, {error_msg}[rip]\n"
            code += "  call printf\n"
            code += "  mov rdi, 1\n"
            code += "  call exit\n"
            
            # Get character at index
            code += f".L_index_in_bounds_{node.name_token.lineno}_{index_label_id}:\n"
            code += "  mov rax, r12 # String pointer\n"
            code += "  add rax, r13 # Add index to get character address\n"
            code += "  movzx rax, BYTE PTR [rax] # Load character (zero-extended to 64 bits)\n"
            
            return code
        elif node.name == "append":
            if len(node.arguments) != 2:
                raise CompilerError(f"append() expects two arguments (list, value), got {len(node.arguments)} at L{node.name_token.lineno}")

            code = f"  # append(list, value) call at L{node.name_token.lineno}\n"
            # Evaluate list pointer (arg 0)
            code += self.visit(node.arguments[0], context) # List pointer in RAX
            code += "  push rax # Save list_ptr on stack (as it might change due to realloc)\n"
            # Evaluate value (arg 1)
            code += self.visit(node.arguments[1], context) # Value in RAX
            code += "  push rax # Save value on stack\n"

            code += "  # --- APPEND IMPLEMENTATION ---\n"
            code += "  pop rsi    # Value to append is now in RSI\n"
            code += "  pop rdi    # List pointer is now in RDI\n"

            # Check if list_ptr (RDI) is null (meaning uninitialized list)
            null_list_label = f".L_append_null_list_{node.name_token.lineno}"
            not_null_list_label = f".L_append_not_null_list_{node.name_token.lineno}"
            code += "  cmp rdi, 0\n"
            code += f"  je {null_list_label}\n"

            # List is not null, proceed with normal append logic
            code += f"{not_null_list_label}:\n"
            code += "  mov r12, QWORD PTR [rdi]     # r12 = capacity = list_ptr[0]\n"
            code += "  mov r13, QWORD PTR [rdi + 8] # r13 = length   = list_ptr[1]\n"
            code += "  cmp r13, r12                 # if length >= capacity\n"
            code += f"  jl .L_append_has_space_{node.name_token.lineno}\n"

            code += "  # No space, reallocate\n"
            code += "  mov rax, r12                 # current capacity in RAX\n"
            code += "  test rax, rax                # Check if capacity is 0\n"
            code += f"  jnz .L_append_double_cap_{node.name_token.lineno}\n"
            code += "  mov r12, 8                   # If capacity was 0, set new capacity to 8 (e.g.)\n"
            code += f"  jmp .L_append_set_new_cap_{node.name_token.lineno}\n"
            code += f".L_append_double_cap_{node.name_token.lineno}:\n"
            code += "  shl r12, 1                   # new_capacity = capacity * 2\n"
            code += f".L_append_set_new_cap_{node.name_token.lineno}:\n"
            # r12 now holds new_capacity

            code += "  push rsi                     # Save value (RSI) before realloc\n"
            code += "  mov rsi, r12                 # rsi = new_capacity (for element storage)\n"
            code += "  imul rsi, 8                  # rsi = new_capacity * 8 (bytes for elements)\n"
            code += "  add rsi, 16                  # rsi = total new size (header + elements)\n"
            # RDI still holds old list_ptr for realloc
            code += "  call realloc                 # rax = realloc(old_list_ptr, total_new_size)\n"
            code += "  pop rsi                      # Restore value (RSI)\n"
            code += "  mov rdi, rax                 # Update list_ptr with realloc result\n"
            code += "  mov QWORD PTR [rdi], r12     # list_ptr[0] = new_capacity\n"
            # R13 (length) is still correct

            code += f".L_append_has_space_{node.name_token.lineno}:\n"
            code += "  mov rax, r13                 # RAX = current length (index to insert at)\n"
            code += "  imul rax, 8                  # offset for element = length * 8\n"
            code += "  add rax, 16                  # total offset = header_size + element_offset\n"
            code += "  mov QWORD PTR [rdi + rax], rsi # list_ptr[length_offset] = value\n"
            code += "  inc r13                      # length++\n"
            code += "  mov QWORD PTR [rdi + 8], r13 # list_ptr[1] = new_length\n"
            # Update the original variable holding the list pointer if it changed
            # The list pointer (RDI) might have changed. We pushed the *original* list_ptr variable's stack address earlier.
            # No, we pushed the value of the list_ptr itself. The caller must handle if the list_ptr var needs update
            # For `append(myList, x)`, if myList is a stack variable, its value (the pointer) needs to be updated
            # This is tricky. For now, append will modify the list in place. If realloc moves it,
            # the original pointer variable on stack becomes stale.
            # A common C pattern is `myList = append(myList, x);` where append returns the new list_ptr.
            # Let's make append return the (potentially new) list pointer in RAX.

            # The list pointer that was on the stack at [rbp + offset] needs to be updated with the new RDI
            # This requires knowing which variable it was.
            # Let's assume for now that the user handles `myList = append(myList, x)` if using this low-level append.
            # For `append(myList, x)` as a statement, the original `myList` variable on stack needs update if realloc happens.

            # Simplification: append() will return the (potentially new) list_ptr in RAX
            code += "  mov rax, rdi                 # Return new list_ptr in RAX\n"
            code += f"  jmp .L_append_end_{node.name_token.lineno}\n"

            # Handle null list case: allocate initial list
            code += f"{null_list_label}:\n"
            code += "  mov r12, 8                   # Initial capacity = 8\n"
            code += "  mov r13, 0                   # Initial length = 0\n"
            code += "  push rsi                     # Save value (RSI) before malloc\n"
            code += "  mov rdi, r12                 # rdi = capacity (for element storage)\n"
            code += "  imul rdi, 8                  # rdi = capacity * 8 (bytes for elements)\n"
            code += "  add rdi, 16                  # rdi = total size for malloc (header + elements)\n"
            code += "  call malloc                  # rax = new_list_ptr\n"
            code += "  pop rsi                      # Restore value (RSI)\n"
            code += "  mov rdi, rax                 # list_ptr = new_list_ptr\n"
            code += "  mov QWORD PTR [rdi], r12     # list_ptr[0] = capacity\n"
            code += "  mov QWORD PTR [rdi + 8], r13 # list_ptr[1] = length (still 0)\n"
            # Now jump to the part that adds the element (which expects length in r13)
            code += f"  jmp .L_append_has_space_{node.name_token.lineno} # This will add the first element\n"

            code += f".L_append_end_{node.name_token.lineno}:\n"
            # Result (new list pointer) is in RAX.
            # The caller who has the list variable (e.g. `myList`) must update it with RAX.
            # If `append` is used as a statement, `myList = append(myList, val)` is implied.
            # For `var myList = ...; append(myList, val);`, we need to find `myList` on stack and update.
            # This is complex. Let's assume `append` returns the new pointer and the user assigns it.
            # If called as `append(list,val)` as a statement, the original variable is NOT updated by this code.
            return code

        elif node.name == "pop":
            if len(node.arguments) != 1:
                raise CompilerError(f"pop() expects exactly one argument (list), got {len(node.arguments)} at L{node.name_token.lineno}")
            
            code = f"  # pop(list) call at L{node.name_token.lineno}\n"
            # Evaluate list pointer
            code += self.visit(node.arguments[0], context) # List pointer in RAX
            code += "  mov rdi, rax # List pointer for pop operation\n"
            
            # Check for null list
            code += "  cmp rdi, 0 # Check for null list pointer\n"
            code += f"  jne .L_pop_not_null_{node.name_token.lineno}\n"
            
            # Handle null list error
            error_msg = self.new_string_label("Error: Cannot pop from a null list.\n")
            code += f"  lea rdi, {error_msg}[rip]\n"
            code += "  call printf\n"
            code += "  mov rdi, 1\n"
            code += "  call exit\n"
            
            code += f".L_pop_not_null_{node.name_token.lineno}:\n"
            
            # Check if list is empty
            code += "  mov r12, QWORD PTR [rdi + 8] # r12 = length\n"
            code += "  cmp r12, 0 # Check if list is empty\n"
            code += f"  jne .L_pop_not_empty_{node.name_token.lineno}\n"
            
            # Handle empty list error
            error_msg = self.new_string_label("Error: Cannot pop from an empty list.\n")
            code += f"  lea rdi, {error_msg}[rip]\n"
            code += "  call printf\n"
            code += "  mov rdi, 1\n"
            code += "  call exit\n"
            
            code += f".L_pop_not_empty_{node.name_token.lineno}:\n"
            
            # Get last element's value
            code += "  dec r12 # r12 = length - 1 (last element index)\n"
            code += "  imul r13, r12, 8 # r13 = (length-1) * 8 (byte offset)\n"
            code += "  add r13, 16 # r13 = header_size + element_offset\n"
            code += "  mov rax, QWORD PTR [rdi + r13] # rax = list[length-1]\n"
            
            # Save return value temporarily
            code += "  push rax # Save the value to be returned\n"
            
            # Update length
            code += "  mov QWORD PTR [rdi + 8], r12 # Update length = length - 1\n"
            
            # Restore return value
            code += "  pop rax # Restore the popped value to return\n"
            
            return code

        elif node.name == "open":
            # open(filename, mode) - Opens a file and returns a file pointer
            if len(node.arguments) != 2:
                raise CompilerError(f"open() expects exactly two arguments (filename, mode), got {len(node.arguments)} L{node.name_token.lineno}")
            
            code = f"  # open() call at L{node.name_token.lineno}\n"
            
            # Get the filename in RDI (first arg to fopen)
            code += self.visit(node.arguments[0], context)
            code += "  mov rdi, rax # Filename (1st fopen arg)\n"
            
            # Get the mode in RSI (second arg to fopen)
            code += self.visit(node.arguments[1], context)
            code += "  mov rsi, rax # Mode (2nd fopen arg)\n"
            
            # Call fopen
            code += "  call fopen # Open file\n"
            
            # Result file pointer is in RAX
            
            # Check for NULL return (error)
            code += "  test rax, rax # Check if file was opened successfully\n"
            code += "  jnz .Lopen_success_{0}\n".format(node.name_token.lineno)
            
            # Handle error without any complicated stack manipulations
            err_msg = "Error: Failed to open file\n"
            err_label = self.new_string_label(err_msg)
            code += f"  lea rdi, {err_label}[rip] # Error message\n"
            code += "  call printf # Print error message\n"
            code += "  mov rdi, 1 # Exit code 1 for error\n"
            code += "  call exit # Exit program\n"
            
            code += ".Lopen_success_{0}:\n".format(node.name_token.lineno)
            
            # Result is in RAX (the FILE* pointer)
            return code
            
        elif node.name == "close":
            # close(file_pointer) - Closes a file
            if len(node.arguments) != 1:
                raise CompilerError(f"close() expects exactly one argument (file pointer), got {len(node.arguments)} L{node.name_token.lineno}")
            
            code = f"  # close() call at L{node.name_token.lineno}\n"
            
            # Get file pointer in RDI (arg to fclose)
            code += self.visit(node.arguments[0], context)
            code += "  mov rdi, rax # File pointer for fclose\n"
            
            # Call fclose
            code += "  call fclose # Close file\n"
            
            # Result (success/failure) is in RAX
            return code
            
        elif node.name == "write":
            # write(file_pointer, content) - Writes content to a file
            if len(node.arguments) != 2:
                raise CompilerError(f"write() expects exactly two arguments (file pointer, content), got {len(node.arguments)} L{node.name_token.lineno}")
            
            code = f"  # write() call at L{node.name_token.lineno}\n"
            
            # Get file pointer in RCX (4th arg to fwrite)
            code += self.visit(node.arguments[0], context)
            code += "  mov r12, rax # Save file pointer in R12\n"
            
            # Get content in RSI (2nd arg to fwrite)
            code += self.visit(node.arguments[1], context)
            code += "  mov r13, rax # Save content pointer in R13\n"
            
            # Get string length using strlen
            code += "  mov rdi, rax # Content for strlen\n"
            if "strlen" not in self.externs:
                self.externs.append("strlen")
            code += "  call strlen # Get string length\n"
            
            # Set up arguments for fwrite: fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream)
            code += "  mov rdi, r13 # Content pointer (1st arg)\n"
            code += "  mov rsi, 1   # Size of elements (2nd arg)\n"
            code += "  mov rdx, rax # Count (3rd arg) - string length\n"
            code += "  mov rcx, r12 # File pointer (4th arg)\n"
            
            # Call fwrite
            code += "  call fwrite # Write to file\n"
            
            # Result is in RAX (number of items written)
            return code
            
        elif node.name == "read":
            # read(file_pointer) - Reads entire file content into a string
            if len(node.arguments) != 1:
                raise CompilerError(f"read() expects exactly one argument (file pointer), got {len(node.arguments)} L{node.name_token.lineno}")
            
            code = f"  # read() call at L{node.name_token.lineno}\n"
            
            # Get file pointer
            code += self.visit(node.arguments[0], context)
            code += "  mov r12, rax # Save file pointer in R12\n"
            
            # Move to end of file to get file size
            if "fseek" not in self.externs:
                self.externs.append("fseek")
            code += "  mov rdi, r12 # File pointer\n"
            code += "  mov rsi, 0   # Offset\n"
            code += "  mov rdx, 2   # SEEK_END (2)\n"
            code += "  call fseek   # Seek to end\n"
            
            # Get file size with ftell
            if "ftell" not in self.externs:
                self.externs.append("ftell")
            code += "  mov rdi, r12 # File pointer\n"
            code += "  call ftell   # Get file size\n"
            code += "  mov r13, rax # Save file size in R13\n"
            
            # Rewind to beginning of file
            code += "  mov rdi, r12 # File pointer\n"
            code += "  mov rsi, 0   # Offset\n"
            code += "  mov rdx, 0   # SEEK_SET (0)\n"
            code += "  call fseek   # Rewind file\n"
            
            # Allocate buffer (file size + 1 for null terminator)
            code += "  lea rdi, [r13 + 1] # Buffer size = file size + 1\n"
            code += "  call malloc # Allocate buffer\n"
            code += "  mov r14, rax # Save buffer pointer in R14\n"
            
            # Read entire file with fread(buffer, 1, size, file)
            if "fread" not in self.externs:
                self.externs.append("fread")
            code += "  mov rdi, r14 # Buffer\n"
            code += "  mov rsi, 1   # Size of each element\n"
            code += "  mov rdx, r13 # Number of elements (file size)\n"
            code += "  mov rcx, r12 # File pointer\n"
            code += "  call fread   # Read file\n"
            
            # Add null terminator
            code += "  mov byte ptr [r14 + r13], 0 # Add null terminator\n"
            
            # Return buffer
            code += "  mov rax, r14 # Return buffer pointer\n"
            
            return code

        elif node.name == "input":
            # input() function - prompts the user and returns a string
            if len(node.arguments) != 1:
                raise CompilerError(f"input() expects exactly one argument (prompt string), got {len(node.arguments)} L{node.name_token.lineno}")
            
            code = f"  # input() call at L{node.name_token.lineno}\n"
            
            # First, print the prompt using printf
            prompt_expr = node.arguments[0]
            code += self.visit(prompt_expr, context)  # Get prompt string in RAX
            code += "  mov rdi, rax # Prompt string for printf\n"
            code += "  mov rax, 0 # No SSE registers for variadic call\n"
            code += "  call printf # Print the prompt\n"
            
            # Flush stdout to ensure prompt is displayed before input
            code += "  mov rdi, 0 # flush stdout\n"
            code += "  call fflush\n"
            
            # Add fflush to externs if not already there
            if "fflush" not in self.externs:
                self.externs.append("fflush")
            
            # Allocate buffer for input on heap (256 bytes)
            buffer_size = 256
            code += f"  mov rdi, {buffer_size} # Size for input buffer malloc\n"
            code += "  call malloc # Allocate buffer for input\n"
            
            # Use fgets for safer input
            if "fgets" not in self.externs:
                self.externs.append("fgets")
            if "stdin" not in self.externs:
                self.externs.append("stdin")
            
            # Setup fgets arguments: fgets(buffer, size, stdin)
            code += "  mov rdi, rax # Buffer for fgets (1st arg)\n"
            code += f"  mov rsi, {buffer_size} # Size parameter (2nd arg)\n"
            code += "  mov rdx, stdin # stdin FILE* (3rd arg)\n"
            code += "  push rdi # Save buffer pointer\n"
            code += "  call fgets # Read line into buffer\n"
            code += "  pop rax # Restore buffer pointer to RAX\n"
            
            # Remove trailing newline if present
            label_end = f".L_input_end_{node.name_token.lineno}"
            code += "  push rax # Save buffer pointer\n"
            code += "  mov rdi, rax # Buffer for strlen\n"
            
            # Add strlen to externs if not already there
            if "strlen" not in self.externs:
                self.externs.append("strlen")
                
            code += "  call strlen # Get string length\n"
            code += "  pop rdi # Restore buffer pointer to RDI\n"
            code += "  test rax, rax # Check if length is 0\n"
            code += f"  jz {label_end} # Skip if empty string\n"
            code += "  lea rcx, [rdi + rax - 1] # Pointer to last char\n"
            code += "  cmp byte ptr [rcx], 10 # Check if it's newline ('\\n')\n"
            code += f"  jne {label_end} # Skip if not newline\n"
            code += "  mov byte ptr [rcx], 0 # Replace newline with null terminator\n"
            code += f"{label_end}:\n"
            code += "  mov rax, rdi # Return buffer pointer in RAX\n"
            
            return code

        elif node.name == "to_int":
            # to_int() function - converts a string to integer
            if len(node.arguments) != 1:
                raise CompilerError(f"to_int() expects exactly one argument (string), got {len(node.arguments)} L{node.name_token.lineno}")
            
            code = f"  # to_int() call at L{node.name_token.lineno}\n"
            
            # Get the string in RAX
            code += self.visit(node.arguments[0], context)
            code += "  test rax, rax # Check if string pointer is null\n"
            code += "  jnz .Lto_int_valid_ptr_{0}\n".format(node.name_token.lineno)
            code += "  mov rax, 0 # Return 0 for null string\n"
            code += "  jmp .Lto_int_done_{0}\n".format(node.name_token.lineno)
            code += ".Lto_int_valid_ptr_{0}:\n".format(node.name_token.lineno)
            code += "  mov rdi, rax # String to convert\n"
            code += "  call atoi # Convert string to integer\n"
            code += ".Lto_int_done_{0}:\n".format(node.name_token.lineno)
            # Result is in RAX
            return code

        elif node.name == "to_string":
            # to_string() function - converts an integer to string
            if len(node.arguments) != 1:
                raise CompilerError(f"to_string() expects exactly one argument (int), got {len(node.arguments)} L{node.name_token.lineno}")
            
            code = f"  # to_string() call at L{node.name_token.lineno}\n"
            
            # Get the integer in RAX
            code += self.visit(node.arguments[0], context)
            
            # Use snprintf to safely convert integer to string
            if "snprintf" not in self.externs:
                self.externs.append("snprintf")
                
            # Allocate buffer for the result
            code += "  mov r12, rax # Save the integer value\n"
            code += "  mov rdi, 24 # Buffer size for 64-bit int\n"
            code += "  call malloc # Allocate buffer\n"
            code += "  mov r13, rax # Save buffer pointer\n"
            
            # Format string for snprintf
            format_str = "%d"
            format_label = self.new_string_label(format_str)
            
            # Set up snprintf arguments
            code += "  mov rdi, rax # Buffer pointer (1st arg)\n"
            code += "  mov rsi, 24 # Buffer size (2nd arg)\n"
            code += f"  lea rdx, {format_label}[rip] # Format string (3rd arg)\n"
            code += "  mov rcx, r12 # Integer to convert (4th arg)\n"
            code += "  xor rax, rax # Clear RAX for variadic call\n"
            code += "  call snprintf # Convert int to string\n"
            
            # Return the buffer pointer that was saved earlier
            code += "  mov rax, r13 # Return buffer pointer\n"
            
            return code

        else:
            # Fallback for other free functions
            explicit_arg_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
            call_assembly = f"  # Function Call to {node.name} at L{node.name_token.lineno}\n"
            # Pass arguments
            for i, arg in enumerate(node.arguments):
                call_assembly += self.visit(arg, context)
                if i < len(explicit_arg_regs):
                    call_assembly += f"  mov {explicit_arg_regs[i]}, rax\n"
                else:
                    call_assembly += "  push rax # stack arg\n"
            call_assembly += f"  call {node.name}\n"
            if len(node.arguments) > len(explicit_arg_regs):
                stack_args = len(node.arguments) - len(explicit_arg_regs)
                call_assembly += f"  add rsp, {stack_args * 8} # clean up stack args\n"
            return call_assembly

    def visit_BinaryOpNode(self, node: BinaryOpNode, context):
        op_assembly = f"  # BinaryOp: {node.op} (type: {node.op_token.type}) at L{node.op_token.lineno}\n"
        
        # Get types for both operands
        left_type = self.get_expr_type(node.left, context)
        right_type = self.get_expr_type(node.right, context)
        
        # Type checking for comparison operations
        if node.op_token.type in [TT_LESS_THAN, TT_GREATER_THAN, TT_EQUAL, 
                                  TT_NOT_EQUAL, TT_LESS_EQUAL, TT_GREATER_EQUAL]:
            # Check that both operands are of the same type for comparison
            if left_type != right_type and left_type != "unknown" and right_type != "unknown":
                raise CompilerError(f"Cannot compare values of different types: {left_type} and {right_type} at L{node.op_token.lineno}")
        
        # Check if this is a string concatenation
        is_string_concat = False
        is_string_comparison = False
        
        if node.op_token.type == TT_PLUS:
            if left_type == "string" or right_type == "string":
                is_string_concat = True
        elif node.op_token.type in [TT_LESS_THAN, TT_GREATER_THAN, TT_EQUAL, 
                                    TT_NOT_EQUAL, TT_LESS_EQUAL, TT_GREATER_EQUAL]:
            if left_type == "string" and right_type == "string":
                is_string_comparison = True
        
        if is_string_concat:
            # String concatenation
            op_assembly += self.visit(node.left, context)  # Result in RAX
            op_assembly += "  push rax # Save left string\n"
            
            op_assembly += self.visit(node.right, context) # Result in RAX
            op_assembly += "  push rax # Save right string\n"
            
            # Get length of left string
            op_assembly += "  mov rdi, QWORD PTR [rsp+8] # Left string\n"
            # Add strlen to externs if not already there
            if "strlen" not in self.externs:
                self.externs.append("strlen")
            op_assembly += "  call strlen # Get left string length\n"
            op_assembly += "  mov rbx, rax # Save left length to RBX\n"
            
            # Get length of right string
            op_assembly += "  mov rdi, QWORD PTR [rsp] # Right string\n"
            op_assembly += "  call strlen # Get right string length\n"
            
            # Allocate buffer for concatenated string (left_len + right_len + 1)
            op_assembly += "  add rax, rbx # Total length\n"
            op_assembly += "  add rax, 1 # Add space for null terminator\n"
            op_assembly += "  mov r12, rax # Save total length\n"
            op_assembly += "  mov rdi, rax # Buffer size\n"
            op_assembly += "  call malloc # Allocate buffer\n"
            op_assembly += "  mov r13, rax # Save buffer pointer\n"
            
            # Copy left string to buffer
            op_assembly += "  mov rdi, r13 # Destination buffer\n"
            op_assembly += "  mov rsi, QWORD PTR [rsp+8] # Source (left string)\n"
            # Add strcpy to externs if not already there
            if "strcpy" not in self.externs:
                self.externs.append("strcpy")
            op_assembly += "  call strcpy # Copy left string\n"
            
            # Concatenate right string
            op_assembly += "  mov rdi, rax # Destination buffer (strcpy returns dest)\n"
            op_assembly += "  mov rsi, QWORD PTR [rsp] # Source (right string)\n"
            # Add strcat to externs if not already there
            if "strcat" not in self.externs:
                self.externs.append("strcat")
            op_assembly += "  call strcat # Concatenate right string\n"
            
            # Clean up stack
            op_assembly += "  add rsp, 16 # Pop strings\n"
            
            # Result is in RAX (buffer pointer)
            return op_assembly
        
        # For string comparison operations
        elif is_string_comparison:
            # Add strcmp to externs if not already there
            if "strcmp" not in self.externs:
                self.externs.append("strcmp")
                
            op_assembly += self.visit(node.left, context)  # Result in RAX
            op_assembly += "  push rax # Save left string\n"
            
            op_assembly += self.visit(node.right, context) # Result in RAX
            op_assembly += "  mov rsi, rax # Right string in RSI\n"
            op_assembly += "  pop rdi # Left string in RDI\n"
            
            # Call strcmp to compare the strings
            op_assembly += "  call strcmp # Compare strings\n"
            
            # strcmp returns < 0 if s1 < s2, 0 if s1 == s2, > 0 if s1 > s2
            # For if statements, we'll handle comparison directly in visit_IfNode
            # For expressions, we need to set RAX to 0 or 1
            op_type = node.op_token.type
            
            if op_type == TT_EQUAL:
                op_assembly += "  test rax, rax # Check if strcmp result is 0 (strings equal)\n"
            elif op_type == TT_NOT_EQUAL:
                op_assembly += "  test rax, rax # Check if strcmp result is not 0 (strings not equal)\n"
            elif op_type == TT_LESS_THAN:
                op_assembly += "  cmp rax, 0 # Check if strcmp result < 0\n"
            elif op_type == TT_LESS_EQUAL:
                op_assembly += "  cmp rax, 1 # Check if strcmp result <= 0\n"
            elif op_type == TT_GREATER_THAN:
                op_assembly += "  cmp rax, 0 # Check if strcmp result > 0\n"
            elif op_type == TT_GREATER_EQUAL:
                op_assembly += "  cmp rax, -1 # Check if strcmp result >= 0\n"
            
            # Note: For if statements, these comparison results will be used 
            # by conditional jumps in visit_IfNode
            
            return op_assembly
            
        # Non-string operations
        # Evaluate left operand
        op_assembly += self.visit(node.left, context) 
        op_assembly += "  push rax # Push left operand\n"
        
        # Evaluate right operand
        op_assembly += self.visit(node.right, context) # Right operand result in RAX
        
        op_assembly += "  pop rbx  # Pop left operand into RBX (RBX=left, RAX=right)\n"
        
        op_type = node.op_token.type

        if op_type == TT_PLUS:
            op_assembly += "  add rbx, rax   # RBX = RBX (left) + RAX (right)\n"
            op_assembly += "  mov rax, rbx   # Move result from RBX to RAX\n"
        elif op_type == TT_MINUS:
            op_assembly += "  sub rbx, rax   # RBX = RBX (left) - RAX (right)\n"
            op_assembly += "  mov rax, rbx   # Move result from RBX to RAX\n"
        elif op_type == TT_STAR:
            # For IMUL rbx, rax, result is in rbx. Or IMUL rax, rbx, result is in rax.
            # Let's use the form that puts result in RAX directly: RAX = RAX * RBX
            op_assembly += "  imul rax, rbx  # RAX = RAX (right) * RBX (left)\n"
        elif op_type == TT_SLASH: # Integer division: left / right (RBX / RAX)
            op_assembly += "  mov rcx, rax   # Divisor (right operand) from RAX to RCX\n"
            op_assembly += "  mov rax, rbx   # Dividend (left operand) from RBX to RAX\n"
            op_assembly += "  cqo            # Sign-extend RAX into RDX:RAX\n"
            op_assembly += "  idiv rcx       # RDX:RAX / RCX. Quotient in RAX, Remainder in RDX\n"
            # Result (quotient) is already in RAX
        elif op_type == TT_PERCENT: # Modulo: left % right (RBX % RAX)
            op_assembly += "  mov rcx, rax   # Divisor (right operand) from RAX to RCX\n"
            op_assembly += "  mov rax, rbx   # Dividend (left operand) from RBX to RAX\n"
            op_assembly += "  cqo            # Sign-extend RAX into RDX:RAX\n"
            op_assembly += "  idiv rcx       # Quotient in RAX, Remainder in RDX\n"
            op_assembly += "  mov rax, rdx   # Move remainder from RDX to RAX\n"
        # Bitwise operations
        elif op_type == TT_AMPERSAND: # Bitwise AND: left & right
            op_assembly += "  and rbx, rax   # RBX = RBX (left) & RAX (right)\n"
            op_assembly += "  mov rax, rbx   # Move result from RBX to RAX\n"
        elif op_type == TT_PIPE: # Bitwise OR: left | right
            op_assembly += "  or rbx, rax    # RBX = RBX (left) | RAX (right)\n"
            op_assembly += "  mov rax, rbx   # Move result from RBX to RAX\n"
        elif op_type == TT_CARET: # Bitwise XOR: left ^ right
            op_assembly += "  xor rbx, rax   # RBX = RBX (left) ^ RAX (right)\n"
            op_assembly += "  mov rax, rbx   # Move result from RBX to RAX\n"
        # Shift operations
        elif op_type == TT_LSHIFT: # Left shift: left << right
            op_assembly += "  mov rcx, rax   # Shift count (right operand) from RAX to RCX\n"
            op_assembly += "  mov rax, rbx   # Value to shift (left operand) from RBX to RAX\n"
            op_assembly += "  sal rax, cl    # RAX = RAX << CL (shift arithmetic left)\n"
        elif op_type == TT_RSHIFT: # Right shift: left >> right (arithmetic shift)
            op_assembly += "  mov rcx, rax   # Shift count (right operand) from RAX to RCX\n"
            op_assembly += "  mov rax, rbx   # Value to shift (left operand) from RBX to RAX\n"
            op_assembly += "  sar rax, cl    # RAX = RAX >> CL (shift arithmetic right - sign extending)\n"
        elif op_type == TT_URSHIFT: # Unsigned right shift: left >>> right (logical shift)
            op_assembly += "  mov rcx, rax   # Shift count (right operand) from RAX to RCX\n"
            op_assembly += "  mov rax, rbx   # Value to shift (left operand) from RBX to RAX\n"
            op_assembly += "  shr rax, cl    # RAX = RAX >>> CL (shift logical right - zero filling)\n"
        elif op_type == TT_LESS_THAN: 
            op_assembly += "  cmp rbx, rax   # Compare left (RBX) with right (RAX) for <\n"
            # Result of cmp is in flags, to be used by conditional jumps (e.g., in IfNode)
            # For expressions, we need to set rax to 0 or 1.
            # op_assembly += "  setl al      # Set AL to 1 if less (SF!=OF), else 0\n"
            # op_assembly += "  movzx rax, al  # Zero-extend AL to RAX\n"
            # The above setl/setg for expressions is not implemented yet. IfNode handles cmp directly.
            # For now, comparison ops are only for If conditions, not general expressions.
        elif op_type == TT_GREATER_THAN:
            op_assembly += "  cmp rbx, rax   # Compare left (RBX) with right (RAX) for >\n"
        elif op_type == TT_LESS_EQUAL:
            op_assembly += "  cmp rbx, rax   # Compare left (RBX) with right (RAX) for <=\n"
        elif op_type == TT_GREATER_EQUAL:
            op_assembly += "  cmp rbx, rax   # Compare left (RBX) with right (RAX) for >=\n"
        elif op_type == TT_EQUAL:
            op_assembly += "  cmp rbx, rax   # Compare left (RBX) with right (RAX) for ==\n"
        elif op_type == TT_NOT_EQUAL:
            op_assembly += "  cmp rbx, rax   # Compare left (RBX) with right (RAX) for !=\n"
        else:
            raise CompilerError(f"Unsupported binary operator type '{node.op_token.type}' (op val: '{node.op}') at L{node.op_token.lineno}")
        return op_assembly

    def get_expr_type(self, node, context):
        """Helper method to determine expression type for string concatenation checks"""
        if isinstance(node, StringLiteralNode):
            return "string"
        elif isinstance(node, IntegerLiteralNode):
            return "int"
        elif isinstance(node, IdentifierNode):
            var_name = node.value
            if var_name in self.current_method_locals:
                return self.current_method_locals[var_name]['type']
            elif var_name in self.current_method_params:
                return self.current_method_params[var_name]['type']
        elif isinstance(node, FunctionCallNode):
            func_name = node.name
            if func_name == "input" or func_name == "to_string":
                return "string"
            elif func_name == "to_int":
                return "int"
            elif func_name == "read":
                return "string"
            elif func_name == "concat_int" or func_name == "concat_string":
                # Special handling for common list-returning functions
                return func_name.replace("concat_", "") + "[]"
            elif func_name == "concat_matrix_int" or func_name == "concat_matrix_string":
                # Special handling for common matrix-returning functions
                return func_name.replace("concat_matrix_", "") + "[][]"
        elif isinstance(node, MethodCallNode):
            # Try to determine method return type
            method_name = node.method_name
            current_class_name = context[0] if context else None
            
            # Look for method in program AST
            if hasattr(self, 'current_program_node'):
                for cls in self.current_program_node.classes:
                    cls_name = cls.name.replace('.', '_')
                    
                    # Check if this is the class we're looking for
                    if isinstance(node.object_expr, ThisNode) and cls_name == current_class_name:
                        # This is a this:method() call
                        for method in cls.methods:
                            if method.name == method_name:
                                return method.return_type
                    elif isinstance(node.object_expr, IdentifierNode):
                        # This could be a Class.method() or obj.method() call
                        obj_name = node.object_expr.value
                        if cls_name == obj_name:
                            # This is a Class.method() call
                            for method in cls.methods:
                                if method.name == method_name:
                                    return method.return_type
            
            # Special case handling for common methods
            if method_name == "concat_int" or method_name == "concat_string":
                return method_name.replace("concat_", "") + "[]"
            elif method_name == "concat_matrix_int" or method_name == "concat_matrix_string":
                return method_name.replace("concat_matrix_", "") + "[][]"
        elif isinstance(node, ArrayAccessNode):
            # Added support for array access type detection
            element_type = self.get_array_element_type(node.array_expr, context)
            return element_type
        elif isinstance(node, ListLiteralNode):
            # Determine the type of the list elements
            if node.elements:
                element_type = self.get_expr_type(node.elements[0], context)
                if element_type == "unknown":
                    return "unknown[]"
                else:
                    return f"{element_type}[]"
            else:
                return "unknown[]"  # Empty list
        # Default for unknown/unhandled expressions
        return "unknown"

    def get_array_element_type(self, array_expr, context):
        """
        Helper method to determine the element type of an array.
        Returns the base type (e.g., "int", "string") without the "[]" suffix.
        """
        if isinstance(array_expr, IdentifierNode):
            var_name = array_expr.value
            var_type = None
            
            # Check local variables first, then parameters
            if var_name in self.current_method_locals:
                var_type = self.current_method_locals[var_name]['type']
            elif var_name in self.current_method_params:
                var_type = self.current_method_params[var_name]['type']
                
            if var_type and var_type.endswith("[]"):
                # Return the base type without the "[]" suffix
                return var_type[:-2]
        
        # For nested arrays (like matrix[i][j]) or complex expressions
        # When evaluating something like matrix[i], which returns a list pointer
        elif isinstance(array_expr, ArrayAccessNode):
            # Recursively get the element type of the parent array
            parent_type = self.get_array_element_type(array_expr.array_expr, context)
            if parent_type and parent_type.endswith("[]"):
                return parent_type[:-2]
        
        # Default to "unknown" if we can't determine the type
        return "unknown"

    def visit_IfNode(self, node: IfNode, context):
        if_assembly = f"  # If statement at L{node.lineno}\n"
        else_label, end_if_label = self.new_if_labels()
        
        # Check if the condition is a string comparison
        is_string_comparison = False
        if isinstance(node.condition, BinaryOpNode):
            left_type = self.get_expr_type(node.condition.left, context)
            right_type = self.get_expr_type(node.condition.right, context)
            if left_type == "string" and right_type == "string":
                is_string_comparison = True
        
        if_assembly += self.visit(node.condition, context)
        jump_instruction = ""
        op_type = node.condition.op_token.type
        
        if is_string_comparison:
            # For string comparisons, the result of strcmp is in RAX
            # strcmp returns < 0 if s1 < s2, 0 if s1 == s2, > 0 if s1 > s2
            if op_type == TT_LESS_THAN:
                jump_instruction = "jge" # Jump if RAX >= 0 (not less)
            elif op_type == TT_GREATER_THAN:
                jump_instruction = "jle" # Jump if RAX <= 0 (not greater)
            elif op_type == TT_LESS_EQUAL:
                jump_instruction = "jg"  # Jump if RAX > 0 (not less/equal)
            elif op_type == TT_GREATER_EQUAL:
                jump_instruction = "jl"  # Jump if RAX < 0 (not greater/equal)
            elif op_type == TT_EQUAL:
                jump_instruction = "jne" # Jump if RAX != 0 (not equal)
            elif op_type == TT_NOT_EQUAL:
                jump_instruction = "je"  # Jump if RAX == 0 (equal)
            else:
                raise CompilerError(f"Unsupported string comparison operator '{op_type}' in if. L{node.condition.op_token.lineno}")
        else:
            # For integer comparisons, we use the flags set by cmp instruction
            if op_type == TT_LESS_THAN:
                jump_instruction = "jge"
            elif op_type == TT_GREATER_THAN:
                jump_instruction = "jle"
            elif op_type == TT_LESS_EQUAL:
                jump_instruction = "jg"
            elif op_type == TT_GREATER_EQUAL:
                jump_instruction = "jl"
            elif op_type == TT_EQUAL:
                jump_instruction = "jne"
            elif op_type == TT_NOT_EQUAL:
                jump_instruction = "je"
            else:
                raise CompilerError(f"Unsupported comparison operator '{node.condition.op_token.type}' in if. L{node.condition.op_token.lineno}")
        
        jump_target = else_label if node.else_block else end_if_label
        if_assembly += f"  {jump_instruction} {jump_target}\n"
        if_assembly += self.visit(node.then_block, context)
        if node.else_block:
            if_assembly += f"  jmp {end_if_label}\n"
            if_assembly += f"{else_label}:\n"
            if_assembly += self.visit(node.else_block, context)
        if_assembly += f"{end_if_label}:\n"
        return if_assembly

    def visit_UnaryOpNode(self, node: UnaryOpNode, context):
        # Postfix increment/decrement: i++, i--
        # The value of the expression is the value of i BEFORE the operation.
        # 1. Get the memory location of the operand (must be an L-value, e.g., IdentifierNode)
        # 2. Load current value from memory into RAX (this is the expression's result)
        # 3. Perform inc/dec on the memory location

        assembly_code = f"  # UnaryOp {node.op_token.type} on {type(node.operand_node)} at L{node.lineno}\n"

        # Handle bitwise NOT (~)
        if node.op_token.type == TT_TILDE:
            assembly_code += self.visit(node.operand_node, context) # Result in RAX
            assembly_code += "  not rax      # Bitwise NOT\n"
            return assembly_code

        # Handle postfix increment/decrement (++, --)
        if not isinstance(node.operand_node, IdentifierNode):
            # For now, only allow inc/dec on simple identifiers (variables).
            # Future: could extend to array elements or fields if they become L-values.
            raise CompilerError(f"Operand of ++/-- must be an identifier, got {type(node.operand_node)} at L{node.lineno}")

        var_name = node.operand_node.value
        var_info = None

        if var_name in self.current_method_locals:
            var_info = self.current_method_locals[var_name]
        elif var_name in self.current_method_params:
            var_info = self.current_method_params[var_name]
        else:
            raise CompilerError(f"Identifier '{var_name}' not found for ++/-- operation at L{node.lineno}")

        if var_info['type'] != 'int':
            raise CompilerError(f"++/-- operator can only be applied to 'int' type variables, not '{var_info['type']}' for '{var_name}' at L{node.lineno}")

        mem_operand = f"QWORD PTR [rbp {var_info['offset_rbp']}]"

        assembly_code += f"  mov rax, {mem_operand}  # Load current value of {var_name} for expression result\n"

        if node.op_token.type == TT_PLUSPLUS:
            assembly_code += f"  inc {mem_operand}       # Increment {var_name} in memory\n"
        elif node.op_token.type == TT_MINUSMINUS:
            assembly_code += f"  dec {mem_operand}       # Decrement {var_name} in memory\n"
        else:
            raise CompilerError(f"Unsupported unary operator: {node.op_token.type} at L{node.lineno}")
        
        return assembly_code

    def visit_LoopNode(self, node: LoopNode, context):
        # Infinite loop: execute body and jump back until a stop
        loop_id = self.next_loop_id
        self.next_loop_id += 1
        start_label = f".L_loop_start_{loop_id}"
        end_label = f".L_loop_end_{loop_id}"
        # Push end_label for stop statements
        self.loop_end_labels.append(end_label)
        code = f"{start_label}:\n"
        code += self.visit(node.body, context)
        code += f"  jmp {start_label}\n"
        code += f"{end_label}:\n"
        # Pop the label after loop
        self.loop_end_labels.pop()
        return code

    def visit_StopNode(self, node: StopNode, context):
        # Jump to nearest loop end label
        if not self.loop_end_labels:
            raise CompilerError(f"Stop used outside of loop at L{node.lineno}")
        end_label = self.loop_end_labels[-1]
        return f"  jmp {end_label} # stop\n"

    def visit_AssignmentNode(self, node: AssignmentNode, context):
        # Assignment statement: variable = expression  OR list[index] = expression
        code = f"  # Assignment to {node.name_token.value if isinstance(node.name_token, Token) else 'list element'} at L{node.lineno}\n"
        
        # Evaluate RHS first, result in RAX
        code += self.visit(node.expression, context)
        code += "  push rax # Save RHS value on stack\n"

        # Determine if it's a simple variable assignment or array element assignment
        if isinstance(node.name_token, Token): # Simple variable assignment: myVar = ...
            var_name = node.name_token.value # node.name is used in parser, but name_token here
            if var_name in self.current_method_locals:
                offset = self.current_method_locals[var_name]['offset_rbp']
            elif var_name in self.current_method_params:
                offset = self.current_method_params[var_name]['offset_rbp']
            else:
                raise CompilerError(f"Assignment to undefined variable '{var_name}' at L{node.lineno}")
            
            code += "  pop rbx    # RHS value into RBX\n"
            code += f"  mov QWORD PTR [rbp {offset}], rbx # {var_name} = RBX\n"
        
        elif isinstance(node.name_token, ArrayAccessNode): # Array element assignment: myList[idx] = ...
            array_access_node = node.name_token # This was misnamed 'name' in AST, should be 'target' or 'lvalue'
                                                 # For now, assume node.name_token is the ArrayAccessNode from parser
            
            # 1. Evaluate array expression (list_ptr)
            code += self.visit(array_access_node.array_expr, context) # list_ptr in RAX
            code += "  push rax # Save list_ptr\n"
            
            # 2. Evaluate index expression
            code += self.visit(array_access_node.index_expr, context) # index in RAX
            code += "  mov rbx, rax # Index in RBX\n"
            
            code += "  pop rax    # list_ptr back in RAX\n"
            code += "  pop rcx    # RHS value in RCX (from first push rax)\n"

            # Bounds check (optional, good practice)
            # code += "  mov rdx, QWORD PTR [rax + 8] # Get length from list_ptr\n"
            # code += "  cmp rbx, rdx                 # Compare index with length\n"
            # code += "  jl .L_assign_index_in_bounds_{node.lineno} \\n"
            # code += "  # Handle index out of bounds error here (e.g., print error and exit)\\n"
            # code += f".L_assign_index_in_bounds_{node.lineno}:\\n"

            # Calculate address: list_ptr + 16 (header) + (index * 8)
            code += "  imul rbx, 8                  # index_offset = index * 8\n"
            code += "  add rbx, 16                  # total_offset = header_size + index_offset\n"
            code += "  add rax, rbx                 # target_address = list_ptr + total_offset\n"
            
            # Store RHS value (RCX) into the calculated address
            code += "  mov QWORD PTR [rax], rcx     # list_ptr[index] = RHS_value\n"
        else:
            raise CompilerError(f"Invalid LHS for assignment at L{node.lineno}")
            
        return code

    def visit_MacroDefNode(self, node: MacroDefNode, context=None):
        # Macro definitions don't generate code directly
        return ""
        
    def visit_MacroInvokeNode(self, node: MacroInvokeNode, context):
        # Look up the macro definition
        macro_name = node.name
        current_class_name = None if context is None else context[0]
        
        # Determine the right macro to use based on context
        macro_def = None
        
        # If the macro is called on an object (obj.@macro or this:@macro)
        if node.object_expr:
            # For 'this:@macro', use the current class
            if isinstance(node.object_expr, ThisNode):
                if current_class_name:
                    lookup_key = f"{current_class_name}_{macro_name}"
                    macro_def = self.macros.get(lookup_key)
            # For 'obj.@macro', try to determine the class of obj
            elif isinstance(node.object_expr, IdentifierNode):
                obj_name = node.object_expr.value
                obj_type = None
                # Look up the variable type
                if obj_name in self.current_method_locals:
                    obj_type = self.current_method_locals[obj_name]['type']
                elif obj_name in self.current_method_params:
                    obj_type = self.current_method_params[obj_name]['type']
                
                if obj_type and obj_type not in ['int', 'string']:
                    # Use the variable's type as the class name
                    lookup_key = f"{obj_type}_{macro_name}"
                    macro_def = self.macros.get(lookup_key)
                else:
                    # Might be a direct class name like Math.@add
                    lookup_key = f"{obj_name}_{macro_name}"
                    macro_def = self.macros.get(lookup_key)
        
        # If no class-specific macro was found, try the global macro
        if macro_def is None:
            # If we're in a class context, try the current class's macro first
            if current_class_name:
                lookup_key = f"{current_class_name}_{macro_name}"
                macro_def = self.macros.get(lookup_key)
            
            # If still not found, try global macro
            if macro_def is None:
                macro_def = self.macros.get(macro_name)
        
        if macro_def is None:
            raise CompilerError(f"Unknown macro '{macro_name}' at L{node.lineno}")
            
        # Check argument count
        if len(node.arguments) != len(macro_def.params):
            raise CompilerError(f"Macro '{macro_name}' expects {len(macro_def.params)} arguments, got {len(node.arguments)} at L{node.lineno}")
            
        # Create a dictionary mapping parameter names to argument expressions
        param_to_arg = {}
        for i, param_name in enumerate(macro_def.params):
            param_to_arg[param_name] = node.arguments[i]
            
        # Clone the macro body with arguments substituted
        # For now, we only support simple expression bodies
        if isinstance(macro_def.body, BinaryOpNode):
            # For binary operators like a + b
            if isinstance(macro_def.body.left, IdentifierNode) and macro_def.body.left.value in param_to_arg:
                left_arg = param_to_arg[macro_def.body.left.value]
                code = self.visit(left_arg, context)
                code += "  push rax # Push left arg for macro expansion\n"
            else:
                code = self.visit(macro_def.body.left, context)
                code += "  push rax # Push left side for macro expansion\n"
                
            if isinstance(macro_def.body.right, IdentifierNode) and macro_def.body.right.value in param_to_arg:
                right_arg = param_to_arg[macro_def.body.right.value]
                code += self.visit(right_arg, context)
            else:
                code += self.visit(macro_def.body.right, context)
                
            # Now perform the operation
            code += "  mov rbx, rax # Move right value to rbx\n"
            code += "  pop rax # Get left value\n"
            
            # Apply the operation based on the operator type
            op_type = macro_def.body.op_token.type
            if op_type == TT_PLUS:
                code += "  add rax, rbx # Macro expansion: add\n"
            elif op_type == TT_MINUS:
                code += "  sub rax, rbx # Macro expansion: subtract\n"
            elif op_type == TT_STAR:
                code += "  imul rax, rbx # Macro expansion: multiply\n"
            elif op_type == TT_SLASH:
                code += "  mov rcx, rbx # Move divisor to rcx\n"
                code += "  cqo # Sign-extend rax into rdx:rax\n"
                code += "  idiv rcx # Macro expansion: divide\n"
            elif op_type == TT_PERCENT:
                code += "  mov rcx, rbx # Move divisor to rcx\n"
                code += "  cqo # Sign-extend rax into rdx:rax\n"
                code += "  idiv rcx # Macro expansion: modulo\n"
                code += "  mov rax, rdx # Move remainder to rax\n"
            else:
                raise CompilerError(f"Unsupported operator in macro expansion: {op_type} at L{node.lineno}")
                
            return code
            
        # For simple identifier parameters, just evaluate the argument directly
        elif isinstance(macro_def.body, IdentifierNode) and macro_def.body.value in param_to_arg:
            return self.visit(param_to_arg[macro_def.body.value], context)
            
        # For literals and other expressions, just evaluate the body
        else:
            return self.visit(macro_def.body, context)

    def visit_ListLiteralNode(self, node: ListLiteralNode, context):
        num_elements = len(node.elements)
        code = f"  # ListLiteral with {num_elements} elements at L{node.lineno}\n"

        # Initial capacity: max(8, num_elements) for some initial space, or just num_elements if exact fit
        initial_capacity = max(8, num_elements) 
        
        # Allocate memory for list structure: header (16 bytes) + elements
        total_size = 16 + (initial_capacity * 8) # Assuming 8 bytes per element
        code += f"  mov rdi, {total_size} # Size for list allocation\n"
        code += "  call malloc           # list_ptr in RAX\n"
        code += "  push rax              # Save list_ptr on stack\n"

        # Store capacity and length
        code += f"  mov QWORD PTR [rax], {initial_capacity} # Store capacity at list_ptr[0]\n"
        code += f"  mov QWORD PTR [rax+8], {num_elements}   # Store length at list_ptr[8]\n"

        # Store elements
        element_base_offset = 16 # Start of element data
        for i, expr_node in enumerate(node.elements):
            # Check if the element is itself a ListLiteralNode (for nested lists/matrices)
            if isinstance(expr_node, ListLiteralNode):
                # For nested list literals, recursively process them
                code += self.visit(expr_node, context) # This will create the inner list and leave its pointer in RAX
            else:
                # Normal element
                code += self.visit(expr_node, context) # Element value in RAX
            
            # Need to get list_ptr from stack to store element
            code += "  pop rbx               # list_ptr into RBX\n"
            current_element_offset = element_base_offset + (i * 8)
            code += f"  mov QWORD PTR [rbx + {current_element_offset}], rax # list_ptr[element_offset] = value\n"
            code += "  push rbx              # Push list_ptr back on stack\n"
        
        code += "  pop rax # Final list_ptr (from last push) into RAX to be the result of this expression\n"
        return code

    def visit_ArrayAccessNode(self, node: ArrayAccessNode, context):
        # Add a unique ID generator for array access operations
        if not hasattr(self, 'array_access_counter'):
            self.array_access_counter = 0
        self.array_access_counter += 1
        unique_id = f"{node.lineno}_{self.array_access_counter}"
        
        # Determine the element type for this array access
        element_type = self.get_array_element_type(node.array_expr, context)
        
        code = f"  # ArrayAccessNode: list[index] of type {element_type} at L{node.lineno}\n"
        
        # 1. Evaluate array_expr (list_ptr) -> result in RAX
        code += self.visit(node.array_expr, context)
        code += "  push rax # Save list_ptr on stack\n"
        
        # 2. Evaluate index_expr -> result in RAX
        code += self.visit(node.index_expr, context)
        code += "  mov rbx, rax # Index in RBX\n"
        
        code += "  pop rax    # list_ptr back in RAX\n"

        # Check for null list pointer
        code += "  cmp rax, 0 # Check for null pointer \n"
        code += f"  jne .L_access_not_null_{unique_id}\n"
        # Handle null pointer access
        error_msg_null = self.new_string_label("Error: Null pointer access in array operation.\n")
        code += f"  lea rdi, {error_msg_null}[rip]\n"
        code += "  call printf\n"
        code += "  mov rdi, 1\n"
        code += "  call exit\n"
        code += f".L_access_not_null_{unique_id}:\n"
        
        # Bounds check
        idx_ok_label = f".L_access_idx_ok_{unique_id}"
        idx_err_label = f".L_access_idx_err_{unique_id}"

        code += "  mov rcx, QWORD PTR [rax + 8] # length in RCX\n"
        code += "  cmp rbx, 0                   # if index < 0\n"
        code += f"  jl {idx_err_label}           # jump to error\n"
        code += "  cmp rbx, rcx                 # if index >= length\n"
        code += f"  jge {idx_err_label}          # jump to error\n"
        code += f"  jmp {idx_ok_label}           # index is OK\n"

        code += f"{idx_err_label}:\n"
        error_msg_bounds = self.new_string_label("Error: Array index out of bounds.\n")
        code += f"  lea rdi, {error_msg_bounds}[rip]\n"
        code += "  call printf\n"
        code += "  mov rdi, 1\n"
        code += "  call exit\n"
        
        code += f"{idx_ok_label}:\n"
        # Calculate address: list_ptr + 16 (header) + (index * 8)
        code += "  imul rbx, 8                  # index_offset = index * 8\n"
        code += "  add rbx, 16                  # total_offset = header_size + index_offset\n"
        code += "  add rax, rbx                 # element_address = list_ptr + total_offset\n"
        
        # Load value from element_address into RAX
        code += "  mov rax, QWORD PTR [rax]     # RAX = value at list_ptr[index]\n"
        
        return code

    def compile(self, node: ProgramNode):
        # Store the program node for reference
        self.current_program_node = node
        return self.visit(node)

if __name__ == '__main__':
    from .parser_lexer import Lexer, Parser
    source_code_funcs = '''
    class Main {
        def main() -> void {
            print("10 + 20 = ", this:add(10, 20))
        }
        def add(a int, b int) -> int {
            return a + b
        }
    }
    '''
    print(f"Compiling source:\n{source_code_funcs}")
    lexer = Lexer(source_code_funcs)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    compiler = Compiler()
    try:
        assembly = compiler.compile(ast)
        print("\nGenerated Assembly (GAS x86-64):")
        print(assembly)
        with open("output_funcs.s", "w") as f:
            f.write(assembly)
        print("\nAssembly saved to output_funcs.s")
        print("Try to assemble and link with: gcc -no-pie output_funcs.s -o funcs_program")
        print("Then run: ./funcs_program")
    except CompilerError as e:
        print(f"\nCompiler Error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during compilation: {e}")
        import traceback
        traceback.print_exc()

    source_code_print_multi = '''
    class Main {
        def main() -> void {
            print("Number: ", 123, " Also: ", this:get_num())
        }
        def get_num() -> int {
            return 456
        }
    }
    '''
    print(f"\nCompiling source:\n{source_code_print_multi}")
    lexer_pm = Lexer(source_code_print_multi)
    tokens_pm = lexer_pm.tokenize()
    parser_pm = Parser(tokens_pm)
    ast_pm = parser_pm.parse()
    compiler_pm = Compiler()
    try:
        assembly_pm = compiler_pm.compile(ast_pm)
        print("\nGenerated Assembly for print_multi:")
        print(assembly_pm)
        with open("output_print_multi.s", "w") as f:
            f.write(assembly_pm)
        print("\nAssembly saved to output_print_multi.s")
    except CompilerError as e:
        print(f"\nCompiler Error (print_multi): {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during compilation (print_multi): {e}")
        import traceback
        traceback.print_exc()

