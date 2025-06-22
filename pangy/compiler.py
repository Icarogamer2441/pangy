from .lexer import (
    Lexer, Token,
    TT_CLASS, TT_DEF, TT_IDENTIFIER, TT_LPAREN, TT_RPAREN, TT_LBRACE, TT_RBRACE, TT_ARROW, 
    TT_TYPE, TT_STRING_LITERAL, TT_PRINT, TT_SEMICOLON, TT_EOF, TT_MACRO, TT_AT, 
    TT_PUBLIC, TT_PRIVATE, TT_INT_LITERAL, TT_COMMA, TT_COLON, TT_DOUBLE_COLON, TT_PLUS, 
    TT_RETURN, TT_THIS, TT_ASSIGN, TT_VAR, TT_MINUS, TT_STAR, TT_SLASH, TT_PERCENT, 
    TT_PLUSPLUS, TT_MINUSMINUS, TT_IF, TT_ELSE, TT_LESS_THAN, TT_GREATER_THAN, TT_EQUAL, 
    TT_NOT_EQUAL, TT_LESS_EQUAL, TT_GREATER_EQUAL, TT_STATIC, TT_DOT, TT_INCLUDE, 
    TT_LOOP, TT_STOP, TT_LBRACKET, TT_RBRACKET, TT_AMPERSAND, TT_PIPE, TT_CARET, 
    TT_TILDE, TT_LSHIFT, TT_RSHIFT, TT_URSHIFT, TT_LOGICAL_AND, TT_LOGICAL_OR, 
    TT_TRUE, TT_FALSE
)
from .parser import (
    Parser, ASTNode, ProgramNode, IncludeNode, ClassNode, ParamNode, MethodNode, 
    BlockNode, PrintNode, ReturnNode, ExprNode, StringLiteralNode, IntegerLiteralNode, 
    FloatLiteralNode, IdentifierNode, ThisNode, MethodCallNode, FunctionCallNode, 
    BinaryOpNode, UnaryOpNode, IfNode, VarDeclNode, LoopNode, WhileNode, StopNode, AssignmentNode, 
    MacroDefNode, MacroInvokeNode, ListLiteralNode, ArrayAccessNode, ClassVarDeclNode, 
    ClassVarAccessNode, TrueLiteralNode, FalseLiteralNode, CLibraryCallNode
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
        self.next_while_id = 0 # For while loops
        self.next_list_header_label_id = 0 # For list static data
        self.next_append_label_id = 0 # For append function labels
        self.next_pop_label_id = 0 # For pop function labels
        self.next_index_label_id = 0 # For index function labels
        self.next_array_access_id = 0 # For array access operations
        self.next_to_int_id = 0 # For to_int function labels
        self.next_to_string_id = 0 # For to_string function labels
        self.next_input_id = 0 # For input function labels
        self.next_float_label_id = 0 # For float literal labels
        self.next_class_var_id = 0 # For class variable access operations
        self.externs = ["printf", "exit", "scanf", "atoi", "malloc", "fopen", "fclose", "fwrite", "fread", "fgets", "realloc", "memcpy", "strlen", "strcpy", "strcat", "system", "popen", "pclose", "free", "chdir"] # Added chdir
        
        self.current_method_params = {}
        self.current_method_locals = {} # For local variables
        self.current_method_stack_offset = 0 # Tracks current available stack slot relative to RBP for new locals
        self.current_method_total_stack_needed = 0 # For params + locals in prologue
        self.current_method_context = None
        self.this_ptr_rbp_offset = None # Offset for the saved 'this' pointer on stack for instance methods
        # For loop support
        self.loop_end_labels = []
        # For macro support
        self.macros = {} # Maps macro names to MacroDefNode objects
        self.class_vars = {}  # Maps class_name -> {var_name -> (offset, type, is_public)}
        self.used_c_libs = set() # To track which C libraries are used, e.g. {"m"}

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

    def new_float_literal_label(self, value):
        # Create a unique label for a float literal and emit it as a double in .rodata
        label = f".LCF{self.next_float_label_id}"
        self.next_float_label_id += 1
        self.data_section_code += f"{label}:\n  .double {value}\n"
        return label

    def new_if_labels(self):
        idx = self.next_if_label_id
        self.next_if_label_id += 1
        return f".L_else_{idx}", f".L_end_if_{idx}"

    def new_while_labels(self):
        idx = self.next_while_id
        self.next_while_id += 1
        return f".L_while_start_{idx}", f".L_while_end_{idx}"

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
        elif lineno_attr is None and hasattr(node, 'method_name_token'): # For MethodCallNode
             lineno_attr = getattr(node.method_name_token, 'lineno', 'unknown')
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
        self.used_c_libs = set() # Reset for each compilation run
        self.next_string_label_id = 0; self.next_printf_format_label_id = 0; self.next_if_label_id = 0; self.next_loop_id = 0; self.next_while_id = 0; self.next_list_header_label_id = 0
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
        
        # Process class variables first to register them
        self.class_vars[class_label] = {}
        offset = 0
        for var_node in node.class_vars:
            var_type = var_node.type
            var_name = var_node.name
            is_public = var_node.is_public
            # Store class variable info (offset from object start, type, public/private)
            self.class_vars[class_label][var_name] = (offset, var_type, is_public)
            offset += 8  # Assuming all variables (including list pointers) need 8 bytes
            
        for method_node in node.methods:
            self.current_method_context = (class_label, method_node.name)
            class_assembly += self.visit(method_node)
        self.current_method_context = None
        return class_assembly

    def visit_ClassVarDeclNode(self, node: ClassVarDeclNode, context=None):
        # This is only called during class processing to set up class variables
        # The actual code for variable declaration is handled in visit_ClassNode
        return ""

    def visit_ClassVarAccessNode(self, node: ClassVarAccessNode, context):
        class_name, method_name = context
        var_name = node.var_name
        
        # Check if this is a 'this:varname' access
        is_this_access = isinstance(node.object_expr, ThisNode)
        
        # Accessing a variable from the current class
        if is_this_access:
            if var_name not in self.class_vars.get(class_name, {}):
                raise CompilerError(f"Undefined class variable '{var_name}' at L{node.lineno}")
                
            offset, var_type, is_public = self.class_vars[class_name][var_name]
            code = f"  # Access to class variable this:{var_name} at L{node.lineno}\n"
            if self.this_ptr_rbp_offset is None:
                raise CompilerError(f"'this' pointer not saved for instance method context. L{node.lineno}")
            
            code += f"  mov rax, QWORD PTR [rbp {self.this_ptr_rbp_offset}]  # Load saved 'this' pointer\n"
            
            # Handle different variable types
            if var_type == "float":
                # For float variables, load directly into XMM0
                code += f"  movsd xmm0, QWORD PTR [rax + {offset}]  # Load float {var_name} from object into XMM0\n"
            else:
                code += f"  mov rax, QWORD PTR [rax + {offset}]  # Load {var_name} from object\n"
            return code
        
        # Accessing a variable from another object (e.g., obj::varname)
        else:
            # First evaluate the object expression
            code = self.visit(node.object_expr, context)
            code += "  mov rcx, rax  # Save object pointer in RCX\n"
            
            # Now we need to determine the class type of the object
            # This is a simplification - in a real compiler, we'd have a type system
            # For now, use the value stored in the local var or param (if available)
            obj_class_type = None
            if isinstance(node.object_expr, IdentifierNode):
                obj_name = node.object_expr.value
                if obj_name in self.current_method_locals:
                    obj_class_type = self.current_method_locals[obj_name]['type']
                elif obj_name in self.current_method_params:
                    obj_class_type = self.current_method_params[obj_name]['type']
                    
            if not obj_class_type:
                raise CompilerError(f"Cannot determine class type for object access at L{node.lineno}")
            
            obj_class_type = obj_class_type.replace('.', '_')  # Safe class name for lookup
            
            if var_name not in self.class_vars.get(obj_class_type, {}):
                raise CompilerError(f"Undefined class variable '{var_name}' in class '{obj_class_type}' at L{node.lineno}")
            
            offset, var_type, is_public = self.class_vars[obj_class_type][var_name]
            
            # Check access permission
            if not is_public and class_name != obj_class_type:
                raise CompilerError(f"Cannot access private variable '{var_name}' from outside class '{obj_class_type}' at L{node.lineno}")
            
            code += f"  # Access to class variable {obj_class_type}::{var_name} at L{node.lineno}\n"
            
            # Handle different variable types
            if var_type == "float":
                # For float variables, load directly into XMM0
                code += f"  movsd xmm0, QWORD PTR [rcx + {offset}]  # Load float {var_name} from object into XMM0\n"
            else:
                code += f"  mov rax, QWORD PTR [rcx + {offset}]  # Load {var_name} from object\n"
            return code

    def visit_MethodNode(self, node: MethodNode, context=None):
        class_name, method_name = self.current_method_context
        method_label = f"{class_name}_{method_name}"
        epilogue_label = f".L_epilogue_{class_name}_{method_name}"
        self.current_method_params = {} 
        self.current_method_locals = {}
        self.this_ptr_rbp_offset = None # Reset for each method
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
            # Reserve stack space for 'this' pointer and save it
            current_rbp_offset_for_params -= 8 # Decrement first for this_ptr_rbp_offset
            self.this_ptr_rbp_offset = current_rbp_offset_for_params
            local_stack_needed_for_params += 8
            method_text_assembly += f"  # 'this' pointer will be saved at [rbp {self.this_ptr_rbp_offset}]\n"
        
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

        # Save 'this' pointer for instance methods after stack is allocated
        if not is_method_static and self.this_ptr_rbp_offset is not None:
            method_text_assembly += f"  mov QWORD PTR [rbp {self.this_ptr_rbp_offset}], rdi # Save 'this' pointer\n"
        
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
            # Track which XMM registers are used for float parameters
            xmm_reg_idx = 0
            
            # First, process non-float parameters
            for p_name, p_data in self.current_method_params.items():
                if 'reg' in p_data and p_data['type'] != 'float':
                    method_text_assembly += f"  mov QWORD PTR [rbp {p_data['offset_rbp']}], {p_data['reg']} # Spill param {p_name}\n"
            
            # Then process float parameters, which use XMM registers
            for p_name, p_data in self.current_method_params.items():
                if 'reg' in p_data and p_data['type'] == 'float':
                    xmm_reg = f"xmm{xmm_reg_idx}"
                    method_text_assembly += f"  movsd QWORD PTR [rbp {p_data['offset_rbp']}], {xmm_reg} # Spill float param {p_name}\n"
                    xmm_reg_idx += 1

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
        arg_expr_nodes = []
        format_types = [] # Track format type for each argument

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
                    # First, evaluate the exact return type of the method
                    # by looking it up in the class definitions
                    method_return_type = self.get_method_return_type(expr, context)
                    
                    if method_return_type == 'float':
                        format_string_parts.append("%g")  # %g for float/double
                        format_types.append("float")
                    elif method_return_type == 'string':
                        format_string_parts.append("%s")  # %s for string
                        format_types.append("string")
                    elif method_return_type == 'int':
                        format_string_parts.append("%d")  # %d for int
                        format_types.append("int")
                    else:
                        # If we couldn't determine the exact return type,
                        # fall back to the more general type detection
                        node_type_for_print = self.get_expr_type(expr, context)
                        if node_type_for_print == 'string':
                            format_string_parts.append("%s")
                            format_types.append("string")
                        elif node_type_for_print == 'float':
                            format_string_parts.append("%g")  # %g for float/double
                            format_types.append("float")
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
            elif isinstance(expr, FloatLiteralNode):
                format_string_parts.append("%g")  # %g for float/double
                arg_expr_nodes.append(expr)
                format_types.append("float")
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
                elif var_type == 'float':
                    format_string_parts.append("%g")  # %g for float/double
                    format_types.append("float")
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
                elif element_type_for_print == 'float':
                    format_string_parts.append("%g")  # %g for float/double
                    format_types.append("float")
                else: # Default to %d for int[], int, or other/unknown element types
                    format_string_parts.append("%d")
                    format_types.append("int")
                arg_expr_nodes.append(expr)
            elif isinstance(expr, ClassVarAccessNode):
                # Handle class variable access (this:var or obj::var)
                var_type = self.get_expr_type(expr, context)
                
                if var_type == 'string':
                    format_string_parts.append("%s")
                    format_types.append("string")
                elif var_type == 'float':
                    format_string_parts.append("%g")  # %g for float/double
                    format_types.append("float")
                else: # Default to %d for int or unknown/other types
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
                elif func_name == "to_stringf":  # to_stringf() returns string
                    format_string_parts.append("%s")
                    format_types.append("string")
                elif func_name == "to_intf":  # to_intf() returns int
                    format_string_parts.append("%d")
                    format_types.append("int")
                elif func_name == "open":  # open() returns file
                    format_string_parts.append("FILE*@%p")
                    format_types.append("pointer")
                elif func_name == "index":  # index() now returns string
                    format_string_parts.append("%s")
                    format_types.append("string")
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
                elif node_type_for_print == 'float':
                    format_string_parts.append("%g")  # %g for float/double
                    format_types.append("float")
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
        
        # Track how many float arguments we have for printf's vararg handling
        float_arg_count = 0
        for ft in format_types:
            if ft == "float":
                float_arg_count += 1
        
        if num_val_args > (len(printf_arg_regs) -1) :
            for i in range(num_val_args - 1, (len(printf_arg_regs) -1) - 1, -1):
                print_assembly += self.visit(arg_expr_nodes[i], context)
                # For float args beyond the first 6 args, need to push the xmm register content
                if i < len(format_types) and format_types[i] == "float":
                    print_assembly += "  sub rsp, 8             # Reserve space for float argument\n"
                    print_assembly += "  movsd [rsp], xmm0      # Store float argument\n"
                else:
                    print_assembly += "  push rax # Push printf stack argument\n"
                    
        for i in range(min(num_val_args, len(printf_arg_regs)-1) - 1, -1, -1):
            print_assembly += self.visit(arg_expr_nodes[i], context)
            target_reg = printf_arg_regs[i+1]
            
            # Special handling for char format (%c)
            if i < len(format_types) and format_types[i] == "char":
                # For %c, we need to ensure only the low byte is used
                print_assembly += f"  movzx {target_reg}, al # Load character (zero-extended) into {target_reg}\n"
            # Special handling for float format (%g)
            elif i < len(format_types) and format_types[i] == "float":
                # For %g, the float value is already in xmm0, and needs to stay there
                # In x86-64 ABI, floating point args go in xmm0-xmm7
                # We don't need to do anything here, as the value is already in the right register (xmm0)
                # Each xmm register corresponds to the argument position
                xmm_reg = f"xmm{i}"
                if i > 0:  # Only move if not already in xmm0
                    print_assembly += f"  movsd {xmm_reg}, xmm0  # Move float to {xmm_reg} for arg {i+1}\n"
            else:
                print_assembly += f"  mov {target_reg}, rax # Load printf arg into {target_reg}\n"
                
        print_assembly += f"  lea {printf_arg_regs[0]}, {format_label}[rip] # Format string for printf\n"
        # For printf with floating point args, RAX must contain the number of vector (xmm) registers used
        print_assembly += f"  mov rax, {float_arg_count} # Number of float/vector args for printf\n"
        print_assembly += "  call printf\n"
        if num_val_args > (len(printf_arg_regs) -1):
            bytes_pushed = (num_val_args - (len(printf_arg_regs) -1)) * 8
            print_assembly += f"  add rsp, {bytes_pushed} # Clean up printf stack arguments\n"
        return print_assembly

    def visit_ReturnNode(self, node: ReturnNode, context):
        """Handle a return statement with optional expression."""
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
            expr_code = self.visit(node.expression, context)
            if expr_code is None:
                raise CompilerError(f"Failed to generate code for return expression at L{node.lineno}")
            ret_assembly += expr_code
            
            # Handle special return types
            if return_type == "float":
                # For float returns, the value is already in XMM0 (no need to do anything)
                ret_assembly += "  # Float return value is already in XMM0\n"
            # For list or matrix returns
            elif return_type and (return_type.endswith('[]') or return_type.endswith('[][]')):
                ret_assembly += f"  # Returning {return_type} pointer in RAX\n"
        
        ret_assembly += f"  jmp {epilogue_label}\n"
        return ret_assembly

    def visit_IntegerLiteralNode(self, node: IntegerLiteralNode, context):
        return f"  mov rax, {node.value} # Integer literal {node.value}\n"
    def visit_FloatLiteralNode(self, node: 'FloatLiteralNode', context):
        # Load float literal into XMM0
        label = self.new_float_literal_label(node.value)
        code = f"  lea rax, {label}[rip] # Address of float literal {node.value}\n"
        code += f"  movsd xmm0, [rax] # Load double {node.value} into xmm0\n"
        return code
    def visit_StringLiteralNode(self, node: StringLiteralNode, context):
        label = self.new_string_label(node.value)
        return f"  lea rax, {label}[rip] # Address of string literal\n"
    def visit_TrueLiteralNode(self, node: TrueLiteralNode, context):
        return f"  mov rax, 1 # True literal\n"
    def visit_FalseLiteralNode(self, node: FalseLiteralNode, context):
        return f"  mov rax, 0 # False literal\n"
    def visit_IdentifierNode(self, node: IdentifierNode, context):
        var_name = node.value
        if var_name in self.current_method_params:
            param_info = self.current_method_params[var_name]
            offset = param_info['offset_rbp']
            
            # Check if this is a float variable
            if 'type' in param_info and param_info['type'] == 'float':
                # For float variables, load the value directly into XMM0
                return f"  movsd xmm0, QWORD PTR [rbp {offset}] # Load float param {var_name} directly from [rbp{offset}] into xmm0\n"
            else:
                return f"  mov rax, QWORD PTR [rbp {offset}] # Param {var_name} from [rbp{offset}]\n"
        
        elif var_name in self.current_method_locals:
            local_info = self.current_method_locals[var_name]
            offset = local_info['offset_rbp']
            
            # Check if this is a float variable
            if 'type' in local_info and local_info['type'] == 'float':
                # For float variables, load the value directly into XMM0
                return f"  movsd xmm0, QWORD PTR [rbp {offset}] # Load float local {var_name} directly from [rbp{offset}] into xmm0\n"
            else:
                return f"  mov rax, QWORD PTR [rbp {offset}] # Local var {var_name} from [rbp{offset}]\n"
        
        elif isinstance(node, IntegerLiteralNode):
            return "int"
        elif isinstance(node, TrueLiteralNode) or isinstance(node, FalseLiteralNode):
            return "bool" # Or could be "int" if booleans are treated as 0/1 integers

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
        var_type_str = local_info['type'] # e.g., "int", "int[]", "float"

        decl_assembly = f"  # Variable Declaration: {var_name} ({var_type_str}) at L{node.lineno}\n"
        
        # 1. Evaluate the right-hand side expression (initialization value)
        #    This will handle ListLiteralNode if used for initialization.
        decl_assembly += self.visit(node.expression, context) # Result will be in RAX or XMM0 for floats
        
        # 2. Store the result into the variable's pre-allocated stack slot
        if var_type_str == "float":
            # For float variables, store from XMM0
            decl_assembly += f"  movq QWORD PTR [rbp {var_offset_rbp}], xmm0 # Initialize float {var_name} at [rbp{var_offset_rbp}]\n"
        else:
            # For other types, store from RAX
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
                
                # Determine actual size based on class fields
                safe_class_name = class_name_for_new.replace('.', '_')
                object_size = 8  # Minimum size
                
                # Add space for class variables if they exist
                if safe_class_name in self.class_vars:
                    # Find the total size needed for all class variables
                    class_vars = self.class_vars[safe_class_name]
                    if class_vars:
                        # Calculate size based on the last variable's offset + 8 bytes
                        # (assuming all variables are 8 bytes)
                        last_var_offset = max(offset for offset, _, _ in class_vars.values())
                        object_size = last_var_offset + 8
                call_assembly += f"  mov rdi, {object_size}  # Size for malloc\n"
                call_assembly += "  call malloc          # Allocate memory, result in RAX\n"
                
                call_assembly += "  mov r15, rax         # Save new object pointer in R15 (consistent register)\n"

                # Initialize class variables if needed
                if safe_class_name in self.class_vars and self.class_vars[safe_class_name]:
                    # Use r15 for the object pointer during initialization
                    
                    # Find the class in the AST to get initialization expressions
                    class_init_exprs = {}
                    for cls in self.current_program_node.classes:
                        if cls.name.replace('.', '_') == safe_class_name:
                            for var_decl in cls.class_vars:
                                class_init_exprs[var_decl.name] = var_decl.expression
                            break

                    # Initialize all class variables
                    for var_name, (offset, var_type, _) in self.class_vars[safe_class_name].items():
                        if var_name in class_init_exprs and class_init_exprs[var_name] is not None:
                            # Initialize with the expression value
                            call_assembly += "  mov rdi, r15      # Pass object pointer to initialization context\n"
                            # Create a temporary context with the object as 'this'
                            temp_context = (safe_class_name, "__init__")
                            # Evaluate the initialization expression
                            call_assembly += self.visit(class_init_exprs[var_name], temp_context)
                            
                            # Store the result based on variable type
                            if var_type == "float":
                                # For float variables, store from XMM0
                                call_assembly += f"  movsd QWORD PTR [r15 + {offset}], xmm0  # Initialize float {var_name} with expression result\n"
                            else:
                                call_assembly += f"  mov QWORD PTR [r15 + {offset}], rax  # Initialize {var_name} with expression result\n"
                        else:
                            # Default initialization (0 for numbers, empty string for strings)
                            if var_type == "string":
                                # Empty string is represented as an empty string pointer
                                empty_str_label = self.new_string_label("")
                                call_assembly += f"  lea rax, {empty_str_label}[rip]  # Empty string for {var_name}\n"
                                call_assembly += f"  mov QWORD PTR [r15 + {offset}], rax  # Initialize {var_name} to empty string\n"
                            elif var_type == "float":
                                # Default float value 0.0
                                call_assembly += f"  pxor xmm0, xmm0  # Set float to 0.0\n"
                                call_assembly += f"  movsd QWORD PTR [r15 + {offset}], xmm0  # Initialize float {var_name} to 0.0\n"
                            else:
                                # Default to 0 for other types
                                call_assembly += f"  mov QWORD PTR [r15 + {offset}], 0  # Initialize {var_name} to 0\n"
                    
                    # No need to restore RAX here, r15 holds the object pointer
                
                # Check if the class has a 'new' method (constructor)
                target_class_node = None
                for cls in self.current_program_node.classes:
                    if cls.name.replace('.', '_') == safe_class_name:
                        target_class_node = cls
                        break

                has_constructor = False
                if target_class_node:
                    for method in target_class_node.methods:
                        if method.name == "new":
                            has_constructor = True
                            break

                if has_constructor:
                    # Call the constructor (the 'new' method) if it exists
                    # The newly allocated object pointer is in RAX. This will be the 'this' pointer for the constructor.
                    # r15 already holds the object pointer
                    
                    # Prepare arguments for the constructor call
                    # The first argument (RDI) will be the 'this' pointer (the new object)
                    # Subsequent arguments (RSI, RDX, etc.) are the arguments passed to ClassName.new(...)
                    
                    # Argument registers for the constructor call (excluding 'this' which is RDI)
                    constructor_arg_regs = ["rsi", "rdx", "rcx", "r8", "r9"]
                    num_constructor_args = len(node.arguments)
                    stack_arg_count_constructor = 0
                    if num_constructor_args > len(constructor_arg_regs):
                        stack_arg_count_constructor = num_constructor_args - len(constructor_arg_regs)

                    # Push stack arguments (if any), from right to left
                    for i in range(num_constructor_args - 1, len(constructor_arg_regs) - 1, -1):
                        # Check if this argument is a float before evaluating it
                        arg_type = self.get_expr_type(node.arguments[i], context)
                        call_assembly += self.visit(node.arguments[i], context)
                        if arg_type == "float":
                            # For float values, need to store the xmm0 register value
                            call_assembly += "  sub rsp, 8             # Reserve space for float argument\n"
                            call_assembly += "  movsd [rsp], xmm0      # Store float argument\n"
                        else:
                            call_assembly += "  push rax # Push constructor stack argument\n"

                    # Evaluate register arguments and push them temporarily to stack (right to left)
                    temp_pushes_for_reg_args_constructor = 0
                    for i in range(min(num_constructor_args, len(constructor_arg_regs)) - 1, -1, -1):
                        # Check if this argument is a float before evaluating it
                        arg_type = self.get_expr_type(node.arguments[i], context)
                        call_assembly += self.visit(node.arguments[i], context)
                        if arg_type == "float":
                            # For float values, need to store the xmm0 register value
                            call_assembly += "  sub rsp, 8             # Reserve space for float argument\n"
                            call_assembly += "  movsd [rsp], xmm0      # Store float argument\n"
                        else:
                            call_assembly += "  push rax # Temporarily store constructor reg argument\n"
                        temp_pushes_for_reg_args_constructor += 1
                    
                    # Set RDI to the new object pointer ('this')
                    call_assembly += "  mov rdi, r15 # Set RDI to the new object pointer ('this')\n"

                    # Track which arguments are floats for proper register loading
                    arg_types = []
                    for i in range(min(num_constructor_args, len(constructor_arg_regs))):
                        arg_types.append(self.get_expr_type(node.arguments[i], context))

                    # Count float arguments for X86-64 ABI
                    float_arg_count = 0
                    for arg_type in arg_types:
                        if arg_type == "float":
                            float_arg_count += 1

                    # Track which XMM registers are used for float arguments
                    xmm_reg_idx = 0

                    # Pop arguments into their designated registers.
                    for i in range(min(num_constructor_args, len(constructor_arg_regs))):
                        target_reg = constructor_arg_regs[i]
                        if i < len(arg_types) and arg_types[i] == "float":
                            # Handle float arguments - they go into XMM registers
                            xmm_reg = f"xmm{xmm_reg_idx}"
                            call_assembly += f"  movsd {xmm_reg}, [rsp] # Load float arg into {xmm_reg}\n"
                            call_assembly += "  add rsp, 8 # Clean up float arg space\n"
                            xmm_reg_idx += 1
                        else:
                            call_assembly += f"  pop {target_reg} # Load constructor arg into {target_reg}\n"
                    
                    alignment_padding_constructor = 0
                    if stack_arg_count_constructor > 0 and (stack_arg_count_constructor % 2 != 0):
                        call_assembly += "  sub rsp, 8 # Align stack for odd constructor stack args\n"
                        alignment_padding_constructor = 8

                    # For functions with floating-point args, RAX must contain the number of vector (XMM) registers used
                    if float_arg_count > 0:
                        call_assembly += f"  mov rax, {float_arg_count} # Number of float/vector args for constructor call\n"

                    # Call the constructor method (e.g., Vec2_new)
                    constructor_label = f"{safe_class_name}_new"
                    call_assembly += f"  call {constructor_label}\n"

                    if alignment_padding_constructor > 0:
                        call_assembly += f"  add rsp, {alignment_padding_constructor} # Restore stack alignment\n"

                    if stack_arg_count_constructor > 0:
                        call_assembly += f"  add rsp, {stack_arg_count_constructor * 8} # Clean up constructor stack arguments\n"
                    
                else:
                    call_assembly += f"  # No 'new' constructor found for class {class_name_for_new}. Skipping constructor call.\n"
                
                call_assembly += "  mov rax, r15         # Return new object pointer in RAX as the result of .new()\n"
                return call_assembly
            
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
            # Check if this argument is a float before evaluating it
            arg_type = self.get_expr_type(node.arguments[i], context)
            call_assembly += self.visit(node.arguments[i], context) 
            if arg_type == "float":
                # For float values, need to store the xmm0 register value
                call_assembly += "  sub rsp, 8             # Reserve space for float argument\n"
                call_assembly += "  movsd [rsp], xmm0      # Store float argument\n"
            else:
                call_assembly += "  push rax # Push stack argument\n"

        # Evaluate register arguments and push them temporarily to stack (right to left)
        temp_pushes_for_reg_args = 0
        for i in range(min(num_explicit_args_to_pass, len(explicit_arg_regs)) - 1, -1, -1):
            # Check if this argument is a float before evaluating it
            arg_type = self.get_expr_type(node.arguments[i], context)
            call_assembly += self.visit(node.arguments[i], context)
            if arg_type == "float":
                # For float values, need to store the xmm0 register value
                call_assembly += "  sub rsp, 8             # Reserve space for float argument\n"
                call_assembly += "  movsd [rsp], xmm0      # Store float argument\n"
            else:
                call_assembly += "  push rax # Temporarily store reg argument\n"
            temp_pushes_for_reg_args += 1

        # Setup RDI if it's an instance call on an object variable (already handled for 'this' calls conceptually)
        # For static calls, RDI is the first explicit argument, so no special setup here.
        if not is_static_call and not is_call_on_this:
             call_assembly += object_address_in_rdi_setup # This loads object address into RDI
        elif is_call_on_this:
             call_assembly += object_address_in_rdi_setup # This is just a comment for 'this' calls
        # For static calls, RDI will be populated by the first argument pop if any, or is not used if no args.

        # Track which arguments are floats for proper register loading
        arg_types = []
        for i in range(min(num_explicit_args_to_pass, len(explicit_arg_regs))):
            arg_types.append(self.get_expr_type(node.arguments[i], context))

        # Count float arguments for X86-64 ABI
        float_arg_count = 0
        for arg_type in arg_types:
            if arg_type == "float":
                float_arg_count += 1

        # Track which XMM registers are used for float arguments
        xmm_reg_idx = 0

        # Pop arguments into their designated registers.
        for i in range(min(num_explicit_args_to_pass, len(explicit_arg_regs))):
            target_reg = explicit_arg_regs[i]
            if i < len(arg_types) and arg_types[i] == "float":
                # Handle float arguments - they go into XMM registers
                xmm_reg = f"xmm{xmm_reg_idx}"
                call_assembly += f"  movsd {xmm_reg}, [rsp] # Load float arg into {xmm_reg}\n"
                call_assembly += "  add rsp, 8 # Clean up float arg space\n"
                xmm_reg_idx += 1
            else:
                call_assembly += f"  pop {target_reg} # Load arg into {target_reg}\n"
        
        # If it's a static call and RDI was not used by an argument, it remains as is (not 'this')
        # If it's an instance call on 'this', RDI was already 'this'.
        # If it's an instance call on object var, RDI was set by object_address_in_rdi_setup.

        alignment_padding = 0
        if stack_arg_count > 0 and (stack_arg_count % 2 != 0):
            call_assembly += "  sub rsp, 8 # Align stack for odd stack args\n"
            alignment_padding = 8

        # For functions with floating-point args, RAX must contain the number of vector (XMM) registers used
        if float_arg_count > 0:
            call_assembly += f"  mov rax, {float_arg_count} # Number of float/vector args for call\n"

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
        elif node.name == "show":
            # Built-in show function - like print but without a newline
            code = f"  # show() call at L{node.name_token.lineno}\n"
            format_string_parts = []
            arg_expr_nodes = []
            format_types = []

            for i, expr in enumerate(node.arguments):
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
                        elif node_type_for_print == 'float':
                            format_string_parts.append("%g")  # %g for float/double
                            format_types.append("float")
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
                elif isinstance(expr, FloatLiteralNode):
                    format_string_parts.append("%g")  # %g for float/double
                    arg_expr_nodes.append(expr)
                    format_types.append("float")
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
                    elif var_type == 'float':
                        format_string_parts.append("%g")  # %g for float/double
                        format_types.append("float")
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
                    elif element_type_for_print == 'float':
                        format_string_parts.append("%g")  # %g for float/double
                        format_types.append("float")
                    else: # Default to %d for int[], int, or other/unknown element types
                        format_string_parts.append("%d")
                        format_types.append("int")
                    arg_expr_nodes.append(expr)
                elif isinstance(expr, ClassVarAccessNode):
                    # Handle class variable access (this:var or obj::var)
                    var_type = self.get_expr_type(expr, context)
                    
                    if var_type == 'string':
                        format_string_parts.append("%s")
                        format_types.append("string")
                    elif var_type == 'float':
                        format_string_parts.append("%g")  # %g for float/double
                        format_types.append("float")
                    else: # Default to %d for int or unknown/other types
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
                    elif func_name == "to_stringf":  # to_stringf() returns string
                        format_string_parts.append("%s")
                        format_types.append("string")
                    elif func_name == "to_intf":  # to_intf() returns int
                        format_string_parts.append("%d")
                        format_types.append("int")
                    elif func_name == "open":  # open() returns file
                        format_string_parts.append("FILE*@%p")
                        format_types.append("pointer")
                    elif func_name == "index":  # index() now returns string
                        format_string_parts.append("%s")
                        format_types.append("string")
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
                    elif node_type_for_print == 'float':
                        format_string_parts.append("%g")  # %g for float/double
                        format_types.append("float")
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
                    raise CompilerError(f"Cannot show expression of type {type(expr)} at L{node.name_token.lineno}")
            
            format_string = "".join(format_string_parts)
            # Unlike print(), show() doesn't add a newline
            format_label = self.new_printf_format_label(format_string)
            printf_arg_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
            num_val_args = len(arg_expr_nodes)
            
            # Track how many float arguments we have for printf's vararg handling
            float_arg_count = 0
            for ft in format_types:
                if ft == "float":
                    float_arg_count += 1
                    
            if num_val_args > (len(printf_arg_regs) -1) :
                for i in range(num_val_args - 1, (len(printf_arg_regs) -1) - 1, -1):
                    code += self.visit(arg_expr_nodes[i], context)
                    # For float args beyond the first 6 args, need to push the xmm register content
                    if i < len(format_types) and format_types[i] == "float":
                        code += "  sub rsp, 8             # Reserve space for float argument\n"
                        code += "  movsd [rsp], xmm0      # Store float argument\n"
                    else:
                        code += "  push rax # Push printf stack argument\n"
                        
            for i in range(min(num_val_args, len(printf_arg_regs)-1) - 1, -1, -1):
                code += self.visit(arg_expr_nodes[i], context)
                target_reg = printf_arg_regs[i+1]
                
                # Special handling for char format (%c)
                if i < len(format_types) and format_types[i] == "char":
                    # For %c, we need to ensure only the low byte is used
                    code += f"  movzx {target_reg}, al # Load character (zero-extended) into {target_reg}\n"
                # Special handling for float format (%g)
                elif i < len(format_types) and format_types[i] == "float":
                    # For %g, the float value is already in xmm0, and needs to stay there
                    # In x86-64 ABI, floating point args go in xmm0-xmm7
                    # We don't need to do anything here, as the value is already in the right register (xmm0)
                    # Each xmm register corresponds to the argument position
                    xmm_reg = f"xmm{i}"
                    if i > 0:  # Only move if not already in xmm0
                        code += f"  movsd {xmm_reg}, xmm0  # Move float to {xmm_reg} for arg {i+1}\n"
                else:
                    code += f"  mov {target_reg}, rax # Load printf arg into {target_reg}\n"
                    
            code += f"  lea {printf_arg_regs[0]}, {format_label}[rip] # Format string for printf\n"
            # For printf with floating point args, RAX must contain the number of vector (xmm) registers used
            code += f"  mov rax, {float_arg_count} # Number of float/vector args for printf\n"
            code += "  call printf\n"
            if num_val_args > (len(printf_arg_regs) -1):
                bytes_pushed = (num_val_args - (len(printf_arg_regs) -1)) * 8
                code += f"  add rsp, {bytes_pushed} # Clean up printf stack arguments\n"
            return code
        elif node.name == "exec":
            # exec(command, mode) - Execute a terminal command
            if len(node.arguments) != 2:
                raise CompilerError(f"exec() expects exactly two arguments (command, mode), got {len(node.arguments)} L{node.name_token.lineno}")
            
            code = f"  # exec() call at L{node.name_token.lineno}\n"
            
            # Generate a unique label for this exec call to avoid conflicts
            if not hasattr(self, 'exec_counter'):
                self.exec_counter = 0
            exec_id = self.exec_counter
            self.exec_counter += 1
            
            # Get the command string
            code += self.visit(node.arguments[0], context)
            code += "  push rax # Save command string\n"
            
            # Get the mode (0=visible, 1=return output, 2=return exit code)
            code += self.visit(node.arguments[1], context)
            code += "  mov r12, rax # Save mode in R12\n"
            
            # Restore command string
            code += "  pop rdi # Command string in RDI\n"
            
            # Check for 'cd' command - special handling for directory changes
            code += "  # Check if the command starts with 'cd '\n"
            code += "  push rdi # Save command string\n"
            # Check first char is 'c'
            code += "  movzx rax, BYTE PTR [rdi]\n"
            code += "  cmp rax, 'c'\n"
            code += f"  jne .Lnot_cd_cmd_{exec_id}\n"
            # Check second char is 'd'
            code += "  movzx rax, BYTE PTR [rdi+1]\n"
            code += "  cmp rax, 'd'\n"
            code += f"  jne .Lnot_cd_cmd_{exec_id}\n"
            # Check third char is space
            code += "  movzx rax, BYTE PTR [rdi+2]\n"
            code += "  cmp rax, ' '\n"
            code += f"  jne .Lnot_cd_cmd_{exec_id}\n"
            
            # It's a cd command, extract the path (skip 'cd ' prefix)
            code += "  # Handle 'cd' command by changing Pangy process directory\n"
            code += "  lea rdi, [rdi+3] # Skip 'cd ' prefix\n"
            code += "  call chdir # Change directory\n"
            code += "  xor rax, rax # Return 0 (success)\n"
            code += "  pop rdi # Restore and discard command string\n"
            code += f"  jmp .Lexec_end_{exec_id}\n"
            
            code += f".Lnot_cd_cmd_{exec_id}:\n"
            code += "  pop rdi # Restore command string\n"
            
            # Check the mode
            code += "  cmp r12, 0 # Mode 0: Execute visibly without return\n"
            code += f"  je .Lexec_mode0_{exec_id}\n"
            code += "  cmp r12, 1 # Mode 1: Execute in background, return output\n"
            code += f"  je .Lexec_mode1_{exec_id}\n"
            code += "  cmp r12, 2 # Mode 2: Execute in background, return exit code\n"
            code += f"  je .Lexec_mode2_{exec_id}\n"
            
            # Invalid mode
            error_msg = self.new_string_label("Error: Invalid mode for exec(). Use 0, 1, or 2.\n")
            code += f"  lea rdi, {error_msg}[rip]\n"
            code += "  call printf\n"
            code += "  mov rdi, 1\n"
            code += "  call exit\n"
            
            # Mode 0: Just execute the command using system()
            code += f".Lexec_mode0_{exec_id}:\n"
            code += "  call system # Execute command using system()\n"
            code += "  xor rax, rax # Return void (0)\n"
            code += f"  jmp .Lexec_end_{exec_id}\n"
            
            # Mode 1: Execute in background and return output as string
            code += f".Lexec_mode1_{exec_id}:\n"
            # Open pipe to command using popen()
            mode_str = self.new_string_label("r") # Read mode for popen
            code += "  mov rsi, rdi # Save command string in RSI temporarily\n"
            code += f"  lea rdi, {mode_str}[rip] # Mode 'r' for reading\n"
            code += "  mov rdx, rsi # Command string to RDX\n"
            code += "  mov rsi, rdi # Mode string to RSI\n"
            code += "  mov rdi, rdx # Command string to RDI\n"
            code += "  call popen # Open pipe to command\n"
            code += "  mov r13, rax # Save FILE* pointer in R13\n"
            
            # Check if popen failed (returned NULL)
            code += "  test r13, r13\n"
            code += f"  jnz .Lpopen_success_{exec_id}\n"
            
            # Handle popen error
            error_msg = self.new_string_label("Error: Failed to execute command with popen()\n")
            code += f"  lea rdi, {error_msg}[rip]\n"
            code += "  call printf\n"
            code += "  mov rdi, 1\n"
            code += "  call exit\n"
            
            # If popen succeeded
            code += f".Lpopen_success_{exec_id}:\n"
            
            # Allocate buffer for output (initial 4096 bytes)
            code += "  mov rdi, 4096 # Initial buffer size\n" 
            code += "  call malloc\n"
            code += "  mov r14, rax # Buffer in R14\n"
            
            # Initialize buffer counters
            code += "  mov r15, 0 # Current position in buffer\n"
            code += "  mov rbx, 4096 # Current buffer size\n"
            
            # Start reading loop
            code += f".Lread_loop_{exec_id}:\n"
            
            # Calculate buffer position for this read
            code += "  lea rdi, [r14 + r15] # Current position in buffer\n"
            
            # Remaining space
            code += "  mov rsi, rbx # Buffer size\n"
            code += "  sub rsi, r15 # Subtract current position to get remaining space\n"
            code += "  dec rsi # Leave space for null terminator\n"
            
            # Check if we need to expand the buffer
            code += "  cmp rsi, 10 # If less than 10 bytes remain\n"
            code += f"  jg .Lbuffer_ok_{exec_id}\n"
            
            # Expand buffer
            code += "  mov rdi, rbx # Current size\n"
            code += "  shl rdi, 1 # Double the size\n"
            code += "  mov rcx, rdi # Save new size in RCX\n"
            code += "  mov rdi, r14 # Current buffer\n"
            code += "  mov rsi, rcx # New size\n"
            code += "  call realloc\n"
            code += "  mov r14, rax # Update buffer pointer\n"
            code += "  mov rbx, rcx # Update buffer size\n"
            
            # Recalculate buffer position and space
            code += "  lea rdi, [r14 + r15] # Current position in buffer\n"
            code += "  mov rsi, rbx # Buffer size\n"
            code += "  sub rsi, r15 # Remaining space\n"
            code += "  dec rsi # Leave space for null terminator\n"
            
            code += f".Lbuffer_ok_{exec_id}:\n"
            # Read a line from the pipe using fgets
            code += "  mov rdx, r13 # FILE* stream\n"
            code += "  call fgets # Read a line into buffer at current position\n"
            
            # Check if we've reached EOF (fgets returns NULL)
            code += "  test rax, rax\n"
            code += f"  jz .Lread_done_{exec_id}\n"
            
            # Update position by finding the length of what we just read
            code += "  mov rdi, rax # Result from fgets (pointer to buffer we passed)\n"
            code += "  call strlen\n"
            code += "  add r15, rax # Update position\n"
            
            # Continue reading
            code += f"  jmp .Lread_loop_{exec_id}\n"
            
            # Reading complete
            code += f".Lread_done_{exec_id}:\n"
            
            # Null-terminate the buffer
            code += "  mov byte ptr [r14 + r15], 0\n"
            
            # Close the pipe
            code += "  mov rdi, r13 # FILE* pointer\n"
            code += "  call pclose\n"
            
            # Return the buffer containing output
            code += "  mov rax, r14\n"
            code += f"  jmp .Lexec_end_{exec_id}\n"
            
            # Mode 2: Execute in background and return exit code
            code += f".Lexec_mode2_{exec_id}:\n"
            # Open pipe to command using popen()
            mode_str = self.new_string_label("r") # Read mode for popen
            code += "  mov rsi, rdi # Save command string in RSI temporarily\n"
            code += f"  lea rdi, {mode_str}[rip] # Mode 'r' for reading\n"
            code += "  mov rdx, rsi # Command string to RDX\n"
            code += "  mov rsi, rdi # Mode string to RSI\n"
            code += "  mov rdi, rdx # Command string to RDI\n"
            code += "  call popen # Open pipe to command\n"
            code += "  mov r13, rax # Save FILE* pointer in R13\n"
            
            # Check if popen failed (returned NULL)
            code += "  test r13, r13\n"
            code += f"  jnz .Lpopen_exit_success_{exec_id}\n"
            
            # Handle popen error
            error_msg = self.new_string_label("Error: Failed to execute command with popen()\n")
            code += f"  lea rdi, {error_msg}[rip]\n"
            code += "  call printf\n"
            code += "  mov rdi, 1\n"
            code += "  call exit\n"
            
            # If popen succeeded
            code += f".Lpopen_exit_success_{exec_id}:\n"
            
            # Allocate a small buffer to discard output
            code += "  mov rdi, 4096 # Buffer size\n"
            code += "  call malloc\n"
            code += "  mov r14, rax # Buffer in R14\n"
            
            # Read all output and discard
            code += f".Lread_discard_loop_{exec_id}:\n"
            code += "  mov rdi, r14 # Buffer\n"
            code += "  mov rsi, 4096 # Buffer size\n"
            code += "  mov rdx, r13 # FILE* stream\n"
            code += "  call fgets\n"
            code += "  test rax, rax\n"
            code += f"  jnz .Lread_discard_loop_{exec_id}\n"
            
            # Free the discard buffer
            code += "  mov rdi, r14 # Buffer\n"
            code += "  call free\n"
            
            # Close pipe and get exit status
            code += "  mov rdi, r13 # FILE* pointer\n"
            code += "  call pclose # Returns exit status in RAX\n"
            code += "  mov rcx, rax # Copy status to RCX\n"
            
            # Extract exit code from status
            # In normal Unix wait status, the exit code is in the high byte
            # Handle cases where command might have been terminated by a signal
            code += "  shr rcx, 8 # Exit code in high byte, shift right\n"
            code += "  and rcx, 0xFF # Mask to get just the exit code byte\n"
            code += "  mov rax, rcx # Move to RAX as the return value\n"
            
            # End of exec function
            code += f".Lexec_end_{exec_id}:\n"
            
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
            index_id = self.next_index_label_id
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
            code += f"  jne .L_index_not_null_{index_id}\n"
            
            # Handle null string error
            error_msg = self.new_string_label("Error: Cannot index a null string.\n")
            code += f"  lea rdi, {error_msg}[rip]\n"
            code += "  call printf\n"
            code += "  mov rdi, 1\n"
            code += "  call exit\n"
            
            code += f".L_index_not_null_{index_id}:\n"
            
            # Get string length using strlen for bounds check
            if "strlen" not in self.externs:
                self.externs.append("strlen")
            code += "  mov rdi, r12 # String pointer for strlen\n"
            code += "  call strlen # Get string length\n"
            code += "  mov r14, rax # Save string length to r14\n"
            
            # Bounds check
            code += "  cmp r13, 0 # Check if index < 0\n"
            code += f"  jl .L_index_out_of_bounds_{index_id}\n"
            code += "  cmp r13, r14 # Check if index >= length\n"
            code += f"  jge .L_index_out_of_bounds_{index_id}\n"
            code += f"  jmp .L_index_in_bounds_{index_id}\n"
            
            # Handle out of bounds error
            code += f".L_index_out_of_bounds_{index_id}:\n"
            error_msg = self.new_string_label("Error: String index out of bounds.\n")
            code += f"  lea rdi, {error_msg}[rip]\n"
            code += "  call printf\n"
            code += "  mov rdi, 1\n"
            code += "  call exit\n"
            
            # Get character at index
            code += f".L_index_in_bounds_{index_id}:\n"
            code += "  mov rax, r12 # String pointer\n"
            code += "  add rax, r13 # Add index to get character address\n"
            code += "  movzx rax, BYTE PTR [rax] # Load character ASCII value (zero-extended to 64 bits)\n"
            
            # RAX holds the ASCII value of the character.
            # Convert this to a 2-byte string (char + null).
            code += "  # Convert char ASCII value in RAX to a 1-char string\n"
            code += "  push rax             # Save ASCII value on stack\n"
            code += "  mov rdi, 2           # Request 2 bytes for malloc (char + NULL)\n"
            code += "  call malloc          # Allocate memory, new string ptr in RAX\n"
            code += "  mov rbx, rax         # Save new string ptr in RBX\n"
            code += "  pop rcx              # Restore ASCII value to RCX (cl is the byte)\n"
            code += "  mov byte ptr [rbx], cl # Store the character\n"
            code += "  mov byte ptr [rbx+1], 0  # Null-terminate the string\n"
            code += "  mov rax, rbx         # Return new string ptr in RAX\n"

            return code
        elif node.name == "append":
            if len(node.arguments) != 2:
                raise CompilerError(f"append() expects two arguments (list, value), got {len(node.arguments)} at L{node.name_token.lineno}")

            # Use a unique ID for this append call
            append_id = self.next_append_label_id
            self.next_append_label_id += 1

            code = f"  # append(list, value) call at L{node.name_token.lineno}\n"
            # Evaluate list pointer (arg 0)
            code += self.visit(node.arguments[0], context) # List pointer in RAX
            code += "  push rax # Save list_ptr on stack (as it might change due to realloc)\n"
            
            # Check if we're appending a float value
            value_type = self.get_expr_type(node.arguments[1], context)
            is_float_append = (value_type == "float")
            
            # Evaluate value (arg 1)
            code += self.visit(node.arguments[1], context) # Value in RAX (or XMM0 for floats)
            
            if is_float_append:
                # For float values, we need to allocate memory to store a copy of the float
                code += "  sub rsp, 8              # Reserve space for float value\n"
                code += "  movsd [rsp], xmm0       # Save float value on stack temporarily\n"
                code += "  mov rdi, 8              # Allocate 8 bytes for the float\n"
                code += "  call malloc             # Allocate memory for float value\n"
                code += "  movsd xmm0, [rsp]       # Restore float value\n"
                code += "  movsd [rax], xmm0       # Store float value in allocated memory\n"
                code += "  mov rsi, rax            # RSI = pointer to float value\n"
                code += "  add rsp, 8              # Clean up stack\n"
                code += "  pop rdi                 # List pointer in RDI\n"
            else:
                code += "  push rax # Save value on stack\n"
                code += "  # --- APPEND IMPLEMENTATION ---\n"
                code += "  pop rsi    # Value to append is now in RSI\n"
                code += "  pop rdi    # List pointer is now in RDI\n"

            # Check if list_ptr (RDI) is null (meaning uninitialized list)
            null_list_label = f".L_append_null_list_{append_id}"
            not_null_list_label = f".L_append_not_null_list_{append_id}"
            code += "  cmp rdi, 0\n"
            code += f"  je {null_list_label}\n"

            # List is not null, proceed with normal append logic
            code += f"{not_null_list_label}:\n"
            code += "  mov r12, QWORD PTR [rdi]     # r12 = capacity = list_ptr[0]\n"
            code += "  mov r13, QWORD PTR [rdi + 8] # r13 = length   = list_ptr[1]\n"
            code += "  cmp r13, r12                 # if length >= capacity\n"
            code += f"  jl .L_append_has_space_{append_id}\n"

            code += "  # No space, reallocate\n"
            code += "  mov rax, r12                 # current capacity in RAX\n"
            code += "  test rax, rax                # Check if capacity is 0\n"
            code += f"  jnz .L_append_double_cap_{append_id}\n"
            code += "  mov r12, 8                   # If capacity was 0, set new capacity to 8 (e.g.)\n"
            code += f"  jmp .L_append_set_new_cap_{append_id}\n"
            code += f".L_append_double_cap_{append_id}:\n"
            code += "  shl r12, 1                   # new_capacity = capacity * 2\n"
            code += f".L_append_set_new_cap_{append_id}:\n"
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

            code += f".L_append_has_space_{append_id}:\n"
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
            code += f"  jmp .L_append_end_{append_id}\n"

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
            code += f"  jmp .L_append_has_space_{append_id} # This will add the first element\n"

            code += f".L_append_end_{append_id}:\n"
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
            
            # Use a unique ID for pop operations
            pop_id = self.next_pop_label_id
            self.next_pop_label_id += 1
            
            code = f"  # pop() call at L{node.name_token.lineno}\n"
            
            # Evaluate list pointer (arg 0)
            code += self.visit(node.arguments[0], context) # List pointer in RAX
            
            # Get list pointer in RDI
            code += "  mov rdi, rax # List pointer in RDI\n"
            
            # Check if list_ptr (RDI) is null (meaning uninitialized list)
            code += "  cmp rdi, 0 # Check for null list\n"
            code += f"  jne .L_pop_not_null_{pop_id}\n"
            
            # Handle null list error
            error_msg = self.new_string_label("Error: Cannot pop from a null list.\n")
            code += f"  lea rdi, {error_msg}[rip]\n"
            code += "  call printf\n"
            code += "  mov rdi, 1\n"
            code += "  call exit\n"
            
            code += f".L_pop_not_null_{pop_id}:\n"
            
            # Get list length and check if it's empty
            code += "  mov r12, QWORD PTR [rdi + 8] # r12 = length\n"
            code += "  cmp r12, 0 # Check if list is empty\n"
            code += f"  jne .L_pop_not_empty_{pop_id}\n"
            
            # Handle empty list error
            error_msg = self.new_string_label("Error: Cannot pop from an empty list.\n")
            code += f"  lea rdi, {error_msg}[rip]\n"
            code += "  call printf\n"
            code += "  mov rdi, 1\n"
            code += "  call exit\n"
            
            code += f".L_pop_not_empty_{pop_id}:\n"
            
            # Get the last element
            code += "  mov r12, QWORD PTR [rdi + 8] # r12 = length\n"
            code += "  dec r12 # r12 = length - 1 (last element index)\n"
            code += "  imul r13, r12, 8 # r13 = last_index * 8 (offset in bytes)\n"
            code += "  add r13, 16 # r13 = header_size + element_offset\n"
            
            # Determine the element type for appropriate return handling
            element_type = self.get_array_element_type(node.arguments[0], context)
            
            # Get the value at the last element position
            if element_type == "float":
                # For float lists, we need to handle the float value properly
                code += "  mov rax, QWORD PTR [rdi + r13] # rax = pointer to float value\n"
                code += "  push rax # Save the popped float pointer\n"
                
                # Update length
                code += "  mov QWORD PTR [rdi + 8], r12 # Update length = length - 1\n"
                
                # Restore float value and load it into XMM0
                code += "  pop rax # Restore the popped float pointer\n"
                code += "  movsd xmm0, QWORD PTR [rax] # Load float value into XMM0\n"
            else:
                # For regular (non-float) values
                code += "  mov rax, QWORD PTR [rdi + r13] # rax = last element value\n"
                code += "  push rax # Save the popped value to return later\n"
                
                # Update length
                code += "  mov QWORD PTR [rdi + 8], r12 # Update length = length - 1\n"
                
                # Restore return value
                code += "  pop rax # Restore the popped value to return\n"
            
            return code

        elif node.name == "insert":
            if len(node.arguments) != 3:
                raise CompilerError(f"insert() expects three arguments (list, index, value), got {len(node.arguments)} at L{node.name_token.lineno}")
            
            # Use a unique ID for insert operations
            if not hasattr(self, 'next_insert_label_id'):
                self.next_insert_label_id = 0
            insert_id = self.next_insert_label_id
            self.next_insert_label_id += 1
            
            code = f"  # insert(list, index, value) call at L{node.name_token.lineno}\n"
            
            # Evaluate list pointer (arg 0)
            code += self.visit(node.arguments[0], context) # List pointer in RAX
            code += "  push rax # Save list_ptr on stack (as it might change due to realloc)\n"
            
            # Evaluate index (arg 1)
            code += self.visit(node.arguments[1], context) # Index in RAX
            code += "  push rax # Save index on stack\n"
            
            # Check if we're inserting a float value
            value_type = self.get_expr_type(node.arguments[2], context)
            is_float_insert = (value_type == "float")
            
            # Evaluate value (arg 2)
            code += self.visit(node.arguments[2], context) # Value in RAX or XMM0 for floats
            
            if is_float_insert:
                # For float values, we need to allocate memory to store a copy of the float
                code += "  sub rsp, 8              # Reserve space for float value\n"
                code += "  movsd [rsp], xmm0       # Save float value on stack temporarily\n"
                code += "  mov rdi, 8              # Allocate 8 bytes for the float\n"
                code += "  call malloc             # Allocate memory for float value\n"
                code += "  movsd xmm0, [rsp]       # Restore float value\n"
                code += "  movsd [rax], xmm0       # Store float value in allocated memory\n"
                code += "  mov rdx, rax            # RDX = pointer to float value\n"
                code += "  add rsp, 8              # Clean up stack\n"
                code += "  pop rsi                 # Index in RSI\n"
                code += "  pop rdi                 # List pointer in RDI\n"
            else:
                code += "  push rax # Save value on stack\n"
                code += "  # --- INSERT IMPLEMENTATION ---\n"
                code += "  pop rdx    # Value to insert is now in RDX\n"
                code += "  pop rsi    # Index to insert at is now in RSI\n"
                code += "  pop rdi    # List pointer is now in RDI\n"
            
            # Check if list_ptr (RDI) is null (meaning uninitialized list)
            null_list_label = f".L_insert_null_list_{insert_id}"
            not_null_list_label = f".L_insert_not_null_list_{insert_id}"
            code += "  cmp rdi, 0 # Check for null list\n"
            code += f"  je {null_list_label}\n"
            
            # List is not null, proceed with normal insert logic
            code += f"{not_null_list_label}:\n"
            code += "  mov r12, QWORD PTR [rdi]     # r12 = capacity = list_ptr[0]\n"
            code += "  mov r13, QWORD PTR [rdi + 8] # r13 = length = list_ptr[1]\n"
            
            # Check if index is in bounds
            code += "  cmp rsi, 0 # Check if index < 0\n"
            code += f"  jl .L_insert_out_of_bounds_{insert_id}\n"
            code += "  cmp rsi, r13 # Check if index > length\n"
            code += f"  jg .L_insert_out_of_bounds_{insert_id}\n"
            code += f"  jmp .L_insert_in_bounds_{insert_id}\n"
            
            # Handle out of bounds error
            code += f".L_insert_out_of_bounds_{insert_id}:\n"
            error_msg = self.new_string_label("Error: List insert index out of bounds.\n")
            code += f"  lea rdi, {error_msg}[rip]\n"
            code += "  call printf\n"
            code += "  mov rdi, 1\n"
            code += "  call exit\n"
            
            # Insert within bounds
            code += f".L_insert_in_bounds_{insert_id}:\n"
            
            # Check if we need to resize the list
            code += "  cmp r13, r12 # if length >= capacity\n"
            code += f"  jl .L_insert_has_space_{insert_id}\n"
            
            # No space, reallocate
            code += "  # Need to resize list\n"
            code += "  mov rax, r12 # current capacity in RAX\n"
            code += "  test rax, rax # Check if capacity is 0\n"
            code += f"  jnz .L_insert_double_cap_{insert_id}\n"
            code += "  mov r12, 8 # If capacity was 0, set new capacity to 8\n"
            code += f"  jmp .L_insert_set_new_cap_{insert_id}\n"
            code += f".L_insert_double_cap_{insert_id}:\n"
            code += "  shl r12, 1 # new_capacity = capacity * 2\n"
            code += f".L_insert_set_new_cap_{insert_id}:\n"
            
            # Reallocate the list
            code += "  push rsi # Save index (RSI) before realloc\n"
            code += "  push rdx # Save value (RDX) before realloc\n"
            code += "  mov rsi, r12 # rsi = new_capacity (for element storage)\n"
            code += "  imul rsi, 8 # rsi = new_capacity * 8 (bytes for elements)\n"
            code += "  add rsi, 16 # rsi = total new size (header + elements)\n"
            # RDI still holds old list_ptr for realloc
            code += "  call realloc # rax = realloc(old_list_ptr, total_new_size)\n"
            code += "  pop rdx # Restore value (RDX)\n"
            code += "  pop rsi # Restore index (RSI)\n"
            code += "  mov rdi, rax # Update list_ptr with realloc result\n"
            code += "  mov QWORD PTR [rdi], r12 # list_ptr[0] = new_capacity\n"
            # R13 (length) is still correct
            
            # Now we have enough space for the insert
            code += f".L_insert_has_space_{insert_id}:\n"
            
            # Save registers we'll need
            code += "  push rdi # Save list pointer\n"
            code += "  push rsi # Save index\n"
            code += "  push rdx # Save value\n"
            code += "  push r13 # Save length\n"
            
            # Shift elements to make room for the new one
            # Start from the end of the list and move each element one position forward
            code += "  mov rcx, r13 # rcx = length (counter for loop)\n"
            code += "  cmp rcx, rsi # Compare length with insertion index\n"
            code += f"  je .L_insert_no_shift_{insert_id} # If inserting at the end, no need to shift\n"
            
            # Need to shift elements
            code += f".L_insert_shift_loop_{insert_id}:\n"
            code += "  dec rcx # rcx = current position (starting from length-1 down to index)\n"
            code += "  cmp rcx, rsi # Compare current position with insertion index\n"
            code += f"  jl .L_insert_shift_done_{insert_id} # If we've gone past the index, we're done shifting\n"
            
            # Shift element at position rcx to position rcx+1
            code += "  imul r8, rcx, 8 # r8 = current_pos * 8 (offset in bytes)\n"
            code += "  add r8, 16 # r8 = header_size + element_offset for source\n"
            code += "  mov r9, r8 # r9 = source offset\n"
            code += "  add r9, 8 # r9 = destination offset (source + 8 bytes)\n"
            code += "  mov r10, QWORD PTR [rdi + r8] # r10 = element value at current position\n"
            code += "  mov QWORD PTR [rdi + r9], r10 # Move element one position forward\n"
            
            code += f"  jmp .L_insert_shift_loop_{insert_id} # Continue shifting loop\n"
            
            # Shifting done, now insert the new element
            code += f".L_insert_shift_done_{insert_id}:\n"
            code += f".L_insert_no_shift_{insert_id}:\n"
            
            # Restore saved registers
            code += "  pop r13 # Restore length\n"
            code += "  pop rdx # Restore value\n"
            code += "  pop rsi # Restore index\n"
            code += "  pop rdi # Restore list pointer\n"
            
            # Insert the value at the specified index
            code += "  imul r8, rsi, 8 # r8 = index * 8 (offset in bytes)\n"
            code += "  add r8, 16 # r8 = header_size + element_offset\n"
            code += "  mov QWORD PTR [rdi + r8], rdx # list_ptr[index] = value\n"
            
            # Increment length
            code += "  inc r13 # length++\n"
            code += "  mov QWORD PTR [rdi + 8], r13 # list_ptr[1] = new_length\n"
            
            # Return the (potentially new) list pointer in RAX
            code += "  mov rax, rdi # Return list_ptr in RAX\n"
            code += f"  jmp .L_insert_end_{insert_id}\n"
            
            # Handle null list case: allocate initial list
            code += f"{null_list_label}:\n"
            code += "  # Create a new list for null list case\n"
            code += "  mov r12, 8 # Initial capacity = 8\n"
            code += "  mov r13, 0 # Initial length = 0\n"
            code += "  push rsi # Save index (RSI) before malloc\n"
            code += "  push rdx # Save value (RDX) before malloc\n"
            code += "  mov rdi, r12 # rdi = capacity (for element storage)\n"
            code += "  imul rdi, 8 # rdi = capacity * 8 (bytes for elements)\n"
            code += "  add rdi, 16 # rdi = total size for malloc (header + elements)\n"
            code += "  call malloc # rax = new_list_ptr\n"
            code += "  pop rdx # Restore value (RDX)\n"
            code += "  pop rsi # Restore index (RSI)\n"
            code += "  mov rdi, rax # list_ptr = new_list_ptr\n"
            code += "  mov QWORD PTR [rdi], r12 # list_ptr[0] = capacity\n"
            code += "  mov QWORD PTR [rdi + 8], r13 # list_ptr[1] = length (still 0)\n"
            
            # Check index bounds for new list (should be 0)
            code += "  cmp rsi, 0 # For a new list, the only valid index is 0\n"
            code += f"  je .L_insert_in_bounds_{insert_id}\n"
            code += f"  jmp .L_insert_out_of_bounds_{insert_id}\n"
            
            code += f".L_insert_end_{insert_id}:\n"
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
            
            # Add unique ID for labels
            input_id = self.next_input_id
            self.next_input_id += 1
            
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
            label_end = f".L_input_end_{input_id}"
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
            
            # Use unique IDs for labels
            to_int_id = self.next_to_int_id
            self.next_to_int_id += 1
            
            code = f"  # to_int() call at L{node.name_token.lineno}\n"
            
            # Get the string in RAX
            code += self.visit(node.arguments[0], context)
            code += "  test rax, rax # Check if string pointer is null\n"
            code += f"  jnz .Lto_int_valid_ptr_{to_int_id}\n"
            code += "  mov rax, 0 # Return 0 for null string\n"
            code += f"  jmp .Lto_int_done_{to_int_id}\n"
            code += f".Lto_int_valid_ptr_{to_int_id}:\n"
            code += "  mov rdi, rax # String to convert\n"
            code += "  call atoi # Convert string to integer\n"
            code += f".Lto_int_done_{to_int_id}:\n"
            # Result is in RAX
            return code

        elif node.name == "to_string":
            # to_string() function - converts an integer to string
            if len(node.arguments) != 1:
                raise CompilerError(f"to_string() expects exactly one argument (int), got {len(node.arguments)} L{node.name_token.lineno}")
            
            # Use unique IDs for labels
            to_string_id = self.next_to_string_id
            self.next_to_string_id += 1
            
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

        elif node.name == "to_stringf":
            # to_stringf() function - converts a float to string
            if len(node.arguments) != 1:
                raise CompilerError(f"to_stringf() expects exactly one argument (float), got {len(node.arguments)} L{node.name_token.lineno}")
            
            # Use unique IDs for labels
            to_stringf_id = self.next_to_string_id  # Reuse the to_string ID counter
            self.next_to_string_id += 1
            
            code = f"  # to_stringf() call at L{node.name_token.lineno}\n"
            
            # Get the float in XMM0
            code += self.visit(node.arguments[0], context)
            
            # Use snprintf to safely convert float to string
            if "snprintf" not in self.externs:
                self.externs.append("snprintf")
                
            # Allocate buffer for the result
            code += "  sub rsp, 16 # Reserve 16 bytes aligned space for float value\n"
            code += "  movsd [rsp], xmm0 # Save the float value on stack\n"
            code += "  mov rdi, 32 # Buffer size for float\n"
            code += "  call malloc # Allocate buffer\n"
            code += "  mov r13, rax # Save buffer pointer\n"
            
            # Format string for snprintf with %g format for float
            format_str = "%g"
            format_label = self.new_string_label(format_str)
            
            # Set up snprintf arguments
            code += "  mov rdi, rax # Buffer pointer (1st arg)\n"
            code += "  mov rsi, 32 # Buffer size (2nd arg)\n"
            code += f"  lea rdx, {format_label}[rip] # Format string (3rd arg)\n"
            code += "  movsd xmm0, [rsp] # Restore float value to XMM0\n"
            
            # For variadic functions with floating-point args, need to set RAX to number of vector registers used
            code += "  mov rax, 1 # 1 vector argument for variadic call\n"
            code += "  call snprintf # Convert float to string\n"
            code += "  add rsp, 16 # Clean up stack\n"
            
            # Return the buffer pointer that was saved earlier
            code += "  mov rax, r13 # Return buffer pointer\n"
            
            return code

        elif node.name == "to_intf":
            # to_intf() function - converts a float to integer
            if len(node.arguments) != 1:
                raise CompilerError(f"to_intf() expects exactly one argument (float), got {len(node.arguments)} L{node.name_token.lineno}")
            
            # Use unique IDs for labels
            to_intf_id = self.next_to_int_id  # Reuse the to_int ID counter
            self.next_to_int_id += 1
            
            code = f"  # to_intf() call at L{node.name_token.lineno}\n"
            
            # Get the float in XMM0
            code += self.visit(node.arguments[0], context)
            
            # Convert the float in XMM0 to integer in RAX using cvttsd2si
            # We use cvttsd2si which truncates toward zero
            code += "  cvttsd2si rax, xmm0 # Convert float to integer (truncate)\n"
            
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
        
        # Handle string concatenation with floats
        if node.op_token.type == TT_PLUS and (left_type == "string" or right_type == "string"):
            is_string_concat = True
            op_assembly += f"  # String concatenation detected\n"
            
            # If either side is a float, we need to convert it to string first
            if left_type == "float":
                # Convert left float to string
                op_assembly += self.visit(node.left, context)  # Result in XMM0
                
                # Call to_stringf internally
                to_stringf_id = self.next_to_string_id
                self.next_to_string_id += 1
                
                # Use snprintf to safely convert float to string
                if "snprintf" not in self.externs:
                    self.externs.append("snprintf")
                    
                # Allocate buffer for the result
                op_assembly += "  sub rsp, 16 # Reserve 16 bytes aligned space for float value\n"
                op_assembly += "  movsd [rsp], xmm0 # Save the float value on stack\n"
                op_assembly += "  mov rdi, 32 # Buffer size for float\n"
                op_assembly += "  call malloc # Allocate buffer\n"
                op_assembly += "  mov r13, rax # Save buffer pointer\n"
                
                # Format string for snprintf with %g format for float
                format_str = "%g"
                format_label = self.new_string_label(format_str)
                
                # Set up snprintf arguments
                op_assembly += "  mov rdi, rax # Buffer pointer (1st arg)\n"
                op_assembly += "  mov rsi, 32 # Buffer size (2nd arg)\n"
                op_assembly += f"  lea rdx, {format_label}[rip] # Format string (3rd arg)\n"
                op_assembly += "  movsd xmm0, [rsp] # Restore float value to XMM0\n"
                op_assembly += "  mov rax, 1 # 1 vector argument for variadic call\n"
                op_assembly += "  call snprintf # Convert float to string\n"
                op_assembly += "  add rsp, 16 # Clean up stack\n"
                op_assembly += "  mov rax, r13 # String in RAX\n"
                op_assembly += "  push rax # Save left string\n"
                
                # Process right operand
                op_assembly += self.visit(node.right, context)  # Result in RAX or XMM0
                op_assembly += "  push rax # Save right string/value\n"
            elif right_type == "float":
                # Handle left operand (string)
                op_assembly += self.visit(node.left, context)  # Result in RAX
                op_assembly += "  push rax # Save left string\n"
                
                # Convert right float to string
                op_assembly += self.visit(node.right, context)  # Result in XMM0
                
                # Call to_stringf internally
                to_stringf_id = self.next_to_string_id
                self.next_to_string_id += 1
                
                # Use snprintf to safely convert float to string
                if "snprintf" not in self.externs:
                    self.externs.append("snprintf")
                    
                # Allocate buffer for the result
                op_assembly += "  sub rsp, 16 # Reserve 16 bytes aligned space for float value\n"
                op_assembly += "  movsd [rsp], xmm0 # Save the float value on stack\n"
                op_assembly += "  mov rdi, 32 # Buffer size for float\n"
                op_assembly += "  call malloc # Allocate buffer\n"
                op_assembly += "  mov r13, rax # Save buffer pointer\n"
                
                # Format string for snprintf with %g format for float
                format_str = "%g"
                format_label = self.new_string_label(format_str)
                
                # Set up snprintf arguments
                op_assembly += "  mov rdi, rax # Buffer pointer (1st arg)\n"
                op_assembly += "  mov rsi, 32 # Buffer size (2nd arg)\n"
                op_assembly += f"  lea rdx, {format_label}[rip] # Format string (3rd arg)\n"
                op_assembly += "  movsd xmm0, [rsp] # Restore float value to XMM0\n"
                op_assembly += "  mov rax, 1 # 1 vector argument for variadic call\n"
                op_assembly += "  call snprintf # Convert float to string\n"
                op_assembly += "  add rsp, 16 # Clean up stack\n"
                op_assembly += "  mov rax, r13 # String in RAX\n"
                op_assembly += "  push rax # Save right string\n"
            else:
                # Regular string concatenation without floats
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
            
            # Calculate total length
            op_assembly += "  add rax, rbx # Total length\n"
            op_assembly += "  add rax, 1 # Add 1 for null terminator\n"
            
            # Allocate memory for new string
            op_assembly += "  mov r12, rax # Save total length in r12\n"
            op_assembly += "  mov rdi, rax # Argument for malloc\n"
            if "malloc" not in self.externs:
                self.externs.append("malloc")
            op_assembly += "  call malloc # Allocate memory for new string\n"
            
            # Store the result in a safe register
            op_assembly += "  mov r13, rax # Save result string pointer\n"
            
            # Copy left string
            op_assembly += "  mov rdi, rax # Destination\n"
            op_assembly += "  mov rsi, [rsp+8] # Source (left string)\n"
            if "strcpy" not in self.externs:
                self.externs.append("strcpy")
            op_assembly += "  call strcpy # Copy left string\n"
            
            # Calculate end of left string
            op_assembly += "  mov rdi, r13 # Result string\n"
            op_assembly += "  call strlen # Get length after copying left\n"
            op_assembly += "  add rax, r13 # Pointer to end of left string\n"
            
            # Append right string
            op_assembly += "  mov rdi, rax # Destination (end of left string)\n"
            op_assembly += "  mov rsi, [rsp] # Source (right string)\n"
            if "strcpy" not in self.externs:
                self.externs.append("strcpy")
            op_assembly += "  call strcpy # Append right string\n"
            
            # Clean up the stack
            op_assembly += "  add rsp, 16 # Pop saved strings\n"
            
            # Final result is in r13
            op_assembly += "  mov rax, r13 # Final concatenated string\n"
            
            return op_assembly
        
        # Handle floating-point operations
        elif left_type == "float" or right_type == "float":
            float_code = f"  # Floating-point {node.op} at L{node.op_token.lineno}\n"
            # Evaluate left operand -> xmm0
            float_code += self.visit(node.left, context)
            float_code += "  sub rsp, 8             # Reserve space for left float\n"
            float_code += "  movsd [rsp], xmm0      # Store left float\n"
            # Evaluate right operand -> xmm0
            float_code += self.visit(node.right, context)
            # Swap: xmm0=right, load left into xmm1
            float_code += "  movsd xmm1, [rsp]      # Load left float into xmm1\n"
            float_code += "  add rsp, 8             # Restore stack\n"
            
            op = node.op_token.type
            # Handle arithmetic operations
            if op == TT_PLUS:
                float_code += "  addsd xmm0, xmm1      # xmm0 = left + right\n"
            elif op == TT_MINUS:
                float_code += "  subsd xmm1, xmm0      # xmm1 = left - right\n"
                float_code += "  movsd xmm0, xmm1      # Move result to xmm0\n"
            elif op == TT_STAR:
                float_code += "  mulsd xmm0, xmm1      # xmm0 = left * right\n"
            elif op == TT_SLASH:
                float_code += "  divsd xmm1, xmm0      # xmm1 = left / right\n"
                float_code += "  movsd xmm0, xmm1      # Move result to xmm0\n"
            # Handle comparison operations
            elif op in [TT_LESS_THAN, TT_GREATER_THAN, TT_EQUAL, TT_NOT_EQUAL, TT_LESS_EQUAL, TT_GREATER_EQUAL]:
                # For floating point comparisons, we use ucomisd instruction which sets CPU flags
                # ucomisd compares xmm0 with xmm1, so we need to swap the operands to get the right comparison
                float_code += "  ucomisd xmm0, xmm1    # Compare right (xmm0) with left (xmm1)\n"
                
                # For expressions, we need to set RAX to 0 or 1 based on the comparison result
                # Note: we're comparing xmm0 (right) with xmm1 (left), so the condition is inverted
                if op == TT_LESS_THAN:         # left < right becomes right > left
                    float_code += "  seta al             # Set AL to 1 if right > left (carry flag set)\n"
                elif op == TT_GREATER_THAN:    # left > right becomes right < left
                    float_code += "  setb al             # Set AL to 1 if right < left (below flag set)\n"
                elif op == TT_LESS_EQUAL:      # left <= right becomes right >= left
                    float_code += "  setae al            # Set AL to 1 if right >= left (carry flag not set)\n"
                elif op == TT_GREATER_EQUAL:   # left >= right becomes right <= left
                    float_code += "  setbe al            # Set AL to 1 if right <= left (below or equal flag set)\n"
                elif op == TT_EQUAL:
                    float_code += "  sete al             # Set AL to 1 if equal (zero flag set)\n"
                elif op == TT_NOT_EQUAL:
                    float_code += "  setne al            # Set AL to 1 if not equal (zero flag not set)\n"
                    
                float_code += "  movzx rax, al        # Zero-extend AL to RAX\n"
                # Return early since we've already set RAX with the comparison result
                return float_code
            else:
                raise CompilerError(f"Unsupported float operator '{node.op}' at L{node.op_token.lineno}")
            return float_code
        
        # Regular integer operations and string concatenation (unchanged)
        # ... existing code ...
        
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
                op_assembly += f"  # String concatenation detected\n"
        elif node.op_token.type in [TT_LESS_THAN, TT_GREATER_THAN, TT_EQUAL, 
                                    TT_NOT_EQUAL, TT_LESS_EQUAL, TT_GREATER_EQUAL]:
            if left_type == "string" and right_type == "string":
                is_string_comparison = True
        
        if is_string_comparison:
            # String comparison using strcmp
            op_assembly += self.visit(node.left, context)  # Result in RAX
            op_assembly += "  push rax # Save left string\n"
            
            op_assembly += self.visit(node.right, context) # Result in RAX
            op_assembly += "  mov rsi, rax # Right string to RSI\n"
            op_assembly += "  mov rdi, [rsp] # Left string to RDI\n"
            
            # Add strcmp to externs if not already there
            if "strcmp" not in self.externs:
                self.externs.append("strcmp")
            
            op_assembly += "  call strcmp # Compare strings\n"
            op_assembly += "  add rsp, 8 # Pop left string\n"
            
            # Result of strcmp is in RAX: <0 if left<right, 0 if equal, >0 if left>right
            if node.op_token.type == TT_EQUAL:
                op_assembly += "  cmp rax, 0 # Check if strings are equal\n"
                op_assembly += "  sete al # Set AL to 1 if equal\n"
            elif node.op_token.type == TT_NOT_EQUAL:
                op_assembly += "  cmp rax, 0 # Check if strings are equal\n"
                op_assembly += "  setne al # Set AL to 1 if not equal\n"
            elif node.op_token.type == TT_LESS_THAN:
                op_assembly += "  cmp rax, 0 # Check if left < right\n"
                op_assembly += "  setl al # Set AL to 1 if less than\n"
            elif node.op_token.type == TT_GREATER_THAN:
                op_assembly += "  cmp rax, 0 # Check if left > right\n"
                op_assembly += "  setg al # Set AL to 1 if greater than\n"
            elif node.op_token.type == TT_LESS_EQUAL:
                op_assembly += "  cmp rax, 0 # Check if left <= right\n"
                op_assembly += "  setle al # Set AL to 1 if less than or equal\n"
            elif node.op_token.type == TT_GREATER_EQUAL:
                op_assembly += "  cmp rax, 0 # Check if left >= right\n"
                op_assembly += "  setge al # Set AL to 1 if greater than or equal\n"
            
            op_assembly += "  movzx rax, al # Zero-extend AL to RAX\n"
            return op_assembly
        
        # For regular integer operations (including string concat which has been handled above)
        op_assembly += self.visit(node.left, context)  # Result in RAX
        op_assembly += "  push rax # Save left operand\n"
        
        op_assembly += self.visit(node.right, context) # Result in RAX
        op_assembly += "  mov rbx, rax # Move right operand to RBX\n"
        op_assembly += "  pop rax # Restore left operand to RAX\n"
        
        op = node.op_token.type
        # Handle the operation
        if op == TT_PLUS:
            op_assembly += "  add rax, rbx # Add right to left\n"
        elif op == TT_MINUS:
            op_assembly += "  sub rax, rbx # Subtract right from left\n"
        elif op == TT_STAR:
            op_assembly += "  imul rax, rbx # Multiply left by right\n"
        elif op == TT_SLASH:
            # Handle division by zero
            op_assembly += "  test rbx, rbx # Check if divisor is zero\n"
            op_assembly += "  jnz .Ldiv_ok # Skip if not zero\n"
            op_assembly += "  mov rdi, 1 # Exit code 1\n"
            op_assembly += "  call exit # Exit on division by zero\n"
            op_assembly += ".Ldiv_ok:\n"
            op_assembly += "  cqo # Sign-extend RAX into RDX:RAX\n"
            op_assembly += "  idiv rbx # Divide RDX:RAX by RBX, quotient in RAX\n"
        elif op == TT_PERCENT:
            # Handle modulo by zero
            op_assembly += "  test rbx, rbx # Check if divisor is zero\n"
            op_assembly += "  jnz .Lmod_ok # Skip if not zero\n"
            op_assembly += "  mov rdi, 1 # Exit code 1\n"
            op_assembly += "  call exit # Exit on modulo by zero\n"
            op_assembly += ".Lmod_ok:\n"
            op_assembly += "  cqo # Sign-extend RAX into RDX:RAX\n"
            op_assembly += "  idiv rbx # Divide RDX:RAX by RBX, remainder in RDX\n"
            op_assembly += "  mov rax, rdx # Move remainder to RAX\n"
        elif op == TT_LOGICAL_AND:
            op_assembly += "  and rax, rbx # Bitwise AND\n"
        elif op == TT_LOGICAL_OR:
            op_assembly += "  or rax, rbx # Bitwise OR\n"
        elif op == TT_AMPERSAND:
            op_assembly += "  and rax, rbx # Bitwise AND (&)\n"
        elif op == TT_PIPE:
            op_assembly += "  or rax, rbx # Bitwise OR (|)\n"
        elif op == TT_CARET:
            op_assembly += "  xor rax, rbx # Bitwise XOR\n"
        elif op == TT_LSHIFT:
            op_assembly += "  mov rcx, rbx # Move shift count to RCX\n"
            op_assembly += "  shl rax, cl # Left shift by CL\n"
        elif op == TT_RSHIFT:
            op_assembly += "  mov rcx, rbx # Move shift count to RCX\n"
            op_assembly += "  shr rax, cl # Right shift by CL\n"
        elif op == TT_URSHIFT:
            op_assembly += "  mov rcx, rbx # Move shift count to RCX\n"
            op_assembly += "  shr rax, cl # Unsigned right shift by CL\n"
        # Handle comparison operations
        elif op == TT_EQUAL:
            op_assembly += "  cmp rax, rbx # Compare left with right\n"
            op_assembly += "  sete al # Set AL to 1 if equal\n"
            op_assembly += "  movzx rax, al # Zero-extend AL to RAX\n"
        elif op == TT_NOT_EQUAL:
            op_assembly += "  cmp rax, rbx # Compare left with right\n"
            op_assembly += "  setne al # Set AL to 1 if not equal\n"
            op_assembly += "  movzx rax, al # Zero-extend AL to RAX\n"
        elif op == TT_LESS_THAN:
            op_assembly += "  cmp rax, rbx # Compare left with right\n"
            op_assembly += "  setl al # Set AL to 1 if less than\n"
            op_assembly += "  movzx rax, al # Zero-extend AL to RAX\n"
        elif op == TT_GREATER_THAN:
            op_assembly += "  cmp rax, rbx # Compare left with right\n"
            op_assembly += "  setg al # Set AL to 1 if greater than\n"
            op_assembly += "  movzx rax, al # Zero-extend AL to RAX\n"
        elif op == TT_LESS_EQUAL:
            op_assembly += "  cmp rax, rbx # Compare left with right\n"
            op_assembly += "  setle al # Set AL to 1 if less than or equal\n"
            op_assembly += "  movzx rax, al # Zero-extend AL to RAX\n"
        elif op == TT_GREATER_EQUAL:
            op_assembly += "  cmp rax, rbx # Compare left with right\n"
            op_assembly += "  setge al # Set AL to 1 if greater than or equal\n"
            op_assembly += "  movzx rax, al # Zero-extend AL to RAX\n"
        else:
            raise CompilerError(f"Unsupported operator '{node.op}' at L{node.op_token.lineno}")
            
        return op_assembly

    def get_expr_type(self, node, context):
        """Helper method to determine expression type for string concatenation checks"""
        if isinstance(node, StringLiteralNode):
            return "string"
        elif isinstance(node, IntegerLiteralNode):
            return "int"
        elif isinstance(node, FloatLiteralNode):
            return "float"
        elif isinstance(node, IdentifierNode):
            var_name = node.value
            if var_name in self.current_method_locals:
                return self.current_method_locals[var_name]['type']
            elif var_name in self.current_method_params:
                return self.current_method_params[var_name]['type']
        elif isinstance(node, ClassVarAccessNode):
            # Get type of class variable
            class_name, method_name = context
            var_name = node.var_name
            
            # Check if this is a 'this:varname' access
            is_this_access = isinstance(node.object_expr, ThisNode)
            
            if is_this_access:
                # Accessing variable from current class
                if var_name in self.class_vars.get(class_name, {}):
                    _, var_type, _ = self.class_vars[class_name][var_name]
                    return var_type
            else:
                # Accessing variable from another object (e.g., obj::varname)
                if isinstance(node.object_expr, IdentifierNode):
                    obj_name = node.object_expr.value
                    obj_class_type = None
                    
                    if obj_name in self.current_method_locals:
                        obj_class_type = self.current_method_locals[obj_name]['type']
                    elif obj_name in self.current_method_params:
                        obj_class_type = self.current_method_params[obj_name]['type']
                    
                    if obj_class_type:
                        obj_class_type = obj_class_type.replace('.', '_')
                        if var_name in self.class_vars.get(obj_class_type, {}):
                            _, var_type, _ = self.class_vars[obj_class_type][var_name]
                            return var_type
            
            return "unknown"  # Default if type cannot be determined
        elif isinstance(node, FunctionCallNode):
            func_name = node.name
            if func_name == "input" or func_name == "to_string":
                return "string"
            elif func_name == "to_int":
                return "int"
            elif func_name == "read":
                return "string"
            elif func_name == "index": # index() now returns a string
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
        
        # Step 1: Evaluate the condition.
        # The visit method for conditions (e.g., BinaryOpNode for comparisons, logical ops)
        # is expected to leave 1 in RAX for true, and 0 in RAX for false.
        
        if_assembly += self.visit(node.condition, context)
        
        # Check if the result of the condition is false (0).
        if_assembly += "  cmp rax, 0                   # Check if condition result is false (0)\n"
        
        jump_target = else_label if node.else_block else end_if_label
        if_assembly += f"  jz {jump_target}             # Jump if false\n"
        
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

        # Handle unary minus (-)
        if node.op_token.type == TT_MINUS:
            assembly_code += self.visit(node.operand_node, context) # Evaluate the operand (result in RAX)
            assembly_code += "  neg rax      # Negate the value in RAX\n"
            return assembly_code

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
            # This case should ideally not be reached with the checks above
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

    def visit_WhileNode(self, node: 'WhileNode', context):
        while_start_label, while_end_label = self.new_while_labels()
        
        # Also push end label for 'stop' statements
        self.loop_end_labels.append(while_end_label)

        code = f"{while_start_label}: # While loop at L{node.lineno}\n"
        
        # Evaluate condition
        code += self.visit(node.condition, context)
        code += "  cmp rax, 0\n"
        code += f"  je {while_end_label}\n" # Jump if condition is false

        # Body of the loop
        code += self.visit(node.body, context)

        # Jump back to the start to re-evaluate condition
        code += f"  jmp {while_start_label}\n"

        code += f"{while_end_label}:\n"

        # Pop the label for 'stop' statements
        self.loop_end_labels.pop()

        return code

    def visit_StopNode(self, node: StopNode, context):
        # Jump to nearest loop end label
        if not self.loop_end_labels:
            raise CompilerError(f"Stop used outside of loop at L{node.lineno}")
        end_label = self.loop_end_labels[-1]
        return f"  jmp {end_label} # stop\n"

    def visit_AssignmentNode(self, node: AssignmentNode, context):
        assign_assembly = f"  # Assignment at L{node.lineno}\n"

        # Handle simple variable assignment: var = expr
        if isinstance(node.target, IdentifierNode):
            var_name = node.target.value
            # Check if it's a parameter or local variable
            local_info = None
            var_offset_rbp = None
            
            if var_name in self.current_method_locals:
                local_info = self.current_method_locals[var_name]
                var_offset_rbp = local_info['offset_rbp']
            elif var_name in self.current_method_params:
                local_info = self.current_method_params[var_name]
                var_offset_rbp = local_info['offset_rbp']
            
            if var_offset_rbp is None:
                raise CompilerError(f"Undefined variable '{var_name}' in assignment at L{node.lineno}")
            
            var_type_str = local_info['type'] # e.g., "int", "int[]", "float"
            
            # Evaluate the expression
            assign_assembly += self.visit(node.expression, context) # Result will be in RAX or XMM0 for floats
            
            # Handle different variable types
            if var_type_str == "float":
                # For float variables, store from XMM0
                assign_assembly += f"  movsd QWORD PTR [rbp {var_offset_rbp}], xmm0 # Assign float {var_name} at [rbp{var_offset_rbp}]\n"
            else:
                # For non-float variables (including pointers like lists), store from RAX
                assign_assembly += f"  mov QWORD PTR [rbp {var_offset_rbp}], rax # Assign {var_name} at [rbp{var_offset_rbp}]\n"
            
            # Assignments have no value in our language model (returns 0)
            assign_assembly += "  xor rax, rax # Zero return value\n"
            return assign_assembly
        
        # Handle assignment to array element: array[index] = expr
        elif isinstance(node.target, ArrayAccessNode):
            # Generate a unique ID for this array assignment
            array_assign_id = self.next_array_access_id
            self.next_array_access_id += 1
            
            # Get the element type for proper assignment
            element_type = self.get_array_element_type(node.target.array_expr, context)
            
            # First, evaluate the array expression to get the array pointer
            assign_assembly += self.visit(node.target.array_expr, context)
            assign_assembly += "  push rax # Save array pointer\n"
            
            # Next, evaluate the index expression
            assign_assembly += self.visit(node.target.index_expr, context)
            assign_assembly += "  push rax # Save index\n"
            
            # Now evaluate the right-hand side expression
            assign_assembly += self.visit(node.expression, context)
            
            # For float assignments, the value is in XMM0, need to handle differently
            if element_type == 'float':
                # For now, push the float value to the stack (we'll need to store it later)
                assign_assembly += "  sub rsp, 8 # Reserve space for float value\n"
                assign_assembly += "  movsd QWORD PTR [rsp], xmm0 # Save float value on stack\n"
            else:
                # For non-float values, just push rax
                assign_assembly += "  push rax # Save right-hand side value\n"
            
            # Get the index back
            assign_assembly += "  pop rbx # Right-hand side value in RBX\n"
            assign_assembly += "  pop rcx # Index in RCX\n"
            assign_assembly += "  pop rax # Array pointer in RAX\n"
            
            # Check for null pointer
            assign_assembly += "  cmp rax, 0 # Check for null array\n"
            assign_assembly += f"  jne .L_assign_not_null_{array_assign_id}\n"
            # Handle null pointer access
            error_msg_null = self.new_string_label("Error: Null pointer access in array assignment.\n")
            assign_assembly += f"  lea rdi, {error_msg_null}[rip]\n"
            assign_assembly += "  call printf\n"
            assign_assembly += "  mov rdi, 1\n"
            assign_assembly += "  call exit\n"
            assign_assembly += f".L_assign_not_null_{array_assign_id}:\n"
            
            # Bounds check
            idx_ok_label = f".L_assign_idx_ok_{array_assign_id}"
            idx_err_label = f".L_assign_idx_err_{array_assign_id}"
            
            assign_assembly += "  mov rdx, QWORD PTR [rax + 8] # length in RDX\n"
            assign_assembly += "  cmp rcx, 0                   # if index < 0\n"
            assign_assembly += f"  jl {idx_err_label}           # jump to error\n"
            assign_assembly += "  cmp rcx, rdx                 # if index >= length\n"
            assign_assembly += f"  jge {idx_err_label}          # jump to error\n"
            assign_assembly += f"  jmp {idx_ok_label}           # index is OK\n"
            
            assign_assembly += f"{idx_err_label}:\n"
            error_msg_bounds = self.new_string_label("Error: Array index out of bounds during assignment.\n")
            assign_assembly += f"  lea rdi, {error_msg_bounds}[rip]\n"
            assign_assembly += "  call printf\n"
            assign_assembly += "  mov rdi, 1\n"
            assign_assembly += "  call exit\n"
            
            assign_assembly += f"{idx_ok_label}:\n"
            # Calculate address: list_ptr + 16 (header) + (index * 8)
            assign_assembly += "  imul rcx, 8                  # index_offset = index * 8\n"
            assign_assembly += "  add rcx, 16                  # total_offset = header_size + index_offset\n"
            assign_assembly += "  add rax, rcx # Calculate address: array_ptr + header + (index * 8)\n"
            
            # Store the value
            if element_type == 'float':
                # For floats, we handle differently
                # Get the current pointer to the float value
                assign_assembly += "  mov rcx, QWORD PTR [rax]  # Get current float pointer\n"
                
                # Check if it's null
                assign_assembly += "  cmp rcx, 0 # Check if float pointer is null\n"
                assign_assembly += f"  jne .L_float_ptr_exists_{array_assign_id}\n"
                
                # Allocate new memory for the float value
                assign_assembly += "  push rax # Save element address\n"
                assign_assembly += "  mov rdi, 8 # Size for float allocation\n"
                assign_assembly += "  call malloc # Allocate memory for float value\n"
                assign_assembly += "  mov rcx, rax # Store float ptr in RCX\n"
                assign_assembly += "  pop rax # Restore element address\n"
                
                assign_assembly += f".L_float_ptr_exists_{array_assign_id}:\n"
                # Store the float value at the pointer
                assign_assembly += "  movsd QWORD PTR [rcx], xmm0 # Store float value at allocated memory\n"
                assign_assembly += "  mov QWORD PTR [rax], rcx # Update the float pointer in the list\n"
            else:
                # For regular values, store from RBX
                assign_assembly += "  mov QWORD PTR [rax], rbx # Store the value\n"
            
            # Assignment has no value (returns 0)
            assign_assembly += "  xor rax, rax # Zero return value\n"
            return assign_assembly
        
        # Handle assignment to class variable: this:var = expr OR obj::var = expr
        elif isinstance(node.target, ClassVarAccessNode):
            class_var_target = node.target
            class_name, method_name = context # Current method's class context

            assign_assembly = f"  # Assignment to class variable at L{node.lineno}\n"

            # 1. Determine the target class and get variable info (offset, type, public/private)
            target_class_for_lookup = class_name # Default to current class
            if not isinstance(class_var_target.object_expr, ThisNode):
                # If not 'this', try to infer the class type of the object expression
                obj_expr_type = self.get_expr_type(class_var_target.object_expr, context)
                if obj_expr_type and obj_expr_type != "unknown" and not obj_expr_type.endswith("[]"):
                    target_class_for_lookup = obj_expr_type.replace('.', '_')
                else:
                    raise CompilerError(f"Cannot determine class type for object in assignment '{class_var_target.object_expr.value}::{class_var_target.var_name}' at L{node.lineno}")
            
            target_var_name = class_var_target.var_name
            safe_target_class_name = target_class_for_lookup.replace('.', '_')

            if safe_target_class_name not in self.class_vars or \
               target_var_name not in self.class_vars[safe_target_class_name]:
                raise CompilerError(f"Undefined class variable '{target_var_name}' in class '{safe_target_class_name}' for assignment at L{node.lineno}")
            
            offset, var_type, is_public = self.class_vars[safe_target_class_name][target_var_name]

            # Access check for obj::var (needs to be done after object type is known)
            if not isinstance(class_var_target.object_expr, ThisNode) and not is_public and class_name != safe_target_class_name:
                 raise CompilerError(f"Cannot assign to private variable '{target_var_name}' of class '{safe_target_class_name}' from class '{class_name}' at L{node.lineno}")


            # 2. Evaluate the RHS expression (value to assign)
            # Result will be in RAX (for int/ptr) or XMM0 (for float)
            assign_assembly += self.visit(node.expression, context)
            
            # 3. Save the RHS value onto the stack
            if var_type == "float":
                assign_assembly += "  sub rsp, 8             # Reserve space for float value\n"
                assign_assembly += "  movsd [rsp], xmm0      # Save RHS float value on stack\n"
            else:
                assign_assembly += "  push rax               # Save RHS non-float value on stack\n"

            # 4. Get the object pointer into R11
            if isinstance(class_var_target.object_expr, ThisNode):
                if self.this_ptr_rbp_offset is None:
                    raise CompilerError(f"'this' pointer not available for assignment to 'this:{target_var_name}'. L{node.lineno}")
                assign_assembly += f"  mov r11, QWORD PTR [rbp {self.this_ptr_rbp_offset}]  # Load 'this' pointer into R11\n"
            else:
                # Evaluate the object expression (e.g., 'obj' in obj::var)
                assign_assembly += self.visit(class_var_target.object_expr, context) # Object pointer in RAX
                assign_assembly += "  mov r11, rax             # Move object pointer to R11\n"
            
            # 5. Restore the RHS value
            if var_type == "float":
                assign_assembly += "  movsd xmm0, [rsp]      # Restore RHS float value to XMM0\n"
                assign_assembly += "  add rsp, 8             # Clean up stack\n"
            else:
                assign_assembly += "  pop r10                # Restore RHS non-float value to R10\n"

            # 6. Perform Assignment
            # R11 contains the object pointer (this or obj)
            # Value to store is in XMM0 (if float) or R10 (if non-float)
            if var_type == "float":
                assign_assembly += f"  movsd QWORD PTR [r11 + {offset}], xmm0  # Assign to float class var {target_var_name}\n"
            else:
                assign_assembly += f"  mov QWORD PTR [r11 + {offset}], r10   # Assign to class var {target_var_name}\n"

            # Assignments have no value in our language model (returns 0)
            assign_assembly += "  xor rax, rax # Zero return value for assignment expression\n"
            return assign_assembly

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

        # Determine the element type of this list (important for float lists)
        element_type = "unknown"
        is_float_matrix = False
        if num_elements > 0:
            element_type = self.get_expr_type(node.elements[0], context)
            # Check if this is a matrix with float elements
            if element_type.endswith("[]"):
                # This is a list of lists - check first inner list to see if it contains floats
                if isinstance(node.elements[0], ListLiteralNode) and len(node.elements[0].elements) > 0:
                    inner_element_type = self.get_expr_type(node.elements[0].elements[0], context)
                    is_float_matrix = (inner_element_type == "float")
        
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
                
                # Need to get list_ptr from stack to store element
                code += "  pop rbx               # list_ptr into RBX\n"
                current_element_offset = element_base_offset + (i * 8)
                code += f"  mov QWORD PTR [rbx + {current_element_offset}], rax # list_ptr[element_offset] = value\n"
                code += "  push rbx              # Push list_ptr back on stack\n"
            elif element_type == "float" or isinstance(expr_node, FloatLiteralNode):
                # For float elements, we need to handle the value differently
                # 1. Evaluate the expression (puts float value in XMM0)
                code += self.visit(expr_node, context)
                
                # 2. Allocate memory for the float value (8 bytes)
                code += "  push rax              # Save any registers\n"
                code += "  mov rdi, 8            # Size for float allocation\n"
                code += "  call malloc           # Allocate memory for float value\n"
                code += "  movsd QWORD PTR [rax], xmm0 # Store float value at allocated memory\n"
                
                # 3. Store the pointer to the float value in the list
                code += "  mov rcx, rax          # Save float ptr in RCX\n"
                code += "  pop rax               # Restore RAX\n"
                code += "  pop rbx               # list_ptr into RBX\n"
                current_element_offset = element_base_offset + (i * 8)
                code += f"  mov QWORD PTR [rbx + {current_element_offset}], rcx # list_ptr[element_offset] = float_ptr\n"
                code += "  push rbx              # Push list_ptr back on stack\n"
            else:
                # Normal element (integer, string, etc.)
                code += self.visit(expr_node, context) # Element value in RAX
                
                # Need to get list_ptr from stack to store element
                code += "  pop rbx               # list_ptr into RBX\n"
                current_element_offset = element_base_offset + (i * 8)
                code += f"  mov QWORD PTR [rbx + {current_element_offset}], rax # list_ptr[element_offset] = value\n"
                code += "  push rbx              # Push list_ptr back on stack\n"
        
        code += "  pop rax # Final list_ptr (from last push) into RAX to be the result of this expression\n"
        return code

    def visit_ArrayAccessNode(self, node: ArrayAccessNode, context):
        # Use a unique ID for array access operations
        array_access_id = self.next_array_access_id
        self.next_array_access_id += 1
        
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
        code += f"  jne .L_access_not_null_{array_access_id}\n"
        # Handle null pointer access
        error_msg_null = self.new_string_label("Error: Null pointer access in array operation.\n")
        code += f"  lea rdi, {error_msg_null}[rip]\n"
        code += "  call printf\n"
        code += "  mov rdi, 1\n"
        code += "  call exit\n"
        code += f".L_access_not_null_{array_access_id}:\n"
        
        # Bounds check
        idx_ok_label = f".L_access_idx_ok_{array_access_id}"
        idx_err_label = f".L_access_idx_err_{array_access_id}"

        code += "  mov rcx, QWORD PTR [rax + 8] # length in RCX\n"
        code += "  cmp rbx, 0                   # if index < 0\n"
        code += f"  jl {idx_err_label}           # jump to error\n"
        code += "  cmp rbx, rcx                 # if index >= length\n"
        code += f"  jge {idx_err_label}          # jump to error\n"
        code += f"  jmp {idx_ok_label}           # index is OK\n"

        code += f"{idx_err_label}:\n"
        error_msg_bounds = self.new_string_label("Error: Array index out of bounds during access.\n")
        code += f"  lea rdi, {error_msg_bounds}[rip]\n"
        code += "  call printf\n"
        code += "  mov rdi, 1\n"
        code += "  call exit\n"

        code += f"{idx_ok_label}:\n"
        # Calculate address: list_ptr + 16 (header) + (index * 8)
        code += "  imul rbx, 8                  # index_offset = index * 8\n"
        code += "  add rbx, 16                  # total_offset = header_size + index_offset\n"
        code += "  add rax, rbx                 # element_address = list_ptr + total_offset\n"
        
        # For float arrays, we need special handling
        if element_type == "float":
            # For float arrays, the element is a pointer to a float value
            code += "  mov rax, QWORD PTR [rax]     # Get pointer to float value\n"
            code += "  test rax, rax                # Check if pointer is NULL\n"
            code += f"  jnz .L_valid_float_ptr_{array_access_id}\n"
            # Handle null float pointer
            error_msg_null_float = self.new_string_label("Error: Null float pointer access.\n")
            code += f"  lea rdi, {error_msg_null_float}[rip]\n"
            code += "  call printf\n"
            code += "  mov rdi, 1\n"
            code += "  call exit\n"
            code += f".L_valid_float_ptr_{array_access_id}:\n"
            code += "  movsd xmm0, QWORD PTR [rax]  # Load float value directly into XMM0\n"
        elif element_type.endswith("[]"):
            # For nested lists (e.g., matrices), the value is already a pointer
            code += "  mov rax, QWORD PTR [rax]     # Get list pointer from element\n"
        else:
            # For regular values (int, string), just load the value
            code += "  mov rax, QWORD PTR [rax]     # Get element value\n"
            
        return code

    def compile(self, node: ProgramNode):
        # Store the program node for reference
        self.current_program_node = node
        return self.visit(node)

    def get_method_return_type(self, method_call_node, context):
        """
        Determine the exact return type of a method call by looking up the method definition
        in the program AST.
        
        Args:
            method_call_node: The MethodCallNode representing the method call
            context: The current compilation context (class_name, method_name)
            
        Returns:
            The return type string (e.g., 'int', 'float', 'string') or 'unknown' if not found
        """
        if not hasattr(self, 'current_program_node'):
            return 'unknown'
            
        current_class, _ = context
        method_name = method_call_node.method_name
        
        # Determine the class that contains this method
        target_class = None
        
        # If this is a call on 'this', we're in the same class
        if isinstance(method_call_node.object_expr, ThisNode):
            target_class = current_class
        # If this is a call on another object or class, determine its type
        elif isinstance(method_call_node.object_expr, IdentifierNode):
            obj_name = method_call_node.object_expr.value
            obj_type = None
            
            # Check if it's a class name (for static method calls like ClassName.method())
            for cls in self.current_program_node.classes:
                safe_cls_name = cls.name.replace('.', '_')
                if safe_cls_name == obj_name:
                    obj_type = safe_cls_name
                    break
                    
            # If not a class name, check local variables and parameters
            if obj_type is None:
                if obj_name in self.current_method_locals:
                    obj_type = self.current_method_locals[obj_name]['type']
                elif obj_name in self.current_method_params:
                    obj_type = self.current_method_params[obj_name]['type']
            
            # Clean up object type (remove potential trailing [] for array types)
            if obj_type:
                if obj_type.endswith('[]'):
                    obj_type = obj_type[:-2]  # Remove []
                # Fix class names with dots
                obj_type = obj_type.replace('.', '_')
                    
            target_class = obj_type

        # Search for the method in the target class or all classes if target unknown
        if target_class:
            classes_to_search = [cls for cls in self.current_program_node.classes 
                               if cls.name.replace('.', '_') == target_class]
        else:
            classes_to_search = self.current_program_node.classes
            
        # Look for the method and return its return type
        for cls in classes_to_search:
            for method in cls.methods:
                if method.name == method_name:
                    return method.return_type
                    
        return 'unknown'
        
    # ... existing get_expr_type method remains ...

    def visit_CLibraryCallNode(self, node: CLibraryCallNode, context):
        # 1. Add the library to the set of used C libraries
        lib_name = node.library_token.value
        if lib_name not in self.used_c_libs:
            self.used_c_libs.add(lib_name)
        
        # 2. Add the function to externs if not already there
        func_name = node.function_name
        if func_name not in self.externs:
            self.externs.append(func_name)
            # Prepend the extern declaration to the text section.
            # This is a bit of a hack; a better way is to collect all externs first.
            self.text_section_code = f".extern {func_name}\n" + self.text_section_code

        # 3. Handle arguments
        code = f"  # C Library Call to {func_name} from lib '{lib_name}' at L{node.lineno}\n"
        
        # Define the registers for the first few arguments (System V AMD64 ABI)
        int_arg_regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
        float_arg_regs = ["xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7"]
        
        int_arg_count = 0
        float_arg_count = 0
        
        # Evaluate arguments and move them to registers
        # We process arguments in reverse to handle stack pushing correctly if needed,
        # but for register passing, order doesn't matter as much as type.
        # However, let's just process them in order for simplicity.
        for i, arg_node in enumerate(node.arguments):
            # Determine argument type to use correct register
            # This is a simplification. A real compiler would have a more robust type checking system.
            # We'll infer based on literal type or assume int/pointer for identifiers.
            arg_type = self.get_expr_type(arg_node, context)
            
            code += self.visit(arg_node, context) # The result is in rax (int/ptr) or xmm0 (float)

            if arg_type == "float":
                if float_arg_count < len(float_arg_regs):
                    # If the result is in rax (e.g. from a function returning float), it needs to be moved to xmm
                    # But our convention is that float expressions put result in xmm0.
                    # So if arg_node was a float expr, result is in xmm0. We move it to target xmm.
                    target_reg = float_arg_regs[float_arg_count]
                    if target_reg != "xmm0":
                         code += f"  movq {target_reg}, xmm0 # Move arg {i+1} to {target_reg}\n"
                    float_arg_count += 1
                else:
                    # Stack passing for floats - more complex, not implemented for now
                    raise CompilerError(f"Stack passing for float arguments not yet supported. L{node.lineno}")
            else: # int, string, pointers
                if int_arg_count < len(int_arg_regs):
                    target_reg = int_arg_regs[int_arg_count]
                    if target_reg != "rax":
                        code += f"  mov {target_reg}, rax # Move arg {i+1} to {target_reg}\n"
                    int_arg_count += 1
                else:
                    # Stack passing for ints
                    raise CompilerError(f"Stack passing for integer/pointer arguments not yet supported. L{node.lineno}")

        # Number of float arguments passed in XMM registers must be in AL for variadic functions
        code += f"  mov al, {float_arg_count} # Number of float args in xmm registers\n"
        code += f"  call {func_name}\n"
        
        # The return value is in RAX (for int/ptr) or XMM0 (for float).
        # We need to know the return type of the C function. We'll assume it matches
        # the surrounding expression context, but for now, we leave it in RAX or XMM0.
        # The calling code (e.g. visit_BinaryOpNode) should handle it.
        # This is a major simplification.
        
        return code

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