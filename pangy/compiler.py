from .parser_lexer import (
    ASTNode, ProgramNode, ClassNode, MethodNode, BlockNode, PrintNode,
    ParamNode, IntegerLiteralNode, IdentifierNode, ThisNode,
    MethodCallNode, FunctionCallNode, BinaryOpNode, UnaryOpNode, ReturnNode, StringLiteralNode, IfNode, VarDeclNode, LoopNode, StopNode, AssignmentNode,
    MacroDefNode, MacroInvokeNode,
    TT_LESS_THAN, TT_GREATER_THAN, TT_EQUAL, TT_NOT_EQUAL, TT_LESS_EQUAL, TT_GREATER_EQUAL,
    TT_PLUS, TT_MINUS, TT_STAR, TT_SLASH, TT_PERCENT,
    TT_PLUSPLUS, TT_MINUSMINUS
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
        self.externs = ["printf", "exit", "scanf", "atoi", "malloc", "fopen", "fclose", "fwrite", "fread", "fgets"] # Added file operation externs
        
        self.current_method_params = {}
        self.current_method_locals = {} # For local variables
        self.current_method_stack_offset = 0 # Tracks current available stack slot relative to RBP for new locals
        self.current_method_total_stack_needed = 0 # For params + locals in prologue
        self.current_method_context = None
        # For loop support
        self.next_loop_id = 0
        self.loop_end_labels = []
        # For macro support
        self.macros = {} # Maps macro names to MacroDefNode objects

    def new_string_label(self, value):
        if value in self.string_literals:
            return self.string_literals[value]
        label = f".LC{self.next_string_label_id}"
        self.string_literals[value] = label
        self.next_string_label_id += 1
        escaped_value = value.replace('\\', '\\\\') \
                             .replace('"', '\\"') \
                             .replace('\n', '\\n') \
                             .replace('\t', '\\t') \
                             .replace('\0', '\\0')
        self.data_section_code += f'{label}:\n  .string "{escaped_value}"\n'
        return label

    def new_printf_format_label(self, fmt_string_value):
        label = f".LCPF{self.next_printf_format_label_id}"
        self.next_printf_format_label_id += 1
        
        # Escape for .asciz directive in GAS:
        # 1. literal \ (backslash) becomes \\ (two backslashes)
        # 2. literal " (double quote) becomes \" (backslash, double quote)
        # 3. literal newline (char code 10) becomes the sequence \n (backslash, n)
        # 4. literal tab (char code 9) becomes the sequence \t (backslash, t)
        # Add other C-style escapes as needed.
        escaped_for_asciz = fmt_string_value.replace('\\', '\\\\') \
                                           .replace('"', '\\"') \
                                           .replace('\n', '\\n') \
                                           .replace('\t', '\\t')
        
        self.data_section_code += f'{label}:\n  .asciz "{escaped_for_asciz}"\n'
        return label

    def new_if_labels(self):
        idx = self.next_if_label_id
        self.next_if_label_id += 1
        return f".L_else_{idx}", f".L_end_if_{idx}"

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
        
        # Declare externs for GAS
        temp_externs = list(self.externs) # Use a copy in case self.externs is modified
        if "malloc" not in temp_externs: # Add malloc if not already present (e.g. by .new() detection)
            temp_externs.append("malloc")
        
        for ext_symbol in temp_externs:
            self.text_section_code += f".extern {ext_symbol}\n"
        self.text_section_code += "\n" # Blank line after externs

        self.string_literals = {}
        self.next_string_label_id = 0; self.next_printf_format_label_id = 0; self.next_if_label_id = 0
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
        method_text_assembly = ""
        
        # Determine if method is static (based on AST node flag)
        # Main.main is special: it's an entry point but we need to decide its static/instance nature for 'this'
        # For Math.main() from example, it's explicitly static.
        is_method_static = node.is_static
        is_entry_point_main = node.name == "main" and class_name == "Main" and not node.is_static # Default Main.main is instance

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

        local_stack_needed_for_vars = 0
        current_rbp_offset_for_locals = current_rbp_offset_for_params 
        if isinstance(node.body, BlockNode):
            for stmt in node.body.statements:
                if isinstance(stmt, VarDeclNode):
                    var_size = 8 
                    local_stack_needed_for_vars += var_size
                    current_rbp_offset_for_locals -= var_size
                    self.current_method_locals[stmt.name] = {
                        'offset_rbp': current_rbp_offset_for_locals, 
                        'type': stmt.type,
                        'size': var_size,
                        'is_initialized': False
                    }
        
        self.current_method_total_stack_needed = local_stack_needed_for_params + local_stack_needed_for_vars
        actual_stack_to_allocate = self.current_method_total_stack_needed
        if actual_stack_to_allocate % 16 != 0:
             actual_stack_to_allocate = ((actual_stack_to_allocate // 16) + 1) * 16
        
        if actual_stack_to_allocate > 0:
            method_text_assembly += f"  sub rsp, {actual_stack_to_allocate} # Allocate stack ({actual_stack_to_allocate})\n"
        
        # Spill parameters from registers to their stack slots
        # (No 'this' to spill for static methods from RDI unless it was a param)
        for p_name, p_data in self.current_method_params.items():
            if 'reg' in p_data:
                method_text_assembly += f"  mov QWORD PTR [rbp {p_data['offset_rbp']}], {p_data['reg']} # Spill param {p_name}\n"

        method_text_assembly += self.visit(node.body)
        method_text_assembly += f"{epilogue_label}:\n"
        
        if (is_entry_point_main or (node.is_static and node.name == "main")) and node.return_type == "void": 
            method_text_assembly += "  mov rax, 0 # Default return 0 for main\n"
        
        method_text_assembly += "  mov rsp, rbp \n  pop rbp\n  ret\n\n"
        return method_text_assembly

    def visit_BlockNode(self, node: BlockNode, context):
        return "".join([self.visit(stmt, context) for stmt in node.statements])

    def visit_PrintNode(self, node: PrintNode, context):
        print_assembly = f"  # Print statement at L{node.lineno}\n"
        format_string_parts = []
        arg_expr_nodes = [] # For expressions that are not string literals themselves

        for i, expr in enumerate(node.expressions):
            if isinstance(expr, StringLiteralNode):
                format_string_parts.append(expr.value.replace("%", "%%"))
            elif isinstance(expr, IntegerLiteralNode):
                format_string_parts.append("%d")
                arg_expr_nodes.append(expr)
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
                elif var_type == 'file':
                    # For file pointers, print as pointer value
                    format_string_parts.append("FILE*@%p")
                else: # Default to %d for int or unknown/other types for now
                    format_string_parts.append("%d")
                arg_expr_nodes.append(expr)
            elif isinstance(expr, FunctionCallNode):
                # Determine the function return type to use %s or %d
                func_name = expr.name
                if func_name == "input":  # input() returns string
                    format_string_parts.append("%s")
                elif func_name == "to_int":  # to_int() returns int
                    format_string_parts.append("%d")
                elif func_name == "to_string":  # to_string() returns string
                    format_string_parts.append("%s")
                elif func_name == "open":  # open() returns file
                    format_string_parts.append("FILE*@%p")
                else:  # Default to %d for unknown functions
                    format_string_parts.append("%d")
                arg_expr_nodes.append(expr)
            elif isinstance(expr, (MethodCallNode, BinaryOpNode, ThisNode)):
                # Assume these currently produce integer results or types handled by %d
                # Future: MethodCallNode would need return type lookup for %s/%d choice
                # ThisNode currently resolves to an address (pointer), printing as %d is okay for now.
                format_string_parts.append("%d") 
                arg_expr_nodes.append(expr)
            elif isinstance(expr, MacroInvokeNode):
                # For macro invocations, we'll assume they'll resolve to integers for now
                format_string_parts.append("%d")
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
        if node.expression: ret_assembly += self.visit(node.expression, context)
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
        if var_name not in self.current_method_locals:
            raise CompilerError(f"Internal Compiler Error: Local variable '{var_name}' was not pre-allocated space. L{node.lineno}")

        local_info = self.current_method_locals[var_name]
        var_offset_rbp = local_info['offset_rbp']

        decl_assembly = f"  # Variable Declaration: {var_name} ({node.type}) at L{node.lineno}\n"
        
        # 1. Evaluate the right-hand side expression (initialization value)
        decl_assembly += self.visit(node.expression, context) # Result will be in RAX
        
        # 2. Store the result from RAX into the variable's pre-allocated stack slot
        #    No need to adjust RSP here, space is already made in prologue.
        decl_assembly += f"  mov QWORD PTR [rbp {var_offset_rbp}], rax # Initialize {var_name} = RAX at [rbp{var_offset_rbp}]\n"
        
        self.current_method_locals[var_name]['is_initialized'] = True # Mark as initialized
        return decl_assembly

    def visit_MethodCallNode(self, node: MethodCallNode, context):
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
        
        # Check if this is a string concatenation
        is_string_concat = False
        if node.op_token.type == TT_PLUS:
            left_type = self.get_expr_type(node.left, context)
            right_type = self.get_expr_type(node.right, context)
            if left_type == "string" or right_type == "string":
                is_string_concat = True
        
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
        # Default for unknown/unhandled expressions
        return "unknown"

    def visit_IfNode(self, node: IfNode, context):
        if_assembly = f"  # If statement at L{node.lineno}\n"
        else_label, end_if_label = self.new_if_labels()
        if_assembly += self.visit(node.condition, context)
        jump_instruction = ""
        op_type = node.condition.op_token.type
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

        if not isinstance(node.operand_node, IdentifierNode):
            # For now, only allow inc/dec on simple identifiers (variables).
            # Future: could extend to array elements or fields if they become L-values.
            raise CompilerError(f"Operand of ++/-- must be an identifier, got {type(node.operand_node)} at L{node.lineno}")

        var_name = node.operand_node.value
        var_info = None
        assembly_code = f"  # UnaryOp {node.op_token.type} on {var_name} at L{node.lineno}\n"

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
        # Assignment statement: variable = expression
        code = f"  # Assignment to {node.name} at L{node.lineno}\n"
        # Evaluate RHS
        code += self.visit(node.expression, context)
        # Determine variable location
        var_name = node.name
        if var_name in self.current_method_locals:
            offset = self.current_method_locals[var_name]['offset_rbp']
        elif var_name in self.current_method_params:
            offset = self.current_method_params[var_name]['offset_rbp']
        else:
            raise CompilerError(f"Assignment to undefined variable '{var_name}' at L{node.lineno}")
        # Store result
        code += f"  mov QWORD PTR [rbp {offset}], rax # {var_name} = RAX\n"
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

    def compile(self, node: ProgramNode):
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

