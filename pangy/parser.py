from .lexer import (
    Lexer, Token,
    TT_CLASS, TT_DEF, TT_IDENTIFIER, TT_LPAREN, TT_RPAREN, TT_LBRACE, TT_RBRACE, TT_ARROW, 
    TT_TYPE, TT_STRING_LITERAL, TT_PRINT, TT_SEMICOLON, TT_EOF, TT_MACRO, TT_AT, 
    TT_PUBLIC, TT_PRIVATE, TT_INT_LITERAL, TT_COMMA, TT_COLON, TT_DOUBLE_COLON, TT_PLUS, 
    TT_RETURN, TT_THIS, TT_ASSIGN, TT_VAR, TT_MINUS, TT_STAR, TT_SLASH, TT_PERCENT, 
    TT_PLUSPLUS, TT_MINUSMINUS, TT_IF, TT_ELSE, TT_LESS_THAN, TT_GREATER_THAN, TT_EQUAL, 
    TT_NOT_EQUAL, TT_LESS_EQUAL, TT_GREATER_EQUAL, TT_STATIC, TT_DOT, TT_INCLUDE, 
    TT_LOOP, TT_WHILE, TT_STOP, TT_LBRACKET, TT_RBRACKET, TT_AMPERSAND, TT_PIPE, TT_CARET, 
    TT_TILDE, TT_LSHIFT, TT_RSHIFT, TT_URSHIFT, TT_LOGICAL_AND, TT_LOGICAL_OR, 
    TT_FLOAT_LITERAL,
    TT_TRUE, TT_FALSE,
    TT_USE
)

# AST Node base class
class ASTNode:
    pass

class ProgramNode(ASTNode):
    def __init__(self, classes, includes=None, macros=None): # Added macros
        self.classes = classes # List of ClassNode
        self.includes = includes if includes is not None else [] # List of IncludeNode
        self.macros = macros if macros is not None else [] # List of MacroDefNode

    def __repr__(self):
        return f"ProgramNode(classes={self.classes}, includes={self.includes}, macros={self.macros})"

class IncludeNode(ASTNode): # New AST Node for include directives
    def __init__(self, path, lineno):
        self.path = path # The path (filename.Class.InnerClass or filename.@macroname)
        self.lineno = lineno
    
    def __repr__(self):
        return f"IncludeNode(path='{self.path}', L{self.lineno})"

class ClassNode(ASTNode):
    def __init__(self, name_token, methods, nested_classes=None):
        self.name = name_token.value
        self.name_token = name_token
        self.methods = methods  # List of MethodNode
        self.nested_classes = nested_classes if nested_classes is not None else []  # List of nested ClassNode
        self.macros = []  # List of MacroDefNode
        self.class_vars = []  # List of ClassVarDeclNode

    def __repr__(self):
        return f"ClassNode(name='{self.name}', methods={self.methods}, nested_classes={self.nested_classes}, macros={self.macros}, class_vars={self.class_vars})"

class ParamNode(ASTNode):
    def __init__(self, name_token, type_token):
        self.name = name_token.value
        self.name_token = name_token
        self.type = type_token.value # This will become a string like "int" or "int[]"
        self.type_token = type_token # May need adjustment if type_token is synthetic for arrays

    def __repr__(self):
        return f"ParamNode(name='{self.name}', type='{self.type}')"

class MethodNode(ASTNode):
    def __init__(self, name_token, params, return_type_token, body, is_static=False):
        self.name = name_token.value
        self.name_token = name_token
        self.params = params # List of ParamNode
        self.return_type = return_type_token.value # This will become a string
        self.return_type_token = return_type_token # May need adjustment
        self.body = body # BlockNode
        self.is_static = is_static

    def __repr__(self):
        return f"MethodNode(name='{self.name}', params={self.params}, return_type='{self.return_type}', body={self.body}, static={self.is_static})"

class BlockNode(ASTNode):
    def __init__(self, statements):
        self.statements = statements # List of statement nodes

    def __repr__(self):
        return f"BlockNode(statements={self.statements})"

class PrintNode(ASTNode):
    # Now takes a list of expressions to print
    def __init__(self, expressions, lineno):
        self.expressions = expressions # List of expression ASTNodes
        self.lineno = lineno

    def __repr__(self):
        return f"PrintNode(expressions={self.expressions})"

class ReturnNode(ASTNode):
    def __init__(self, expression_node, lineno):
        self.expression = expression_node # ASTNode for the expression to return
        self.lineno = lineno
    
    def __repr__(self):
        return f"ReturnNode(expression={self.expression})"

# --- Expression Nodes ---
class ExprNode(ASTNode): # Base for all expression nodes
    pass

class StringLiteralNode(ExprNode):
    def __init__(self, token):
        self.value = token.value
        self.token = token

    def __repr__(self):
        return f"StringLiteralNode(value='{self.value}')"

class IntegerLiteralNode(ExprNode):
    def __init__(self, token):
        self.value = token.value
        self.token = token

    def __repr__(self):
        return f"IntegerLiteralNode(value={self.value})"

class FloatLiteralNode(ExprNode):
    def __init__(self, token):
        self.value = token.value
        self.token = token

    def __repr__(self):
        return f"FloatLiteralNode(value={self.value})"

class IdentifierNode(ExprNode):
    def __init__(self, token):
        self.value = token.value
        self.token = token

    def __repr__(self):
        return f"IdentifierNode(value='{self.value}')"

class ThisNode(ExprNode):
    def __init__(self, token):
        self.token = token
    def __repr__(self):
        return f"ThisNode(token={self.token})"


class MethodCallNode(ExprNode):
    def __init__(self, object_expr_node, method_name_token, argument_expr_nodes):
        self.object_expr = object_expr_node # e.g., ThisNode or IdentifierNode for an object variable
        self.method_name = method_name_token.value
        self.method_name_token = method_name_token
        self.arguments = argument_expr_nodes # List of ExprNode

    def __repr__(self):
        return f"MethodCallNode(object={self.object_expr}, method='{self.method_name}', args={self.arguments})"

# Insert new AST node for free function calls
class FunctionCallNode(ExprNode):
    def __init__(self, name_token, argument_expr_nodes):
        self.name = name_token.value
        self.name_token = name_token
        self.arguments = argument_expr_nodes

    def __repr__(self):
        return f"FunctionCallNode(name='{self.name}', args={self.arguments})"

class BinaryOpNode(ExprNode):
    def __init__(self, left_node, op_token, right_node):
        self.left = left_node # Changed back to self.left
        self.op_token = op_token
        self.op = op_token.value # Changed back to self.op
        self.right = right_node # Changed back to self.right
        self.lineno = op_token.lineno

    def __repr__(self):
        return f"BinaryOpNode(left={self.left}, op='{self.op}', right={self.right})"

class UnaryOpNode(ExprNode):
    def __init__(self, op_token, operand_node):
        self.op_token = op_token
        self.operand_node = operand_node
        self.lineno = op_token.lineno

    def __repr__(self):
        return f"UnaryOpNode({self.op_token.type}, {self.operand_node})"

# New AST Node for If statements
class IfNode(ASTNode):
    def __init__(self, condition, then_block, else_block, lineno):
        self.condition = condition
        self.then_block = then_block
        self.else_block = else_block # Can be None or another IfNode for else-if
        self.lineno = lineno

    def __repr__(self):
        return f"IfNode(condition={self.condition}, then_block={self.then_block}, else_block={self.else_block}, L{self.lineno})"

# --- Nodes for Variable Declarations ---
class VarDeclNode(ASTNode): # Statement Node
    def __init__(self, name_token, type_token, expr_node, lineno, is_constant=False):
        self.name = name_token.value
        self.name_token = name_token
        self.type = type_token.value # This will become a string
        self.type_token = type_token # May need adjustment
        self.expression = expr_node # The expression to initialize the variable
        self.lineno = lineno
        self.is_constant = is_constant # Flag to indicate if this is a constant variable

    def __repr__(self):
        const_str = "*" if self.is_constant else ""
        return f"VarDeclNode(name='{const_str}{self.name}', type='{self.type}', expr={self.expression}, L{self.lineno})"

# AST Nodes for Loops, Stop, and Assignment
class LoopNode(ASTNode):
    def __init__(self, body, lineno):
        self.body = body  # BlockNode
        self.lineno = lineno
    def __repr__(self):
        return f"LoopNode(body={self.body}, L{self.lineno})"

class WhileNode(ASTNode):
    def __init__(self, condition, body, lineno):
        self.condition = condition
        self.body = body  # BlockNode
        self.lineno = lineno
    def __repr__(self):
        return f"WhileNode(condition={self.condition}, body={self.body}, L{self.lineno})"

class StopNode(ASTNode):
    def __init__(self, lineno):
        self.lineno = lineno
    def __repr__(self):
        return f"StopNode(L{self.lineno})"

class AssignmentNode(ASTNode):
    def __init__(self, target, expression, lineno):
        self.target = target  # IdentifierNode, ArrayAccessNode, or ClassVarAccessNode
        self.expression = expression
        self.lineno = lineno
    
    def __repr__(self):
        return f"AssignmentNode(target={self.target}, expr={self.expression}, L{self.lineno})"

# Macro-related AST nodes
class MacroDefNode(ASTNode):
    def __init__(self, name_token, params, body, lineno):
        self.name = name_token.value
        self.name_token = name_token
        self.params = params  # List of parameter names (strings)
        self.body = body      # BlockNode or expression
        self.lineno = lineno
        self.class_name = None  # Will be set for class macros
        self.qualified_name = None  # Will be set for class macros
    
    def __repr__(self):
        class_prefix = f"{self.class_name}." if self.class_name else ""
        return f"MacroDefNode(name='{class_prefix}{self.name}', params={self.params}, body={self.body}, L{self.lineno})"

class MacroInvokeNode(ExprNode):
    def __init__(self, name_token, arguments, lineno, object_expr=None):
        self.name = name_token.value
        self.name_token = name_token
        self.arguments = arguments  # List of ExprNode
        self.lineno = lineno
        self.object_expr = object_expr  # The object expression for method calls (e.g., this or an object variable)
    
    def __repr__(self):
        object_str = f"{self.object_expr}:" if self.object_expr else ""
        return f"MacroInvokeNode(name='{object_str}@{self.name}', args={self.arguments}, L{self.lineno})"

# List-related AST nodes
class ListLiteralNode(ExprNode):
    def __init__(self, elements, lbrace_token):
        self.elements = elements # List of ExprNode
        self.lbrace_token = lbrace_token
        self.lineno = lbrace_token.lineno

    def __repr__(self):
        return f"ListLiteralNode(elements={self.elements}, L{self.lineno})"

class ArrayAccessNode(ExprNode):
    def __init__(self, array_expr, index_expr, lbracket_token):
        self.array_expr = array_expr
        self.index_expr = index_expr
        self.lbracket_token = lbracket_token
        self.lineno = lbracket_token.lineno

    def __repr__(self):
        return f"ArrayAccessNode(array={self.array_expr}, index={self.index_expr}, L{self.lineno})"

class ClassVarDeclNode(ASTNode):
    def __init__(self, name_token, type_token, expr_node, is_public, lineno):
        self.name = name_token.value
        self.name_token = name_token
        self.type = type_token.value
        self.type_token = type_token
        self.expression = expr_node  # The expression to initialize the variable
        self.is_public = is_public   # Boolean indicating public (True) or private (False)
        self.lineno = lineno

    def __repr__(self):
        visibility = "public" if self.is_public else "private"
        return f"ClassVarDeclNode(name='{self.name}', type='{self.type}', is_public={self.is_public}, expr={self.expression}, L{self.lineno})"

class ClassVarAccessNode(ExprNode):
    def __init__(self, object_expr, var_name_token):
        self.object_expr = object_expr  # The object expression (e.g., this, or an object variable)
        self.var_name = var_name_token.value
        self.var_name_token = var_name_token
        self.lineno = var_name_token.lineno

    def __repr__(self):
        return f"ClassVarAccessNode(object={self.object_expr}, var_name='{self.var_name}', L{self.lineno})"

class TrueLiteralNode(ExprNode):
    def __init__(self, token):
        self.value = True
        self.token = token

    def __repr__(self):
        return f"TrueLiteralNode(value=True)"

class FalseLiteralNode(ExprNode):
    def __init__(self, token):
        self.value = False
        self.token = token

    def __repr__(self):
        return f"FalseLiteralNode(value=False)"

# Node for C library calls
class CLibraryCallNode(ExprNode):
    def __init__(self, library_token, function_name_token, argument_expr_nodes, lineno):
        self.library = library_token.value  # e.g., "m" for math library
        self.library_token = library_token
        self.function_name = function_name_token.value
        self.function_name_token = function_name_token
        self.arguments = argument_expr_nodes  # List of ExprNode
        self.lineno = lineno

    def __repr__(self):
        return f"CLibraryCallNode(lib='{self.library}', func='{self.function_name}', args={self.arguments}, L{self.lineno})"

# Parser
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.token_idx = 0
        # Ensure tokens list is not empty before accessing, handle empty source case
        self.current_token = self.tokens[self.token_idx] if self.tokens else Token(TT_EOF, None)

    def advance(self):
        self.token_idx += 1
        if self.token_idx < len(self.tokens):
            self.current_token = self.tokens[self.token_idx]
        else:
            # This should be fine as long as TT_EOF is the last token
            self.current_token = self.tokens[-1] if self.tokens else Token(TT_EOF, None)

    def consume(self, token_type):
        if self.current_token.type == token_type:
            token = self.current_token
            self.advance()
            return token
        else:
            expected = token_type
            found = self.current_token.type
            val = self.current_token.value
            line = self.current_token.lineno
            col = self.current_token.colno
            raise Exception(f"ParserError: Expected token {expected} but got {found} ('{val}') at L{line}:C{col}")

    def parse_program(self):
        # program ::= include_directive* macro_definition* class_definition*
        includes = []
        while self.current_token.type == TT_INCLUDE:
            includes.append(self.parse_include_directive())

        # Parse macro definitions
        macros = []
        while self.current_token.type == TT_MACRO:
            macros.append(self.parse_macro_definition())

        # Parse top-level class definitions
        raw_classes = []
        while self.current_token.type == TT_CLASS:
            raw_classes.append(self.parse_class_definition())

        if self.current_token.type != TT_EOF:
            tok = self.current_token
            raise Exception(f"ParserError: Expected EOF, include, macro, or class definition, but got {tok.type} ('{tok.value}') at L{tok.lineno}:C{tok.colno}")

        # Flatten nested classes into a single list with qualified names
        def flatten_class(c, prefix=None):
            # Update class name if nested
            if prefix:
                c.name = f"{prefix}.{c.name}"
                
                # Update macro names with class prefix
                for macro in c.macros:
                    macro.qualified_name = f"{c.name}_{macro.name}"
                
            classes = [c]
            for nc in getattr(c, 'nested_classes', []):
                classes.extend(flatten_class(nc, c.name))
            return classes

        classes = []
        for rc in raw_classes:
            classes.extend(flatten_class(rc))

        # Collect all macros from classes
        for c in classes:
            for macro in getattr(c, 'macros', []):
                macro.class_name = c.name  # Associate macro with its class
                macros.append(macro)

        return ProgramNode(classes, includes, macros)  # Pass all macros to ProgramNode

    def parse_include_directive(self):
        # include_directive ::= "include" IDENTIFIER ("." IDENTIFIER | "." "@" IDENTIFIER)*
        include_token = self.consume(TT_INCLUDE)
        
        # Parse the first part (filename)
        first_token = self.consume(TT_IDENTIFIER)
        path = first_token.value
        
        # Parse optional parts (Class.InnerClass or @macroname)
        while self.current_token.type == TT_DOT:
            self.consume(TT_DOT)
            
            # Check for macro (@name)
            if self.current_token.type == TT_AT:
                self.consume(TT_AT)
                macro_name = self.consume(TT_IDENTIFIER)
                path += f".@{macro_name.value}"
            else:
                # Class or InnerClass
                class_name = self.consume(TT_IDENTIFIER)
                path += f".{class_name.value}"
        
        return IncludeNode(path, include_token.lineno)

    def parse_macro_definition(self):
        # macro_definition ::= "macro" IDENTIFIER "(" macro_params ")" "{" expression "}"
        macro_token = self.consume(TT_MACRO)
        name_token = self.consume(TT_IDENTIFIER)
        self.consume(TT_LPAREN)
        
        # Parse macro parameters (comma-separated identifiers)
        params = []
        if self.current_token.type == TT_IDENTIFIER:
            param_token = self.consume(TT_IDENTIFIER)
            params.append(param_token.value)
            while self.current_token.type == TT_COMMA:
                self.consume(TT_COMMA)
                param_token = self.consume(TT_IDENTIFIER)
                params.append(param_token.value)
        
        self.consume(TT_RPAREN)
        self.consume(TT_LBRACE)
        
        # Parse macro body (single expression for now)
        body = self.parse_expression()
        
        self.consume(TT_RBRACE)
        
        return MacroDefNode(name_token, params, body, macro_token.lineno)

    def parse_class_definition(self):
        # class_definition ::= "class" IDENTIFIER "{" class_definition* class_var_declaration* method_definition* macro_definition* "}"
        self.consume(TT_CLASS)
        name_token = self.consume(TT_IDENTIFIER)
        self.consume(TT_LBRACE)  # '{'

        nested_classes = []
        methods = []
        macros = []
        class_vars = []  # Added for class-level variables
        
        # Loop for nested classes, method definitions, macro definitions, or class var declarations
        while self.current_token.type in (TT_CLASS, TT_STATIC, TT_DEF, TT_MACRO, TT_PUBLIC, TT_PRIVATE):
            if self.current_token.type == TT_CLASS:
                nested_classes.append(self.parse_class_definition())
            elif self.current_token.type == TT_MACRO:
                macros.append(self.parse_macro_definition())
            elif self.current_token.type in (TT_PUBLIC, TT_PRIVATE):
                class_vars.append(self.parse_class_var_declaration())
            else:  # TT_STATIC or TT_DEF
                methods.append(self.parse_method_definition())

        self.consume(TT_RBRACE)  # '}'
        class_node = ClassNode(name_token, methods, nested_classes)
        class_node.macros = macros
        class_node.class_vars = class_vars  # Store the class variables
        return class_node

    def parse_parameters(self):
        # parameters ::= (IDENTIFIER (TYPE | IDENTIFIER) ("." IDENTIFIER)* ("," IDENTIFIER (TYPE | IDENTIFIER) ("." IDENTIFIER)*)*)?
        params = []
        if self.current_token.type == TT_IDENTIFIER:
            # Parameter name
            name_token = self.consume(TT_IDENTIFIER)
            # Parameter type
            type_string = self.parse_type_specifier()
            # Create a synthetic token for ParamNode for now, or change ParamNode
            # Let's change ParamNode to accept type_string directly.
            params.append(ParamNode(name_token, Token(TT_TYPE, type_string, name_token.lineno, name_token.colno))) # Temporary token
            
            # Additional parameters separated by commas
            while self.current_token.type == TT_COMMA:
                self.consume(TT_COMMA)
                name_token = self.consume(TT_IDENTIFIER)
                type_string = self.parse_type_specifier()
                params.append(ParamNode(name_token, Token(TT_TYPE, type_string, name_token.lineno, name_token.colno))) # Temporary token
        return params

    def parse_method_definition(self):
        # method_definition ::= ("static")? "def" IDENTIFIER "(" parameters? ")" "->" TYPE "{" statement* "}"
        is_static = False
        if self.current_token.type == TT_STATIC:
            self.consume(TT_STATIC)
            is_static = True
        
        self.consume(TT_DEF)
        name_token = self.consume(TT_IDENTIFIER)
        self.consume(TT_LPAREN)
        
        params = []
        if self.current_token.type != TT_RPAREN:
            params = self.parse_parameters()

        self.consume(TT_RPAREN)
        # Return type (primitive or class)
        self.consume(TT_ARROW)
        
        # Use parse_type_specifier for return type
        return_type_string = self.parse_type_specifier()
        # Create a synthetic token for MethodNode. 
        # Using name_token's line for the synthetic type token is an approximation for lineno.
        # The column number might not be perfectly accurate for complex types.
        # However, the MethodNode's `return_type` field (string) will be accurate.
        return_type_token = Token(TT_TYPE, return_type_string, name_token.lineno, name_token.colno) # Synthetic token
        
        body = self.parse_block()

        return MethodNode(name_token, params, return_type_token, body, is_static)

    def parse_assignment_or_expression_statement(self):
        start_lineno = self.current_token.lineno # For AssignmentNode if it is one
        
        # Parse the initial expression (could be LHS of assignment or a full expression statement)
        # parse_expression ultimately calls parse_term, which handles IDENTIFIER, array access, etc.
        # This also correctly handles chained array accesses like mymatrix[0][1]
        lhs_expression = self.parse_expression() 

        # After parsing the LHS expression, check if the next token is an assignment operator
        if self.current_token.type == TT_ASSIGN:
            # It's an assignment statement
            # Ensure lhs_expression is a valid L-value (target for assignment)
            if not isinstance(lhs_expression, (IdentifierNode, ArrayAccessNode, ClassVarAccessNode)):
                err_lineno = getattr(lhs_expression, 'lineno', start_lineno)
                # Attempt to get a more specific line number from the token if available
                if hasattr(lhs_expression, 'token') and hasattr(lhs_expression.token, 'lineno'):
                    err_lineno = lhs_expression.token.lineno
                elif hasattr(lhs_expression, 'lbracket_token') and hasattr(lhs_expression.lbracket_token, 'lineno'): # For ArrayAccessNode
                    err_lineno = lhs_expression.lbracket_token.lineno
                elif hasattr(lhs_expression, 'var_name_token') and hasattr(lhs_expression.var_name_token, 'lineno'): # For ClassVarAccessNode
                    err_lineno = lhs_expression.var_name_token.lineno
                
                raise Exception(f"ParserError: Invalid L-value for assignment. LHS is type {type(lhs_expression)} at L{err_lineno}")

            self.consume(TT_ASSIGN) # Consume the '='
            rhs_expression = self.parse_expression() # Parse the RHS
            return AssignmentNode(lhs_expression, rhs_expression, start_lineno)
        else:
            # It's an expression statement. The parsed lhs_expression is the statement itself.
            return lhs_expression

    def parse_statement(self):
        # statement ::= print_statement | return_statement | if_statement | loop_statement | stop_statement | var_declaration | assignment | expression_statement
        start_token = self.current_token
        if start_token.type == TT_PRINT:
            return self.parse_print_statement()
        elif start_token.type == TT_RETURN:
            return self.parse_return_statement()
        elif start_token.type == TT_IF:
            return self.parse_if_statement()
        elif start_token.type == TT_LOOP:
            return self.parse_loop_statement()
        elif start_token.type == TT_WHILE:
            return self.parse_while_statement()
        elif start_token.type == TT_STOP:
            return self.parse_stop_statement()
        elif start_token.type == TT_VAR: # Added for variable declaration
            return self.parse_var_declaration_statement()
        
        # For other cases, it's either an assignment or an expression statement.
        # The following token types can start an expression (and thus, an assignment or expr statement)
        if start_token.type in [
            TT_THIS, TT_IDENTIFIER, TT_LPAREN, # Common expression starts
            TT_INT_LITERAL, TT_STRING_LITERAL, TT_TRUE, TT_FALSE, # Literals
            TT_LBRACE, # List literals
            TT_MINUS, TT_TILDE, # Unary operators
            TT_AT, # Macro invocations like @myMacro()
            TT_PLUSPLUS, TT_MINUSMINUS # Postfix ops can be part of an expr statement, though value often unused
        ]:
            return self.parse_assignment_or_expression_statement()
        else:
            # No self.error method; raise exception instead
            raise Exception(f"ParserError: Unexpected token {start_token.type} ('{start_token.value}') cannot start a statement at L{start_token.lineno}:C{start_token.colno}")

    def parse_print_statement(self):
        # print_statement ::= "print" "(" arguments? ")"
        print_token = self.consume(TT_PRINT)
        self.consume(TT_LPAREN)
        expressions = []
        if self.current_token.type != TT_RPAREN:
            expressions = self.parse_argument_list()
        self.consume(TT_RPAREN)
        # Optional semicolon
        # if self.current_token.type == TT_SEMICOLON: self.consume(TT_SEMICOLON)
        return PrintNode(expressions, print_token.lineno)

    def parse_return_statement(self):
        # return_statement ::= "return" expression
        return_token = self.consume(TT_RETURN)
        expr_node = self.parse_expression()
        # Optional semicolon
        # if self.current_token.type == TT_SEMICOLON: self.consume(TT_SEMICOLON)
        return ReturnNode(expr_node, return_token.lineno)

    def parse_if_statement(self):
        # if_statement ::= "if" "(" expression ")" block ( "else" if_statement | "else" block )?
        if_token = self.consume(TT_IF)
        self.consume(TT_LPAREN)
        condition_expr = self.parse_expression()
        self.consume(TT_RPAREN)
        then_block = self.parse_block()
        
        else_block = None
        if self.current_token.type == TT_ELSE:
            self.consume(TT_ELSE)
            if self.current_token.type == TT_IF: # Handling "else if"
                # The "else" part is another full "if" statement.
                else_block = self.parse_if_statement()
            else: # Handling simple "else { ... }"
                else_block = self.parse_block()
            
        return IfNode(condition_expr, then_block, else_block, if_token.lineno)

    def parse_var_declaration_statement(self):
        # var_declaration ::= "var" ["*"] IDENTIFIER TYPE ( "." IDENTIFIER )* ("=" expression)?
        self.consume(TT_VAR)
        
        # Check if this is a constant variable declaration (indicated by *)
        is_constant = False
        if self.current_token.type == TT_STAR:
            is_constant = True
            self.consume(TT_STAR)
            
        name_token = self.consume(TT_IDENTIFIER)

        # Parse type using the new helper
        type_string = self.parse_type_specifier()
        # Create a synthetic token for VarDeclNode for now, or change VarDeclNode
        # Let's change VarDeclNode to accept type_string.
        type_token = Token(TT_TYPE, type_string, name_token.lineno, name_token.colno) # Temporary token

        expr_node = None
        if self.current_token.type == TT_ASSIGN:
            self.consume(TT_ASSIGN)
            expr_node = self.parse_expression()

        return VarDeclNode(name_token, type_token, expr_node, name_token.lineno, is_constant)

    def parse_loop_statement(self):
        # loop_statement ::= "loop" block
        loop_token = self.consume(TT_LOOP)
        body = self.parse_block()
        return LoopNode(body, loop_token.lineno)

    def parse_while_statement(self):
        # while_statement ::= "while" "(" expression ")" block
        start_token = self.consume(TT_WHILE)
        self.consume(TT_LPAREN)
        condition = self.parse_expression()
        self.consume(TT_RPAREN)
        body = self.parse_block()
        return WhileNode(condition, body, start_token.lineno)

    def parse_stop_statement(self):
        # stop_statement ::= "stop"
        stop_token = self.consume(TT_STOP)
        return StopNode(stop_token.lineno)

    def parse_block(self):
        # block ::= "{" statement* "}"
        self.consume(TT_LBRACE)
        statements = []
        while self.current_token.type not in [TT_RBRACE, TT_EOF]:
            statements.append(self.parse_statement())
        self.consume(TT_RBRACE)
        return BlockNode(statements)

    def parse_argument_list(self):
        # arguments ::= expression ("," expression)*
        args = [self.parse_expression()]
        while self.current_token.type == TT_COMMA:
            self.consume(TT_COMMA)
            args.append(self.parse_expression())
        return args

    # --- Expression Parsing (precedence: Comparison < Additive < Term) ---
    def parse_expression(self):
        # Entry point for parsing any expression
        return self.parse_logical_or_expression()

    def parse_logical_or_expression(self):
        # logical_or_expression ::= logical_and_expression ( "||" logical_and_expression )*
        node = self.parse_logical_and_expression()
        while self.current_token.type == TT_LOGICAL_OR:
            op_token = self.consume(TT_LOGICAL_OR)
            right_node = self.parse_logical_and_expression()
            node = BinaryOpNode(node, op_token, right_node)
        return node

    def parse_logical_and_expression(self):
        # logical_and_expression ::= comparison_expression ( "&&" comparison_expression )*
        node = self.parse_comparison_expression()
        while self.current_token.type == TT_LOGICAL_AND:
            op_token = self.consume(TT_LOGICAL_AND)
            right_node = self.parse_comparison_expression()
            node = BinaryOpNode(node, op_token, right_node)
        return node

    def parse_comparison_expression(self):
        # comparison_expression ::= bitwise_expression ( (LT | GT | LE | GE | EQ | NE) bitwise_expression )*
        node = self.parse_bitwise_expression()

        while self.current_token.type in [TT_LESS_THAN, TT_GREATER_THAN, TT_LESS_EQUAL, TT_GREATER_EQUAL, TT_EQUAL, TT_NOT_EQUAL]:
            op_token = self.consume(self.current_token.type) # Consume comparison operator
            right_node = self.parse_bitwise_expression()
            node = BinaryOpNode(node, op_token, right_node)
        return node

    def parse_bitwise_expression(self):
        # bitwise_expression ::= shift_expression ( (& | | | ^) shift_expression )*
        node = self.parse_shift_expression()

        while self.current_token.type in [TT_AMPERSAND, TT_PIPE, TT_CARET]:
            op_token = self.consume(self.current_token.type) # Consume bitwise operator
            right_node = self.parse_shift_expression()
            node = BinaryOpNode(node, op_token, right_node)
        return node
        
    def parse_shift_expression(self):
        # shift_expression ::= additive_expression ( (<< | >> | >>>) additive_expression )*
        node = self.parse_additive_expression()

        while self.current_token.type in [TT_LSHIFT, TT_RSHIFT, TT_URSHIFT]:
            op_token = self.consume(self.current_token.type) # Consume shift operator
            right_node = self.parse_additive_expression()
            node = BinaryOpNode(node, op_token, right_node)
        return node

    def parse_additive_expression(self):
        # additive_expression ::= multiplicative_expression ( (PLUS | MINUS) multiplicative_expression )*
        node = self.parse_multiplicative_expression()

        while self.current_token.type in [TT_PLUS, TT_MINUS]: 
            op_token = self.consume(self.current_token.type) # Consume + or -
            right_node = self.parse_multiplicative_expression() 
            node = BinaryOpNode(node, op_token, right_node)
        return node

    def parse_multiplicative_expression(self):
        # multiplicative_expression ::= unary_expression ( (STAR | SLASH | PERCENT) unary_expression )*
        node = self.parse_unary_expression()

        while self.current_token.type in [TT_STAR, TT_SLASH, TT_PERCENT]:
            op_token = self.consume(self.current_token.type) # Consume *, /, or %
            right_node = self.parse_unary_expression()
            node = BinaryOpNode(node, op_token, right_node)
        return node
        
    def parse_unary_expression(self):
        # unary_expression ::= (MINUS | TILDE)? term
        if self.current_token.type in (TT_MINUS, TT_TILDE):
            op_token = self.current_token
            self.advance()  # Consume '-' or '~'
            # Parse the operand as a unary expression to handle things like - -5 or -~5
            operand = self.parse_unary_expression()
            return UnaryOpNode(op_token, operand)
        return self.parse_term()
        
    def parse_term(self):
        # term ::= primary ( ( "." | ":" | "::" ) IDENTIFIER | "++" | "--" | "(" arguments? ")" | "[" expression "]" )*
        node = self.parse_primary()

        while self.current_token.type in (TT_DOT, TT_COLON, TT_DOUBLE_COLON, TT_PLUSPLUS, TT_MINUSMINUS, TT_LPAREN, TT_AT, TT_LBRACKET):
            # Handle postfix increment/decrement
            if self.current_token.type in (TT_PLUSPLUS, TT_MINUSMINUS):
                op_token = self.current_token
                self.advance()  # Consume '++' or '--'
                node = UnaryOpNode(op_token, node)
                continue

            # Handle dot operator for method calls or member access
            if self.current_token.type == TT_DOT:
                self.advance()  # Consume '.'
                
                # Check for @macro invocation after the dot
                if self.current_token.type == TT_AT:
                    self.advance()  # Consume '@'
                    macro_name_token = self.consume(TT_IDENTIFIER)
                    self.consume(TT_LPAREN)
                    args = []
                    if self.current_token.type != TT_RPAREN:
                        args = self.parse_argument_list()
                    self.consume(TT_RPAREN)
                    node = MacroInvokeNode(macro_name_token, args, macro_name_token.lineno, node)
                    continue
                
                # Check for method call vs member access
                if (self.token_idx + 1 < len(self.tokens) and 
                    self.tokens[self.token_idx + 1].type == TT_LPAREN):
                    # Method call on identifier
                    method_name_token = self.consume(TT_IDENTIFIER)
                    self.consume(TT_LPAREN)
                    args = []
                    if self.current_token.type != TT_RPAREN:
                        args = self.parse_argument_list()
                    self.consume(TT_RPAREN)
                    node = MethodCallNode(node, method_name_token, args)
                else:
                    # Member access (e.g., ClassName.InnerClass)
                    ident_token = self.consume(TT_IDENTIFIER)
                    # Merge into composite identifier
                    composite_value = f"{node.value}.{ident_token.value}"
                    node = IdentifierNode(Token(TT_IDENTIFIER, composite_value, ident_token.lineno, ident_token.colno))
                continue

            # Handle this:method() or this:@macro() or this:varname calls
            if self.current_token.type == TT_COLON:
                self.advance()  # Consume ':'
                
                # Check for @macro invocation after the colon
                if self.current_token.type == TT_AT:
                    self.advance()  # Consume '@'
                    macro_name_token = self.consume(TT_IDENTIFIER)
                    self.consume(TT_LPAREN)
                    args = []
                    if self.current_token.type != TT_RPAREN:
                        args = self.parse_argument_list()
                    self.consume(TT_RPAREN)
                    node = MacroInvokeNode(macro_name_token, args, macro_name_token.lineno, node)
                    continue
                
                # Check if it's a method call or variable access
                next_token_is_lparen = (self.token_idx + 1 < len(self.tokens) and 
                                       self.tokens[self.token_idx + 1].type == TT_LPAREN)
                
                if next_token_is_lparen:
                    # Method call
                    method_name_token = self.consume(TT_IDENTIFIER)
                    self.consume(TT_LPAREN)
                    args = []
                    if self.current_token.type != TT_RPAREN:
                        args = self.parse_argument_list()
                    self.consume(TT_RPAREN)
                    node = MethodCallNode(node, method_name_token, args)
                else:
                    # Class variable access with this:varname
                    var_name_token = self.consume(TT_IDENTIFIER)
                    node = ClassVarAccessNode(node, var_name_token)
                continue
                
            # Handle objectname::varname for accessing class variables from outside
            if self.current_token.type == TT_DOUBLE_COLON:
                self.advance()  # Consume '::'
                var_name_token = self.consume(TT_IDENTIFIER)
                node = ClassVarAccessNode(node, var_name_token)
                continue

            # Handle direct function calls
            if self.current_token.type == TT_LPAREN:
                self.consume(TT_LPAREN)
                args = []
                if self.current_token.type != TT_RPAREN:
                    args = self.parse_argument_list()
                self.consume(TT_RPAREN)
                if isinstance(node, IdentifierNode):
                    node = FunctionCallNode(node.token, args)
                else:
                    tok = self.current_token
                    raise Exception(f"ParserError: Unexpected function call on non-identifier at L{tok.lineno}:C{tok.colno}")
                continue
                
            # Handle global @macro calls
            if self.current_token.type == TT_AT:
                self.advance()  # Consume '@'
                macro_name_token = self.consume(TT_IDENTIFIER)
                self.consume(TT_LPAREN)
                args = []
                if self.current_token.type != TT_RPAREN:
                    args = self.parse_argument_list()
                self.consume(TT_RPAREN)
                node = MacroInvokeNode(macro_name_token, args, macro_name_token.lineno)
                continue
                
            # Handle array access like list[index]
            if self.current_token.type == TT_LBRACKET:
                lbracket_token = self.consume(TT_LBRACKET)
                index_expr = self.parse_expression()
                self.consume(TT_RBRACKET)
                node = ArrayAccessNode(node, index_expr, lbracket_token)
                continue

            break
        
        return node

    def parse_primary(self):
        token = self.current_token
        if token.type == TT_INT_LITERAL:
            self.consume(TT_INT_LITERAL)
            return IntegerLiteralNode(token)
        elif token.type == TT_FLOAT_LITERAL:
            self.consume(TT_FLOAT_LITERAL)
            return FloatLiteralNode(token)
        elif token.type == TT_STRING_LITERAL:
            self.consume(TT_STRING_LITERAL)
            return StringLiteralNode(token)
        elif token.type == TT_THIS:
            self.consume(TT_THIS)
            return ThisNode(token)
        elif token.type == TT_IDENTIFIER: # Could be a variable name
            self.consume(TT_IDENTIFIER)
            return IdentifierNode(token) # For now, an identifier by itself is just an identifier node
        elif token.type == TT_TRUE:
            self.consume(TT_TRUE)
            return TrueLiteralNode(token)
        elif token.type == TT_FALSE:
            self.consume(TT_FALSE)
            return FalseLiteralNode(token)
        elif token.type == TT_LPAREN:
            lparen_token = self.consume(TT_LPAREN)
            
            # Check for C-style library call: ("m" use.sqrt(2.0))
            if self.current_token.type == TT_STRING_LITERAL:
                library_token = self.consume(TT_STRING_LITERAL)
                self.consume(TT_USE) # Expect 'use'
                self.consume(TT_DOT) # Expect '.'
                function_name_token = self.consume(TT_IDENTIFIER)
                self.consume(TT_LPAREN)
                args = []
                if self.current_token.type != TT_RPAREN:
                    args = self.parse_argument_list()
                self.consume(TT_RPAREN)
                
                # Now consume the final closing parenthesis of the ("lib" use ...) expression
                self.consume(TT_RPAREN)
                
                return CLibraryCallNode(library_token, function_name_token, args, lparen_token.lineno)

            # Otherwise, it's a grouped expression
            node = self.parse_expression()
            self.consume(TT_RPAREN)
            return node
            
        # Handle list literals
        if self.current_token.type == TT_LBRACE:
            return self.parse_list_literal()
        elif token.type == TT_AT:
            # Direct macro invocation: @name(args)
            self.consume(TT_AT)
            macro_name_token = self.consume(TT_IDENTIFIER)
            self.consume(TT_LPAREN)
            args = []
            if self.current_token.type != TT_RPAREN:
                args = self.parse_argument_list()
            self.consume(TT_RPAREN)
            return MacroInvokeNode(macro_name_token, args, macro_name_token.lineno)
        else:
            raise Exception(f"ParserError: Unexpected token {token.type} ('{token.value}') for expression at L{token.lineno}:C{token.colno}")

    def parse_list_literal(self):
        # list_literal ::= "{" (expression ("," expression)*)? "}"
        lbrace_token = self.consume(TT_LBRACE)
        elements = []
        if self.current_token.type != TT_RBRACE:
            elements.append(self.parse_expression())
            while self.current_token.type == TT_COMMA:
                self.consume(TT_COMMA)
                elements.append(self.parse_expression())
        self.consume(TT_RBRACE)
        return ListLiteralNode(elements, lbrace_token)

    def parse_type_specifier(self):
        # Parses type names like "int", "string", "MyClass", and "int[]", "MyClass[]"
        # Returns a string representing the full type.
        # Corrected logic:
        # type_specifier ::= base_type ( "[" "]" )?
        # base_type ::= TYPE | IDENTIFIER ("." IDENTIFIER)*
        
        type_str_parts = []
        # Keep track of the starting token for lineno/colno if needed for a synthetic token,
        # though the primary return is the type_str.
        # base_type_start_token = self.current_token 

        if self.current_token.type == TT_TYPE:
            type_token = self.consume(TT_TYPE)
            type_str_parts.append(type_token.value)
        elif self.current_token.type == TT_IDENTIFIER:
            type_str_parts.append(self.consume(TT_IDENTIFIER).value)
            while self.current_token.type == TT_DOT:
                self.consume(TT_DOT)
                type_str_parts.append(self.consume(TT_IDENTIFIER).value)
        else:
            tok = self.current_token
            raise Exception(f"ParserError: Expected TYPE or IDENTIFIER for base type, but got {tok.type} ('{tok.value}') at L{tok.lineno}:C{tok.colno}")
        
        type_str = ".".join(type_str_parts)

        # Allow for multiple array dimensions like int[][] or string[][][]
        while self.current_token.type == TT_LBRACKET:
            self.consume(TT_LBRACKET)
            self.consume(TT_RBRACKET)
            type_str += "[]"
        
        return type_str

    def parse_class_var_declaration(self):
        # class_var_declaration ::= ("public" | "private") "var" IDENTIFIER TYPE ("=" expression)?
        is_public = self.current_token.type == TT_PUBLIC
        self.consume(TT_PUBLIC if is_public else TT_PRIVATE)
        self.consume(TT_VAR)
        name_token = self.consume(TT_IDENTIFIER)
        
        # Parse type using the helper method
        type_string = self.parse_type_specifier()
        type_token = Token(TT_TYPE, type_string, name_token.lineno, name_token.colno)
        
        # Check for optional initialization
        expr_node = None
        if self.current_token.type == TT_ASSIGN:
            self.consume(TT_ASSIGN)
            expr_node = self.parse_expression()
        
        return ClassVarDeclNode(name_token, type_token, expr_node, is_public, name_token.lineno)

    def parse(self):
        if not self.tokens or self.tokens[0].type == TT_EOF and len(self.tokens) == 1:
            return ProgramNode([]) # Handle empty or only EOF token input
        return self.parse_program()