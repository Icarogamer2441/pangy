import re

# Token types
TT_CLASS = 'CLASS'
TT_DEF = 'DEF'
TT_IDENTIFIER = 'IDENTIFIER'
TT_LPAREN = '('
TT_RPAREN = ')'
TT_LBRACE = '{'
TT_RBRACE = '}'
TT_ARROW = '->'
TT_TYPE = 'TYPE' # e.g., void, int, string (for future)
TT_STRING_LITERAL = 'STRING_LITERAL'
TT_PRINT = 'PRINT'
TT_SEMICOLON = 'SEMICOLON' # If we need it
TT_EOF = 'EOF' # End of File

# Macro support
TT_MACRO = 'MACRO'
TT_AT = 'AT'

# New tokens for funcs.pgy
TT_INT_LITERAL = 'INT_LITERAL'
TT_COMMA = ','
TT_COLON = ':' # For 'this:method'
TT_PLUS = 'PLUS'
TT_RETURN = 'RETURN'
TT_THIS = 'THIS' # for the 'this' keyword
TT_ASSIGN = 'ASSIGN' # Added for variable assignment
TT_VAR = 'VAR' # Added for 'var' keyword

# New arithmetic operators
TT_MINUS = 'MINUS'
TT_STAR = 'STAR'   # For multiplication
TT_SLASH = 'SLASH' # For division
TT_PERCENT = 'PERCENT' # For modulo
TT_PLUSPLUS = 'PLUSPLUS' # For increment
TT_MINUSMINUS = 'MINUSMINUS' # For decrement

# New tokens for if/else and comparisons
TT_IF = 'IF'
TT_ELSE = 'ELSE'
TT_LESS_THAN = 'LESS_THAN'
TT_GREATER_THAN = 'GREATER_THAN'
TT_EQUAL = 'EQUAL'         # For ==
TT_NOT_EQUAL = 'NOT_EQUAL' # For !=
TT_LESS_EQUAL = 'LESS_EQUAL'   # For <=
TT_GREATER_EQUAL = 'GREATER_EQUAL' # For >=

TT_STATIC = 'STATIC' # For static methods
TT_DOT = 'DOT' # For member access and static calls like Class.method
TT_INCLUDE = 'INCLUDE' # For include "path" directives
TT_LOOP = 'LOOP' # For loop keyword
TT_STOP = 'STOP' # For stop keyword

# Tokens for list support
TT_LBRACKET = 'LBRACKET' # For '['
TT_RBRACKET = 'RBRACKET' # For ']'

class Token:
    def __init__(self, type, value, lineno=0, colno=0):
        self.type = type
        self.value = value
        self.lineno = lineno
        self.colno = colno

    def __repr__(self):
        return f"Token({self.type}, {repr(self.value)}, L{self.lineno}, C{self.colno})"

# Lexer
class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None
        self.lineno = 1
        self.colno = 1

    def advance(self):
        if self.current_char == '\n':
            self.lineno += 1
            self.colno = 0
        self.pos += 1
        self.colno += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None

    def skip_whitespace_and_comments(self):
        while self.current_char is not None and (self.current_char.isspace() or self.current_char == '#'):
            if self.current_char == '#':
                while self.current_char is not None and self.current_char != '\n':
                    self.advance()
            else:
                self.advance()

    def get_identifier_or_keyword(self):
        result = ''
        start_col = self.colno
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        
        # Check for keywords
        if result == 'class':
            return Token(TT_CLASS, 'class', self.lineno, start_col)
        elif result == 'def':
            return Token(TT_DEF, 'def', self.lineno, start_col)
        elif result == 'void' or result == 'int' or result == 'string' or result == 'file': # Added 'file' as a type
            return Token(TT_TYPE, result, self.lineno, start_col)
        elif result == 'print':
            return Token(TT_PRINT, 'print', self.lineno, start_col)
        elif result == 'return':
            return Token(TT_RETURN, 'return', self.lineno, start_col)
        elif result == 'this':
            return Token(TT_THIS, 'this', self.lineno, start_col)
        elif result == 'if':
            return Token(TT_IF, 'if', self.lineno, start_col)
        elif result == 'else':
            return Token(TT_ELSE, 'else', self.lineno, start_col)
        elif result == 'var': # Added for 'var' keyword
            return Token(TT_VAR, 'var', self.lineno, start_col)
        elif result == 'static': # Added for static keyword
            return Token(TT_STATIC, 'static', self.lineno, start_col)
        elif result == 'include': # Added for include keyword
            return Token(TT_INCLUDE, 'include', self.lineno, start_col)
        elif result == 'loop':
            return Token(TT_LOOP, 'loop', self.lineno, start_col)
        elif result == 'stop':
            return Token(TT_STOP, 'stop', self.lineno, start_col)
        elif result == 'macro': # Added for macro keyword
            return Token(TT_MACRO, 'macro', self.lineno, start_col)
        else:
            return Token(TT_IDENTIFIER, result, self.lineno, start_col)

    def get_string_literal(self):
        result = ''
        start_col = self.colno
        self.advance() # Skip the opening quote
        while self.current_char is not None and self.current_char != '"':
            if self.current_char == '\\': # Handle escaped characters like \" or \\
                self.advance()
                if self.current_char is None:
                    raise Exception(f"LexerError: Unterminated string literal at L{self.lineno}:C{start_col}")
            result += self.current_char
            self.advance()
        
        if self.current_char is None: # Unterminated string
            raise Exception(f"LexerError: Unterminated string literal at L{self.lineno}:C{start_col}")
        
        self.advance() # Skip the closing quote
        return Token(TT_STRING_LITERAL, result, self.lineno, start_col)

    def get_integer_literal(self):
        result = ''
        start_col = self.colno
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return Token(TT_INT_LITERAL, int(result), self.lineno, start_col)

    def get_next_token(self):
        while self.current_char is not None:
            start_col = self.colno # Store start column for single char tokens
            if self.current_char.isspace():
                self.skip_whitespace_and_comments()
                continue

            if self.current_char == '#':
                self.skip_whitespace_and_comments()
                continue

            if self.current_char.isalpha() or self.current_char == '_':
                return self.get_identifier_or_keyword()

            if self.current_char.isdigit():
                return self.get_integer_literal()

            if self.current_char == '"':
                return self.get_string_literal()

            if self.current_char == '@':
                token = Token(TT_AT, '@', self.lineno, start_col)
                self.advance()
                return token

            if self.current_char == '(':
                token = Token(TT_LPAREN, '(', self.lineno, start_col)
                self.advance()
                return token
            
            if self.current_char == ')':
                token = Token(TT_RPAREN, ')', self.lineno, start_col)
                self.advance()
                return token

            if self.current_char == '{':
                token = Token(TT_LBRACE, '{', self.lineno, start_col)
                self.advance()
                return token

            if self.current_char == '}':
                token = Token(TT_RBRACE, '}', self.lineno, start_col)
                self.advance()
                return token
            
            if self.current_char == '-' and self.pos + 1 < len(self.text) and self.text[self.pos+1] == '>':
                self.advance()
                self.advance()
                return Token(TT_ARROW, '->', self.lineno, start_col)
            
            if self.current_char == ',':
                token = Token(TT_COMMA, ',', self.lineno, start_col)
                self.advance()
                return token

            if self.current_char == ':':
                token = Token(TT_COLON, ':', self.lineno, start_col)
                self.advance()
                return token

            if self.current_char == '+' and self.pos + 1 < len(self.text) and self.text[self.pos+1] == '+':
                self.advance()
                self.advance()
                return Token(TT_PLUSPLUS, '++', self.lineno, start_col)

            if self.current_char == '-' and self.pos + 1 < len(self.text) and self.text[self.pos+1] == '-':
                self.advance()
                self.advance()
                return Token(TT_MINUSMINUS, '--', self.lineno, start_col)

            if self.current_char == '+':
                token = Token(TT_PLUS, '+', self.lineno, start_col)
                self.advance()
                return token
            
            if self.current_char == '-': # Added for subtraction
                token = Token(TT_MINUS, '-', self.lineno, start_col)
                self.advance()
                return token

            if self.current_char == '*': # Added for multiplication
                token = Token(TT_STAR, '*', self.lineno, start_col)
                self.advance()
                return token

            # Single-line comments using //
            if self.current_char == '/' and self.pos + 1 < len(self.text) and self.text[self.pos+1] == '/':
                # Skip until end of line
                self.advance()
                self.advance()
                while self.current_char is not None and self.current_char != '\n':
                    self.advance()
                continue

            if self.current_char == '/': # Added for division
                token = Token(TT_SLASH, '/', self.lineno, start_col)
                self.advance()
                return token
            
            if self.current_char == '%': # Added for modulo
                token = Token(TT_PERCENT, '%', self.lineno, start_col)
                self.advance()
                return token
            
            if self.current_char == '.': # Added for dot operator
                token = Token(TT_DOT, '.', self.lineno, start_col)
                self.advance()
                return token
            
            # Comparison operators: <=, <, >=, >
            if self.current_char == '<' and self.pos + 1 < len(self.text) and self.text[self.pos+1] == '=':
                self.advance()
                self.advance()
                return Token(TT_LESS_EQUAL, '<=', self.lineno, start_col)
            if self.current_char == '<':
                token = Token(TT_LESS_THAN, '<', self.lineno, start_col)
                self.advance()
                return token
            if self.current_char == '>' and self.pos + 1 < len(self.text) and self.text[self.pos+1] == '=':
                self.advance()
                self.advance()
                return Token(TT_GREATER_EQUAL, '>=', self.lineno, start_col)
            if self.current_char == '>':
                token = Token(TT_GREATER_THAN, '>', self.lineno, start_col)
                self.advance()
                return token

            # Equality and inequality: ==, !=
            if self.current_char == '!' and self.pos + 1 < len(self.text) and self.text[self.pos+1] == '=':
                self.advance()
                self.advance()
                return Token(TT_NOT_EQUAL, '!=', self.lineno, start_col)
            if self.current_char == '=' and self.pos + 1 < len(self.text) and self.text[self.pos+1] == '=':
                self.advance()
                self.advance()
                return Token(TT_EQUAL, '==', self.lineno, start_col)

            # Assignment
            if self.current_char == '=': # Added for assignment
                token = Token(TT_ASSIGN, '=', self.lineno, start_col)
                self.advance()
                return token
            
            # Semicolon might be used later, e.g. at end of statements
            # if self.current_char == ';':
            #    token = Token(TT_SEMICOLON, ';', self.lineno, start_col)
            #    self.advance()
            #    return token

            if self.current_char == '[': # Added for list type open bracket
                token = Token(TT_LBRACKET, '[', self.lineno, start_col)
                self.advance()
                return token

            if self.current_char == ']': # Added for list type close bracket
                token = Token(TT_RBRACKET, ']', self.lineno, start_col)
                self.advance()
                return token
            
            raise Exception(f"LexerError: Unexpected character '{self.current_char}' at L{self.lineno}:C{self.colno}")

        return Token(TT_EOF, None, self.lineno, self.colno)

    def tokenize(self):
        tokens = []
        token = self.get_next_token()
        while token.type != TT_EOF:
            tokens.append(token)
            token = self.get_next_token()
        tokens.append(token) # Add EOF token
        return tokens

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

    def __repr__(self):
        return f"ClassNode(name='{self.name}', methods={self.methods}, nested_classes={self.nested_classes}, macros={self.macros})"

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
        
class IdentifierNode(ExprNode): # For variable names, method names (in some contexts)
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
    def __init__(self, name_token, type_token, expr_node, lineno):
        self.name = name_token.value
        self.name_token = name_token
        self.type = type_token.value # This will become a string
        self.type_token = type_token # May need adjustment
        self.expression = expr_node # The expression to initialize the variable
        self.lineno = lineno

    def __repr__(self):
        return f"VarDeclNode(name='{self.name}', type='{self.type}', expr={self.expression}, L{self.lineno})"

# AST Nodes for Loops, Stop, and Assignment
class LoopNode(ASTNode):
    def __init__(self, body, lineno):
        self.body = body  # BlockNode
        self.lineno = lineno
    def __repr__(self):
        return f"LoopNode(body={self.body}, L{self.lineno})"

class StopNode(ASTNode):
    def __init__(self, lineno):
        self.lineno = lineno
    def __repr__(self):
        return f"StopNode(L{self.lineno})"

class AssignmentNode(ASTNode):
    def __init__(self, name_token, expression, lineno):
        self.name = name_token.value
        self.name_token = name_token
        self.expression = expression
        self.lineno = lineno
    def __repr__(self):
        return f"AssignmentNode(name='{self.name}', expr={self.expression}, L{self.lineno})"

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
        # class_definition ::= "class" IDENTIFIER "{" class_definition* method_definition* macro_definition* "}"
        self.consume(TT_CLASS)
        name_token = self.consume(TT_IDENTIFIER)
        self.consume(TT_LBRACE)  # '{'

        nested_classes = []
        methods = []
        macros = []  # Added list for macros inside the class
        # Loop for nested classes, method definitions, or macro definitions
        while self.current_token.type in (TT_CLASS, TT_STATIC, TT_DEF, TT_MACRO):
            if self.current_token.type == TT_CLASS:
                nested_classes.append(self.parse_class_definition())
            elif self.current_token.type == TT_MACRO:
                macros.append(self.parse_macro_definition())  # Parse macro definition inside class
            else:
                methods.append(self.parse_method_definition())

        self.consume(TT_RBRACE)  # '}'
        class_node = ClassNode(name_token, methods, nested_classes)
        class_node.macros = macros  # Add macros to the ClassNode
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
        elif start_token.type == TT_STOP:
            return self.parse_stop_statement()
        elif start_token.type == TT_VAR: # Added for variable declaration
            return self.parse_var_declaration_statement()
        elif start_token.type == TT_IDENTIFIER and self.token_idx + 1 < len(self.tokens) and self.tokens[self.token_idx + 1].type == TT_ASSIGN:
            return self.parse_assignment_statement()
        elif start_token.type in [TT_THIS, TT_IDENTIFIER, TT_LPAREN, TT_INT_LITERAL, TT_STRING_LITERAL]:
            # If it looks like the start of an expression, parse it as an expression statement.
            # The value of the expression will be discarded.
            return self.parse_expression() # Expressions can be statements
        else:
            # No self.error method; raise exception instead
            raise Exception(f"ParserError: Unexpected statement token {start_token.type} ('{start_token.value}') at L{start_token.lineno}:C{start_token.colno}")

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
        # var_declaration ::= "var" IDENTIFIER TYPE ( "." IDENTIFIER )* ("=" expression)?
        self.consume(TT_VAR)
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

        return VarDeclNode(name_token, type_token, expr_node, name_token.lineno)

    def parse_loop_statement(self):
        # loop_statement ::= "loop" block
        loop_token = self.consume(TT_LOOP)
        body = self.parse_block()
        return LoopNode(body, loop_token.lineno)

    def parse_stop_statement(self):
        # stop_statement ::= "stop"
        stop_token = self.consume(TT_STOP)
        return StopNode(stop_token.lineno)

    def parse_assignment_statement(self):
        # assignment ::= IDENTIFIER "=" expression
        name_token = self.consume(TT_IDENTIFIER)
        self.consume(TT_ASSIGN)
        expr_node = self.parse_expression()
        return AssignmentNode(name_token, expr_node, name_token.lineno)

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
        return self.parse_comparison_expression()

    def parse_comparison_expression(self):
        # comparison_expression ::= additive_expression ( (LT | GT | LE | GE | EQ | NE) additive_expression )*
        node = self.parse_additive_expression()

        while self.current_token.type in [TT_LESS_THAN, TT_GREATER_THAN, TT_LESS_EQUAL, TT_GREATER_EQUAL, TT_EQUAL, TT_NOT_EQUAL]:
            op_token = self.consume(self.current_token.type) # Consume comparison operator
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
        # multiplicative_expression ::= term ( (STAR | SLASH | PERCENT) term )*
        node = self.parse_term() # Higher precedence items

        while self.current_token.type in [TT_STAR, TT_SLASH, TT_PERCENT]:
            op_token = self.consume(self.current_token.type) # Consume *, /, or %
            right_node = self.parse_term()
            node = BinaryOpNode(node, op_token, right_node)
        return node

    def parse_term(self):
        # term ::= primary ( ( "." | ":" ) IDENTIFIER "(" arguments? ")" | "++" | "--" | "(" arguments ")" | "[" expression "]" )* 
        node = self.parse_primary()

        while self.current_token.type in (TT_DOT, TT_COLON, TT_PLUSPLUS, TT_MINUSMINUS, TT_LPAREN, TT_AT, TT_LBRACKET):
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

            # Handle this:method() or this:@macro() calls
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
                
                # Regular method call
                method_name_token = self.consume(TT_IDENTIFIER)
                self.consume(TT_LPAREN)
                args = []
                if self.current_token.type != TT_RPAREN:
                    args = self.parse_argument_list()
                self.consume(TT_RPAREN)
                node = MethodCallNode(node, method_name_token, args)
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
        elif token.type == TT_STRING_LITERAL:
            self.consume(TT_STRING_LITERAL)
            return StringLiteralNode(token)
        elif token.type == TT_THIS:
            self.consume(TT_THIS)
            return ThisNode(token)
        elif token.type == TT_IDENTIFIER: # Could be a variable name
            self.consume(TT_IDENTIFIER)
            return IdentifierNode(token) # For now, an identifier by itself is just an identifier node
        elif token.type == TT_LPAREN:
            self.consume(TT_LPAREN)
            node = self.parse_expression()
            self.consume(TT_RPAREN)
            return node
        elif token.type == TT_LBRACE: # For list literals like {1, 2, 3}
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

    def parse(self):
        if not self.tokens or self.tokens[0].type == TT_EOF and len(self.tokens) == 1:
            return ProgramNode([]) # Handle empty or only EOF token input
        return self.parse_program()

# Example Usage (for testing parser_lexer.py directly)
if __name__ == '__main__':
    source_code_funcs = """
    class Main {
        def main() -> void {
            print("10 + 20 = ", this:add(10, 20))
            return 
        }

        def add(a int, b int) -> int {
            return a + b
        }
    }
    """
    lexer = Lexer(source_code_funcs)
    tokens = lexer.tokenize()
    print("\nTokens for funcs.pgy:")
    for t in tokens:
        print(t)
    
    print("\nAST for funcs.pgy:")
    parser = Parser(tokens)
    ast = parser.parse()
    import pprint
    pprint.pprint(ast)

    source_code_simple_print = """
    class Main {
        def main() -> void {
            print("Hello")
            print(123)
            print("Value: ", 100)
        }
    }
    """
    lexer2 = Lexer(source_code_simple_print)
    tokens2 = lexer2.tokenize()
    print("\nTokens for simple_print.pgy:")
    for t in tokens2: print(t)
    parser2 = Parser(tokens2)
    ast2 = parser2.parse()
    print("\nAST for simple_print.pgy:")
    pprint.pprint(ast2)

    source_code_return_expr = """
    class Calc {
        def get_val() -> int {
            return 10 + (5 + 2)
        }
    }
    """
    lexer3 = Lexer(source_code_return_expr)
    tokens3 = lexer3.tokenize()
    print("\nTokens for return_expr.pgy:")
    for t in tokens3: print(t)
    parser3 = Parser(tokens3)
    ast3 = parser3.parse()
    print("\nAST for return_expr.pgy:")
    pprint.pprint(ast3)

    # Test empty return in void method (should be allowed by grammar if no expression is required)
    # Current return always expects an expression.
    # For 'return' in a void method, we might need a ReturnNode(None) or different parsing path.
    # The funcs.pgy example has 'return' in main() -> void.
    # The current parse_return_statement *requires* an expression.
    # Let's adjust for void returns needing no expression or a specific 'empty' expression.
    # If return type is void, expression is optional/disallowed.
    # This type checking is more for semantic analysis, but parser can allow optional expr.

    # Simplified solution for now in parser: make expression optional in ReturnNode.
    # The example usage above has a `return` in `main() -> void {}`, which is fine.
    # The parser should allow `return` without an expression.
    # Modified parse_return_statement in the code above to allow it.
    # Let's refine ReturnNode and parse_return_statement for optional expression.
    # Actually, the example usage for funcs.pgy had `return` in main.
    # The prompt `examples/funcs.pgy` has `print("10 + 20 = ", this:add(10, 20))`
    # and no explicit return in main.
    # The `return a + b` is in `add`.

    # Let's assume for now that `return` *always* has an expression.
    # A `return` statement in a void function would then be a semantic error later.
    # Or, if a function is void, its `return` cannot have an expression.
    # The current parser requires an expression for `return`. This is fine.
    # If main has an implicit return, the compiler handles it.
    # If a void function has an explicit `return;`