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

# C Library support
TT_USE = 'USE'

# Public/Private keywords
TT_PUBLIC = 'PUBLIC'
TT_PRIVATE = 'PRIVATE'

# New tokens for funcs.pgy
TT_INT_LITERAL = 'INT_LITERAL'
TT_COMMA = ','
TT_COLON = ':' # For 'this:method'
TT_DOUBLE_COLON = '::' # For accessing class variables from outside
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
TT_WHILE = 'WHILE' # For while loops
TT_STOP = 'STOP' # For stop keyword

# Tokens for list support
TT_LBRACKET = 'LBRACKET' # For '['
TT_RBRACKET = 'RBRACKET' # For ']'

# Tokens for bitwise operations
TT_AMPERSAND = 'AMPERSAND'  # For '&' (bitwise AND)
TT_PIPE = 'PIPE'            # For '|' (bitwise OR)
TT_CARET = 'CARET'          # For '^' (bitwise XOR)
TT_TILDE = 'TILDE'          # For '~' (bitwise NOT)

# Tokens for shift operations
TT_LSHIFT = 'LSHIFT'        # For '<<' (left shift)
TT_RSHIFT = 'RSHIFT'        # For '>>' (right shift)
TT_URSHIFT = 'URSHIFT'      # For '>>>' (unsigned right shift)

# Tokens for logical operations
TT_LOGICAL_AND = 'LOGICAL_AND' # For '&&'
TT_LOGICAL_OR = 'LOGICAL_OR'   # For '||'

# Boolean Literals
TT_TRUE = 'TRUE'
TT_FALSE = 'FALSE'

TT_FLOAT_LITERAL = 'FLOAT_LITERAL'  # Added for float literals

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
        elif result == 'while':
            return Token(TT_WHILE, 'while', self.lineno, start_col)
        elif result == 'stop':
            return Token(TT_STOP, 'stop', self.lineno, start_col)
        elif result == 'macro': # Added for macro keyword
            return Token(TT_MACRO, 'macro', self.lineno, start_col)
        elif result == 'public': # Added for public keyword
            return Token(TT_PUBLIC, 'public', self.lineno, start_col)
        elif result == 'private': # Added for private keyword
            return Token(TT_PRIVATE, 'private', self.lineno, start_col)
        elif result == 'true':
            return Token(TT_TRUE, True, self.lineno, start_col)
        elif result == 'false':
            return Token(TT_FALSE, False, self.lineno, start_col)
        elif result == 'use': # Added for C library usage
            return Token(TT_USE, 'use', self.lineno, start_col)
        else:
            return Token(TT_IDENTIFIER, result, self.lineno, start_col)

    def get_string_literal(self):
        result = ''
        start_col = self.colno
        self.advance() # Skip the opening quote
        while self.current_char is not None and self.current_char != '"':
            if self.current_char == '\\': # Handle escaped characters
                self.advance()
                if self.current_char is None:
                    raise Exception(f"LexerError: Unterminated string literal at L{self.lineno}:C{start_col}")
                
                # Process escape sequence
                if self.current_char == 'n':
                    result += '\n'  # Newline
                elif self.current_char == 't':
                    result += '\t'  # Tab
                elif self.current_char == 'r':
                    result += '\r'  # Carriage return
                elif self.current_char == '"':
                    result += '"'   # Double quote
                elif self.current_char == '\'':
                    result += '\''  # Single quote
                elif self.current_char == '\\':
                    result += '\\'  # Backslash
                elif self.current_char == '0':
                    result += '\0'  # Null character
                elif self.current_char == 'b':
                    result += '\b'  # Backspace
                elif self.current_char == 'f':
                    result += '\f'  # Form feed
                elif self.current_char == 'v':
                    result += '\v'  # Vertical tab
                else:
                    # For any other character, just include it literally
                    # This allows for things like \x which would just be 'x'
                    result += self.current_char
            else:
                result += self.current_char
            self.advance()
        
        if self.current_char is None: # Unterminated string
            raise Exception(f"LexerError: Unterminated string literal at L{self.lineno}:C{start_col}")
        
        self.advance() # Skip the closing quote
        return Token(TT_STRING_LITERAL, result, self.lineno, start_col)

    def get_integer_literal(self):
        result = ''
        start_col = self.colno
        # Read integer part
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        # Check for fractional part for floats
        if self.current_char == '.':
            result += self.current_char
            self.advance()
            # Require at least one digit after decimal point
            if self.current_char is None or not self.current_char.isdigit():
                raise Exception(f"LexerError: Invalid float literal at L{self.lineno}:C{start_col}")
            while self.current_char is not None and self.current_char.isdigit():
                result += self.current_char
                self.advance()
            # Return float literal token
            return Token(TT_FLOAT_LITERAL, float(result), self.lineno, start_col)
        # Otherwise return integer literal token
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

            if self.current_char == ':' and self.pos + 1 < len(self.text) and self.text[self.pos+1] == ':':
                self.advance()
                self.advance()
                return Token(TT_DOUBLE_COLON, '::', self.lineno, start_col)

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
            
            # Bitwise AND operator '&'
            if self.current_char == '&':
                self.advance()
                if self.current_char == '&':
                    self.advance()
                    return Token(TT_LOGICAL_AND, '&&', self.lineno, start_col)
                else:
                    return Token(TT_AMPERSAND, '&', self.lineno, start_col)
                
            # Bitwise OR operator '|'
            if self.current_char == '|':
                self.advance()
                if self.current_char == '|':
                    self.advance()
                    return Token(TT_LOGICAL_OR, '||', self.lineno, start_col)
                else:
                    return Token(TT_PIPE, '|', self.lineno, start_col)
                
            # Bitwise XOR operator '^'
            if self.current_char == '^':
                token = Token(TT_CARET, '^', self.lineno, start_col)
                self.advance()
                return token
                
            # Bitwise NOT operator '~'
            if self.current_char == '~':
                token = Token(TT_TILDE, '~', self.lineno, start_col)
                self.advance()
                return token
            
            # Comparison operators: <=, <, >=, >
            if self.current_char == '<' and self.pos + 1 < len(self.text) and self.text[self.pos+1] == '=':
                self.advance()
                self.advance()
                return Token(TT_LESS_EQUAL, '<=', self.lineno, start_col)
            # Left shift operator '<<'
            if self.current_char == '<' and self.pos + 1 < len(self.text) and self.text[self.pos+1] == '<':
                self.advance()
                self.advance()
                return Token(TT_LSHIFT, '<<', self.lineno, start_col)
            if self.current_char == '<':
                token = Token(TT_LESS_THAN, '<', self.lineno, start_col)
                self.advance()
                return token
            # Unsigned right shift operator '>>>'
            if self.current_char == '>' and self.pos + 2 < len(self.text) and self.text[self.pos+1] == '>' and self.text[self.pos+2] == '>':
                self.advance()
                self.advance()
                self.advance()
                return Token(TT_URSHIFT, '>>>', self.lineno, start_col)
            # Right shift operator '>>'
            if self.current_char == '>' and self.pos + 1 < len(self.text) and self.text[self.pos+1] == '>':
                self.advance()
                self.advance()
                return Token(TT_RSHIFT, '>>', self.lineno, start_col)
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