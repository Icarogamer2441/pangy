# Pangy Compiler

Pangy is a simple command-line compiler for the Pangy language, featuring:

- Classes with nested definitions
- Static and instance methods
- Primitive types (`int`, `string`, `void`)
- Class types as input parameters and return types (newly added feature)
- `var` declarations, expressions, control flow (`if`, `loop`, `stop`)
- Arithmetic operations (`+`, `-`, `*`, `/`, `%`) and comparisons

## Install

```bash
pip install -e .
```

## Usage

```bash
pangy <input_file>.pgy [options]
```

Positional arguments:

- input_file           Pangy source file to compile (.pgy)

Optional arguments:

- -o, --output OUTPUT  Output file name (default: a.out executable, or .s file if generating assembly)
- -S, --assembly       Output assembly code (.s file) instead of an executable
- --ast                Print the combined Abstract Syntax Tree (AST) and exit
- --tokens             Print tokens from the main input file and exit

Examples:

```bash
# Compile to default executable (a.out)
pangy examples/hello.pgy

# Specify executable name
pangy examples/hello.pgy -o my_program

# Generate assembly code instead of executable
pangy examples/hello.pgy -S
pangy examples/hello.pgy -S -o hello.s

# Print AST
pangy examples/hello.pgy --ast

# Print tokens
pangy examples/hello.pgy --tokens
```

## Examples

see [examples](./examples/) for examples.

## Recent Changes

- **Support for class types as input parameters and return types**: Methods can now accept and return user-defined class types, including nested class qualifiers.
- **Support for built-in `exit(code)` function**: Allow terminating the program with a custom exit code, mapping to the C `exit` function.

## Project Structure

```
pangy/               # Compiler source code
  parser_lexer.py     # Lexer and parser implementation
  compiler.py         # Code generation logic
  cli.py              # Command-line interface
examples/             # Example Pangy programs
```

## License

[MIT](./LICENSE)
