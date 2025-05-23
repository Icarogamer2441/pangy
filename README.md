# Pangy Compiler

Pangy is a simple command-line compiler for the Pangy language, featuring:

- Classes with nested definitions
- Static and instance methods
- Primitive types (`int`, `string`, `void`)
- Class types as input parameters and return types (newly added feature)
- `var` declarations, expressions, control flow (`if`, `loop`, `stop`)
- Arithmetic operations (`+`, `-`, `*`, `/`, `%`) and comparisons

## Usage

To compile and run a Pangy program:

```bash
python3 -m pangy.cli <source_file>.pgy
```

Example:

```bash
python3 -m pangy.cli examples/classinparams.pgy
```

## Examples

- `examples/classinparams.pgy`: demonstrates passing class instances as parameters to methods.

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

MIT
