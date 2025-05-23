# Pangy Programming Language

Pangy is a custom programming language with its own compiler, designed to compile to x86-64 assembly.

## Features

- Class-based object-oriented programming
- Methods with typed parameters
- Control structures (if/else, loops)
- Variables with type declarations
- Support for basic data types (int, string)
- Macro system with object context support
- Print statements with variable substitution
- Module system with selective imports

## Include System

Pangy supports importing code from other files using the `include` directive. You can import entire files, specific classes, inner classes, or even specific macros.

### Import syntax

```
// Import an entire file
include filename

// Import a specific class
include filename.ClassName

// Import an inner class
include filename.ClassName.InnerClass

// Import a specific macro
include filename.@macroname
```

### Example

File: math.pgy
```
class Math {
    def add(a int, b int) -> int {
        return a + b
    }
    
    def sub(a int, b int) -> int {
        return a - b
    }
    
    def mul(a int, b int) -> int {
        return a * b
    }
    
    def div(a int, b int) -> int {
        return a / b
    }
    
    def mod(a int, b int) -> int {
        return a % b
    }
}
```

File: main.pgy
```
include math.Math // you can also import macros with 'include filename.@macroname'
// and inner classes with 'include filename.ClassName.InnerClass'

class Main {
    def main() -> void {
        var math Math = Math.new()
        print("10 + 20 = ", math.add(10, 20))
        print("10 - 20 = ", math.sub(10, 20))
        print("10 * 20 = ", math.mul(10, 20))
        print("10 / 20 = ", math.div(10, 20))
        print("10 % 20 = ", math.mod(10, 20))
    }
}
```

## Macro System

Pangy supports macros at both global and class level. Macros allow for compile-time code generation.

### Macro Definition

```
// Global macro
macro add(a, b) {
    a + b
}

class Math {
    // Class macro
    macro mul(a, b) {
        a * b
    }
}
```

### Macro Invocation

Macros can be invoked using the `@` symbol:

```
// Global macro invocation
var x = @add(10, 20)

// Class macro invocation with 'this'
var y = this:@add(5, 10)

// Class macro invocation with an object
var math = Math.new()
var z = math.@mul(2, 3)
```

## Example

```
class Main {
    def main() -> void {
        var math Math = Math.new()
        print("1: 10 + 20 = ", this:@add(10, 20))
        var a int = this:@add(10, 20)
        print("2: 10 + 20 = ", a)

        math.@do_add(10, 20)
    }

    macro add(a, b) {
        a + b
    }
}

class Math {
    macro add(a, b) {
        a + b
    }

    def do_add(a, b) -> void {
        print("3: 10 + 20 = ", this:@add(a, b))
    }
}
```

## Compilation

To compile a Pangy program:

```bash
python -m pangy your_program.pgy
```

Options:
- `-o OUTPUT` - Specify output file name
- `-S` - Output assembly instead of an executable
- `--ast` - Print the abstract syntax tree
- `--tokens` - Print the lexical tokens
