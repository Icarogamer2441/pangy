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
- Input functions and type conversion utilities
- File operations (open, close, read, write)

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
        var name string = input("Enter your name: ")
        var age int = to_int(input("Enter your age: "))
        
        print("Hello, ", name, "! You are ", age, " years old.")
        print("Age as string: ", to_string(age))
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

## Input and Type Conversion

Pangy provides built-in functions for collecting user input and converting between data types.

### Input Function

The `input()` function displays a prompt and waits for the user to enter text:

```
var name string = input("Enter your name: ")
```

### Type Conversion

Pangy provides functions to convert between different data types:

- `to_int()`: Converts a string to an integer
- `to_string()`: Converts an integer to a string

Example usage:

```
var age_str string = input("Enter your age: ")
var age int = to_int(age_str)
print("Age doubled: ", age * 2)

var num int = 42
var str string = to_string(num)
print("The number as string: ", str)
```

## File Operations

Pangy supports basic file operations with the following functions:

### File Type

The `file` data type represents a file pointer:

```
var file_ptr file = open("example.txt", "w")
```

### File Functions

- `open(filename, mode)`: Opens a file and returns a file pointer
  - Modes: "r" (read), "w" (write), "a" (append)
- `close(file_ptr)`: Closes a file
- `write(file_ptr, content)`: Writes content to a file
- `read(file_ptr)`: Reads content from a file

### Example

```
class Main {
    def main() -> void {
        var file_name string = input("Enter the name of the file to create: ")
        var file_content string = input("Enter the content of the file: ")
        
        var file_path string = file_name + ".txt"
        var file_pointer file = open(file_path, "w")
        
        write(file_pointer, file_content)
        print("File created successfully!")
        close(file_pointer)
        
        var file_pointer_read file = open(file_path, "r")
        var file_content_read string = read(file_pointer_read)
        print("File content: ", file_content_read)
        close(file_pointer_read)
    }
}
```

## Example

```
class Main {
    def main() -> void {
        var name string = input("Enter your name: ")
        var age int = to_int(input("Enter your age: "))
        
        print("Hello, ", name, "! You are ", age, " years old.")
        print("Age as string: ", to_string(age))
    }
}

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
