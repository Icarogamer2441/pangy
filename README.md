# Pangy Programming Language

Pangy is a custom programming language with its own compiler, designed to compile to x86-64 assembly.

## Features

- Class-based object-oriented programming
- Methods with typed parameters
- Control structures (if/else, loops)
- Variables with type declarations
- Support for basic data types (int, string)
- String operations (concatenation, comparison)
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

## Dynamic Lists

Pangy supports dynamic lists (arrays) of integers or strings.

### Declaration and Initialization

Lists are declared with a base type followed by `[]`. They can be initialized with a literal syntax using curly braces `{}`.

```pangy
class Main {
    def main() -> void {
        var myIntList int[] = {10, 20, 30}       // List of integers
        var myStringList string[] = {"hello", "pangy", "world"} // List of strings
        var emptyList int[] = {}                // An empty list of integers

        // You can also declare without immediate initialization,
        // but it will be a null pointer until assigned or appended to.
        var anotherList int[]
        // anotherList = {} // Initialize later
        // append(anotherList, 100) // First append will allocate it
    }
}
```

### Accessing Elements

List elements are accessed using zero-based indexing with square brackets `[]`.

```pangy
var myList int[] = {5, 10, 15}
var firstElement int = myList[0]  // firstElement will be 5
print(myList[1])                 // Prints 10

myList[2] = 20                   // Modifies the element at index 2
print(myList[2])                 // Prints 20
```
Accessing an element with an index out of bounds (less than 0 or greater than or equal to the list length) or on a null list pointer will result in a runtime error and program termination.

### Built-in List Functions

- `length(list_ptr) -> int`
  Returns the current number of elements in the list.
  ```pangy
  var numbers int[] = {1, 2, 3, 4}
  var len int = length(numbers) // len will be 4
  print("List length: ", len)
  ```

- `append(list_ptr, value) -> list_ptr`
  Appends a `value` to the end of the list. If the list needs to grow, it will be reallocated. 
  **Important:** `append` returns the (potentially new) pointer to the list. You should assign the result back to your list variable if the list might be reallocated (which can happen if its capacity is exceeded).
  ```pangy
  var items int[] = {1}
  items = append(items, 2) // items is now {1, 2}
  items = append(items, 3) // items is now {1, 2, 3}
  print(items[0], items[1], items[2]) // Prints 1 2 3

  var strList string[] = {}
  strList = append(strList, "a")
  strList = append(strList, "b")
  print(length(strList)) // Prints 2
  ```

### Example Usage

```pangy
class ListDemo {
    def main() -> void {
        var scores int[] = {}

        scores = append(scores, 100)
        scores = append(scores, 90)
        scores = append(scores, 95)

        print("Number of scores: ", length(scores)) // Prints 3

        var i int = 0
        loop {
            if i >= length(scores) {
                break
            }
            print("Score ", i, ": ", scores[i])
            i++
        }
        // Output:
        // Score 0 : 100
        // Score 1 : 90
        // Score 2 : 95

        scores[1] = 92
        print("Updated score 1: ", scores[1]) // Prints 92
    }
}
```

## Example

```

## Bitwise Operations and Shift Operations

Pangy supports bitwise operations and shift operations on integer values.

### Bitwise Operators

| Operator | Name        | Description                             |
|----------|-------------|-----------------------------------------|
| `&`      | Bitwise AND | Returns 1 for each bit position where both operands have 1 |
| `\|`     | Bitwise OR  | Returns 1 for each bit position where at least one operand has 1 |
| `^`      | Bitwise XOR | Returns 1 for each bit position where exactly one operand has 1 |
| `~`      | Bitwise NOT | Inverts all bits (unary operator)       |

### Shift Operators

| Operator | Name                 | Description                             |
|----------|----------------------|-----------------------------------------|
| `<<`     | Left Shift           | Shifts bits left, filling with zeros    |
| `>>`     | Right Shift          | Shifts bits right, preserving sign bit  |
| `>>>`    | Unsigned Right Shift | Shifts bits right, filling with zeros   |

### Example

```pangy
class Main {
    def main() -> void {
        var a int = 10  // 1010 in binary
        var b int = 20  // 10100 in binary

        print("a & b = ", a & b)    // Bitwise AND: 0 (0000)
        print("a | b = ", a | b)    // Bitwise OR: 30 (11110)
        print("a ^ b = ", a ^ b)    // Bitwise XOR: 30 (11110)
        print("~a = ", ~a)          // Bitwise NOT: -11 (inversion of all bits)
        
        print("a << 1 = ", a << 1)  // Left shift by 1: 20 (10100)
        print("a >> 1 = ", a >> 1)  // Right shift by 1: 5 (101)
        print("a >>> 1 = ", a >>> 1) // Unsigned right shift by 1: 5 (101)
    }
}
```

Bitwise and shift operations are useful for low-level bit manipulation, implementing certain algorithms efficiently, and working with binary data.

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

## String Operations

Pangy supports various string operations including concatenation and comparison.

### String Literals and Escape Sequences

String literals are enclosed in double quotes and support the following escape sequences:

| Escape Sequence | Description |
|-----------------|-------------|
| `\"` | Double quote |
| `\'` | Single quote |
| `\\` | Backslash |
| `\n` | Newline |
| `\t` | Tab |
| `\r` | Carriage return |
| `\0` | Null character |
| `\b` | Backspace |
| `\f` | Form feed |
| `\v` | Vertical tab |

Example:
```pangy
var message string = "Line 1\nLine 2\nLine 3"
print("Tab-separated items:\t1\t2\t3")
print("Quotes: \"quoted text\"")
```

### String Concatenation

Strings can be concatenated using the `+` operator:

```pangy
var greeting string = "Hello, "
var name string = "World"
var message string = greeting + name
print(message)  // Prints "Hello, World"
```

### String Comparison

Strings can be compared using the comparison operators (`==`, `!=`, `<`, `>`, `<=`, `>=`):

```pangy
var str1 string = "hello"
var str2 string = "world"

if (str1 == str2) {
    print("Strings are equal")
} else {
    print("Strings are not equal")
}

if (str1 < str2) {
    print("str1 comes before str2 alphabetically")
}
```

String comparisons are case-sensitive and follow lexicographical ordering (dictionary order).

The compiler performs type checking to ensure that you only compare values of the same type. Attempting to compare a string with an integer will result in a compilation error.

## Command-Line Arguments

Pangy supports command-line arguments through the `main` method's parameters. You can access these arguments by defining your `main` method with the following signature:

```pangy
class Main {
    def main(argc int, argv string[]) -> void {
        // argc is the number of arguments (including the program name)
        // argv is an array of string arguments
    }
}
```

### Example

```pangy
class Main {
    def main(argc int, argv string[]) -> void {
        print("Number of arguments: ", argc)
        
        var i int = 0
        loop {
            if (i >= argc) {
                stop
            }
            
            print("Argument ", i, ": ", argv[i])
            i++
        }
    }
}
```

When you run this program with command-line arguments:

```bash
./my_program arg1 arg2 arg3
```

The output will be:

```
Number of arguments: 4
Argument 0: ./my_program
Argument 1: arg1
Argument 2: arg2
Argument 3: arg3
```

**Note**: The first argument (`argv[0]`) is always the program name, following the traditional C/C++ convention.
