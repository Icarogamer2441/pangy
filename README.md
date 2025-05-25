# Pangy Programming Language

Pangy is a custom programming language implemented with a Python-based lexer, parser, and compiler that targets x86-64 assembly (GAS syntax).

## Features

*   **Static Typing**: Variables are statically typed (e.g., `int`, `string`, `list`, custom class types).
*   **Class-Based Object Orientation**:
    *   Define classes with methods and variables.
    *   Public and private access modifiers for class variables (`public var x int`, `private var y string`).
    *   Instance creation using `ClassName.new()`.
    *   Access instance variables using `this:varName` (within the class) or `object::varName` (from outside, if public).
    *   Method calls: `this:methodName()` or `object.methodName()`.
*   **Control Flow**:
    *   `if/else` statements.
    *   `loop` (infinite loop) with `stop` (break).
*   **Basic Data Types**:
    *   `int` (64-bit integers).
    *   `string` (heap-allocated, null-terminated).
*   **Collections**:
    *   Dynamic Lists (e.g., `var myList int[] = [1, 2, 3]`).
        *   `append(list, value)`: Appends a value to a list (list may be reallocated).
        *   `pop(list)`: Removes and returns the last element from a list.
        *   `length(list_or_string)`: Returns the length of a list or string.
        *   Array access: `myList[index]`.
*   **Functions**:
    *   Built-in functions like `print(...)`, `show(...)`, `exit(code)`, `input(prompt)`, `to_int(string)`, `to_string(int)`, `index(string, position)`.
        *   `print(...)`: Display values with a newline at the end.
        *   `show(...)`: Display values without adding a newline.
    *   System interaction: `exec(command, mode)` for executing terminal commands.
    *   File I/O: `open(filename, mode)`, `close(file_pointer)`, `write(file_pointer, content)`, `read(file_pointer)`.
*   **Operators**:
    *   Arithmetic: `+`, `-`, `*`, `/`, `%`.
    *   String concatenation: `+`.
    *   Comparison: `<`, `>`, `<=`, `>=`, `==`, `!=` (works for integers and strings).
    *   Bitwise: `&`, `|`, `^`, `~`.
    *   Shift: `<<`, `>>` (arithmetic right shift), `>>>` (logical right shift).
    *   Postfix increment/decrement: `i++`, `i--`.
*   **Macros**:
    *   Simple text-substitution macros (e.g., `@define PI 3`).
    *   Function-like macros (e.g., `@define ADD(a,b) a+b`).
    *   Class-level macros.
*   **Comments**: Single-line comments with `//`.
*   **Assembly Generation**: Compiles to x86-64 assembly language (GAS syntax, Intel dialect).
    *   Uses `printf` for printing.
    *   Uses `malloc`, `realloc`, `strcpy`, `strcat`, `strlen` for string and list operations.
    *   Supports `main()` with `argc`, `argv` parameters.

## Compiler Implementation Details

*   **`this` Pointer Handling**: For instance methods, the `this` pointer (initially in `RDI`) is saved onto the stack at the beginning of the method. Subsequent accesses to `this:variable` or assignments like `this:variable = value` load this saved pointer to ensure correctness even if `RDI` is used by intermediate function calls (e.g., during string operations).
*   **String Operations**: String concatenation creates new strings on the heap. `strlen`, `strcpy`, `strcat` are used.
*   **List Operations**: Lists are heap-allocated with a header storing capacity and length. `append` may reallocate memory.
*   **Stack Management**: Method calls manage the stack for parameters and local variables. Stack is kept 16-byte aligned before `call` instructions.

## Getting Started

### Prerequisites

*   Python 3.x
*   GCC (for assembling and linking the output assembly code)

### Compilation and Execution

1.  **Run the Compiler**:
    ```bash
    python -m pangy.compiler your_source_file.pgy
    ```
    This will generate an assembly file (e.g., `output.s`).

2.  **Assemble and Link**:
    ```bash
    gcc -no-pie output.s -o your_program_name
    ```
    (Replace `output.s` with the actual name of your generated assembly file, and `your_program_name` with your desired executable name).

3.  **Run the Executable**:
    ```bash
    ./your_program_name
    ```

### Example `hello.pgy`

```pangy
class Main {
    def main() -> void {
        print("Hello, Pangy!")
        var x int = 10
        var y int = 20
        print("x + y = ", x + y)

        var s1 string = "Hello"
        var s2 string = "World"
        print(s1 + " " + s2 + "!")
    }
}
```

### Public/Private Example

See `examples/publicprivate.pgy` for an example of class variable access control. The compiler now correctly handles saving and restoring the `this` pointer, ensuring that member variable assignments within methods (like `this:result = ...`) work correctly even after calls to external functions that might modify `RDI` (such as string manipulation functions).

## Development

The language is under active development. Key components are:

*   `pangy/parser_lexer.py`: Contains the lexer and parser logic (AST node definitions).
*   `pangy/compiler.py`: The core compiler that translates AST to x86-64 assembly.

Feel free to explore the examples directory for more code samples.

## Include System

Pangy supports importing code from other files using the `include` directive. The system resolves include paths by first checking relative to the current file's directory, and then searching in a global library directory (`~/.pangylibs`).

### Import Syntax

The `include` directive uses a dot-separated path to specify what to import. This path can refer to:
- An entire file.
- A specific class within a file.
- An inner class within a parent class.
- A specific macro within a file.

**Syntax Examples:**

```pangy
// Import an entire file (e.g., 'mylibrary.pgy' or 'mylibrary/utils.pgy')
include mylibrary
include mylibrary.utils

// Import a specific class 'MyClass'
// This will look for 'mylibrary.pgy' and import 'MyClass' from it.
include mylibrary.MyClass
// This will look for 'mylibrary/utils.pgy' and import 'MyClass' from it.
include mylibrary.utils.MyClass

// Import an inner class 'Inner' from 'MyClass' in 'mylibrary/utils.pgy'
include mylibrary.utils.MyClass.Inner

// Import a specific macro '@myMacro'
// This will look for 'mylibrary.pgy' and import '@myMacro' from it.
include mylibrary.@myMacro
// This will look for 'mylibrary/utils.pgy' and import '@myMacro' from it.
include mylibrary.utils.@myMacro
```

### File Resolution

1.  **Local Directory**: Pangy first attempts to resolve the include path relative to the directory of the file containing the `include` statement.
    *   `include mymodule` will look for `mymodule.pgy` in the same directory.
    *   `include mylib.utils` will look for `mylib/utils.pgy` relative to the current file.

2.  **Pangy Libraries (`~/.pangylibs`)**: If the file is not found locally, Pangy searches in the `~/.pangylibs` directory. This directory serves as a central repository for shared libraries.
    *   `include mylib.utils` will look for `~/.pangylibs/mylib/utils.pgy`.
    *   `include anotherlib` will look for `~/.pangylibs/anotherlib.pgy`.

If an include cannot be resolved in either location, the compiler will issue an error.

### Example

**File: `~/.pangylibs/math/operations.pgy`** (after installing a 'math' library)
```pangy
class Operations {
    def add(a int, b int) -> int {
        return a + b
    }
    def subtract(a int, b int) -> int {
        return a - b
    }
}

macro PI 3
```

**File: `myproject/main.pgy`**
```pangy
// Assuming 'math/operations.pgy' exists in ~/.pangylibs
include math.operations.Operations // Import only the Operations class
include math.operations.@PI       // Import only the PI macro

// Example of a local import
// include localutils // Would look for 'localutils.pgy' in 'myproject/'

class Main {
    def main() -> void {
        var ops Operations = Operations.new()
        print("10 + 5 = ", ops.add(10, 5))
        print("PI = ", @PI)

        // var localHelper LocalUtils = LocalUtils.new() // If localutils.pgy existed
    }
}
```
The `main` function in `cli.py` now orchestrates the parsing of the main file and all its transitive dependencies (resolved via the new include logic) into a single Abstract Syntax Tree (AST). This master AST, containing all necessary classes and macros, is then passed to the compiler.

## Library Management

Pangy allows you to install and use shared libraries from a central location, `~/.pangylibs`.

### The `~/.pangylibs` Directory

This directory (located in your user's home directory, e.g., `/home/user/.pangylibs`) is where shared Pangy libraries are stored. When you `include` a module, and it's not found locally, the compiler will look for it here.

The structure inside `~/.pangylibs` mirrors the import path. For example, an `include mylib.utils` would correspond to `~/.pangylibs/mylib/utils.pgy`.

### Installing Libraries

You can install a library (a folder containing `.pgy` files and potentially subfolders) using the `pangy install` command.

**Command:**
```bash
python -m pangy install path/to/your/library_folder
```
Or, if `pangy` is an executable script in your PATH:
```bash
pangy install path/to/your/library_folder
```

**Example:**

Suppose you have a library named `stringutils` located at `~/dev/pangy_libs/stringutils`. This folder might contain:
```
stringutils/
  formatters.pgy
  validators.pgy
  helpers/
    common.pgy
```

To install it:
```bash
python -m pangy install ~/dev/pangy_libs/stringutils
```
This command will:
1. Create `~/.pangylibs` if it doesn't exist.
2. Copy the entire `stringutils` folder into `~/.pangylibs`, resulting in `~/.pangylibs/stringutils/`.

Now, in any Pangy project, you can use:
```pangy
include stringutils.formatters.MyFormatter
include stringutils.helpers.common
```
The compiler will find these in `~/.pangylibs/stringutils/formatters.pgy` and `~/.pangylibs/stringutils/helpers/common.pgy` respectively.

If a library with the same name already exists in `~/.pangylibs`, the `install` command will overwrite it, after printing a warning.

### Default Libraries

Pangy comes with a few default libraries for common tasks. You can install them using the provided script:

```bash
./install_libs.sh
```

This script will copy the `defaultlibs` directory (containing `listlib` and `splitlib`) into your `~/.pangylibs` folder.

**Using the Default Libraries:**

*   **List Utilities (`listlib`)**:
    *   Printing lists: `include listlib.listprint.ListPrint`
        *   Use `ListPrint.new().print_int(your_int_list)`
        *   Use `ListPrint.new().print_str(your_string_list)`
    *   Concatenating lists: `include listlib.listconc.ListConcat`
        *   Use `ListConcat.new().concat_int(list1, list2)`
        *   Use `ListConcat.new().concat_str(list1, list2)`

*   **String Splitting (`splitlib`)**:
    *   `include splitlib.split.Splitter`
        *   `Splitter.new().split(input_string, delimiter_string)`: Splits a string by a specified delimiter.
        *   `Splitter.new().split_empty(input_string)`: Splits a string by whitespace characters (space, tab, newline, carriage return).

**Example using `splitlib`:**
```pangy
include splitlib.split.Splitter
include listlib.listprint.ListPrint // For printing the result

class Main {
    def main() -> void {
        var text string = "hello world pangy lang"
        var splitter Splitter = Splitter.new()
        var printer ListPrint = ListPrint.new()

        var words string[] = splitter.split(text, " ")
        printer.print_str(words) // Output: {"hello", "world", "pangy", "lang"}

        var data string = "item1\titem2\nitem3"
        var items string[] = splitter.split_empty(data)
        printer.print_str(items) // Output: {"item1", "item2", "item3"}
    }
}
```

*   **Utility Functions (`utils`)**:
    *   Mathematical utilities: `include utils.math.math_utils.MathUtils`
        *   Provides functions like `min`, `max`, and `abs` for integer values.
        *   Example:
          ```pangy
          include utils.math.math_utils.MathUtils

          class Main {
              def main() -> void {
                  var math_utils MathUtils = MathUtils.new()
                  print("min(5, 10): ", math_utils.min(5, 10))
                  print("max(5, 10): ", math_utils.max(5, 10))
                  print("abs(-5): ", math_utils.abs((-5)))
              }
          }
          ```
    *   String utilities: `include utils.str.string_utils.StringUtils`
        *   Provides functions like `join`, `trim`, `startsWith`, and `endsWith` for string manipulation.
        *   Example:
          ```pangy
          include utils.str.string_utils.StringUtils

          class Main {
              def main() -> void {
                  var string_utils StringUtils = StringUtils.new()
                  var words string[] = {"hello", "world"}
                  print("join({\"hello\", \"world\"}, \" \"): ", string_utils.join(words, " "))
                  print("trim(\"   hello   \"): |", string_utils.trim("   hello   "), "|")
                  print("startsWith(\"hello\", \"he\"): ", string_utils.startsWith("hello", "he"))
                  print("endsWith(\"world\", \"ld\"): ", string_utils.endsWith("world", "ld"))
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

## Terminal Command Execution

Pangy provides functionality to execute terminal commands with the `exec()` function.

### Syntax

```pangy
exec(command, mode)
```

Where:
- `command`: A string containing the shell command to execute
- `mode`: An integer specifying how to handle execution:
  - `0`: Execute command visibly (output appears in the terminal) and don't return any value
  - `1`: Execute command in the background and return its output as a string
  - `2`: Execute command in the background and return its exit code as an integer

### Example

```pangy
class Main {
    def main() -> void {
        // Execute command visibly, no return value
        exec("echo 'Hello, World!'", 0)
        exec("ls", 0)
        
        // Execute command in background, return output as string
        var result string = exec("echo 'Hello, World!'", 1)
        print("Command output: ", result)
        
        // Execute command in background, return exit code
        var exitCode int = exec("ls -la", 2)
        print("Command exit code: ", exitCode)
    }
}
```

### Security Considerations

Be cautious when using the `exec()` function as it allows direct access to the underlying system. Always validate any user input before including it in commands to prevent security vulnerabilities like command injection.

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

### Returning Lists and Matrices from Methods

Pangy fully supports returning lists and matrices (multi-dimensional arrays) from methods. Method return types can be specified as `int[]`, `string[]`, `int[][]`, or `string[][]`.

```pangy
class ListUtils {
    // Return a list of integers
    def get_numbers() -> int[] {
        var numbers int[] = {1, 2, 3, 4, 5}
        return numbers
    }
    
    // Return a list of strings
    def get_words() -> string[] {
        var words string[] = {"hello", "world"}
        return words
    }
    
    // Return a matrix (2D array) of integers
    def get_matrix() -> int[][] {
        var matrix int[][] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
        return matrix
    }
    
    // Method to concatenate two lists
    def concat(a int[], b int[]) -> int[] {
        var result int[] = {}
        
        var i int = 0
        loop {
            if (i >= length(a)) {
                stop
            }
            
            append(result, a[i])
            i++
        }
        
        i = 0
        loop {
            if (i >= length(b)) {
                stop
            }
            
            append(result, b[i])
            i++
        }
        
        return result
    }
}

class Main {
    def main() -> void {
        var utils ListUtils = ListUtils.new()
        
        // Get a list from a method
        var numbers int[] = utils.get_numbers()
        print("First number: ", numbers[0])
        
        // Concatenate two lists
        var list1 int[] = {1, 2, 3}
        var list2 int[] = {4, 5, 6}
        var combined int[] = utils.concat(list1, list2)
        
        var i int = 0
        loop {
            if (i >= length(combined)) {
                stop
            }
            print("combined[", i, "]: ", combined[i])
            i++
        }
        
        // Get a matrix from a method
        var matrix int[][] = utils.get_matrix()
        print("Matrix element [1][1]: ", matrix[1][1])  // Prints 5
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
  Returns the current number of elements in the list. Also works with strings to get the number of characters.
  ```pangy
  var numbers int[] = {1, 2, 3, 4}
  var len int = length(numbers) // len will be 4
  print("List length: ", len)
  
  var text string = "Hello"
  var textLen int = length(text) // textLen will be 5
  print("String length: ", textLen)
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
python -m pangy compile your_program.pgy
# or just:
python -m pangy your_program.pgy
```

To install a library:
```bash
python -m pangy install path/to/library_folder
```

Refer to `python -m pangy --help` for all commands and options.

Common Options for `compile`:
- `-o OUTPUT` - Specify output file name for executable or assembly.
- `-S` - Output assembly code (.s file) instead of an executable.
- `--ast` - Print the combined Abstract Syntax Tree from all included files and exit.
- `--tokens` - Print the lexical tokens from the main input file and exit.

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

### String Functions

Pangy provides built-in functions for working with strings:

- `length(str) -> int`
  Returns the number of characters in the string.
  ```pangy
  var text string = "Hello, world!"
  var len int = length(text) // len will be 13
  print("String length: ", len)
  ```

- `index(str, position) -> int`
  Returns the ASCII value of the character at the specified position (zero-based index).
  ```pangy
  var text string = "Hello"
  var char int = index(text, 0) // char will be 72 (ASCII for 'H')
  print("First character: ", char)
  
  // Display as character instead of ASCII value
  print("First character: ", index(text, 0).as_string())  // Prints 'H' instead of 72
  ```
  
  Accessing an index out of bounds (negative or beyond the string length) will result in a runtime error.

### Type Display Control

Pangy allows you to control how values are displayed in print statements using the following methods:

- `.as_string()` - Displays an integer value as its corresponding ASCII character
  ```pangy
  var ascii int = 65
  print(ascii)            // Prints: 65
  print(ascii.as_string()) // Prints: A
  
  print(index("Hello", 0))           // Prints: 72 (ASCII value of 'H')
  print(index("Hello", 0).as_string()) // Prints: H
  ```

- `.as_int()` - Explicitly displays a value as an integer (useful for clarity)
  ```pangy
  var value = index("9", 0)
  print(value.as_int()) // Explicitly prints as integer: 57 (ASCII code for '9')
  ```

## Output Functions

Pangy provides two output functions with different behavior:

### print() - Output with Newline

The `print()` function displays values to the console and automatically adds a newline at the end.

```pangy
print("Hello")     // Outputs: Hello\n
print("A", "B", "C") // Outputs: ABC\n
```

### show() - Output without Newline

The `show()` function displays values to the console without adding a newline at the end.

```pangy
show("Hello ")
show("World")     // Outputs: Hello World (no newline)
print("!")        // Outputs: !\n

// Useful for prompts
show("Enter your name: ")
var name string = input("")  // Input prompt on the same line
```

Both functions support the same arguments and type handling, including:
- String literals
- Integer literals 
- Variables
- Expressions
- Function calls
- Type display control (`.as_string()`, `.as_int()`)

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

## Class Variables (Public and Private)

Pangy supports class-level variables with public and private visibility modifiers. These variables are associated with object instances and can be accessed using the appropriate syntax.

### Declaration

Class variables must be declared at the class level (outside any method) using the `public` or `private` keywords:

```pangy
class Math {
    public var result int = 0  // Public class variable

    private var a int = 0      // Private class variable
    private var b int = 0      // Private class variable

    def add(a int, b int) -> void {
        this:a = a
        this:b = b
        this:result = this:a + this:b
    }
}
```

### Access Modifiers

- `public`: Variables are accessible from outside the class
- `private`: Variables are only accessible from within the class methods

### Accessing Class Variables

From inside the class methods, use the `this:` prefix:

```pangy
this:variable_name = value
var x = this:variable_name
```

From outside the class, use the `::` accessor for public variables:

```pangy
var math Math = Math.new()
math.add(1, 2)
print(math::result)  // Accessing public variable
// print(math::a)    // Error: Cannot access private variable
```

### Example

```pangy
class Math {
    public var result int = 0

    private var a int = 0
    private var b int = 0

    def add(a int, b int) -> void {
        this:a = a
        this:b = b
        this:result = this:a + this:b
    }
}

class Main {
    def main() -> void {
        var math Math = Math.new()
        math.add(1, 2)
        print(math::result)
        // print(math::a) // this will throw an error of private access
        // print(math::b) // this will also throw an error of private access
    }
}
```

Class variables can be of any type, including primitives (int, string) and custom classes.
