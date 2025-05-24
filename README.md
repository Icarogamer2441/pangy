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
