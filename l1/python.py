
'''
Basic Python: Basic data types (Containers, Lists, Dictionaries, Sets, Tuples), Functions, Classes'
'''

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
# print(quicksort([3,6,8,10,1,2,1]))




'''numbers'''
# x = 3
# print(x, type(x))
# print("x+1= ",x + 1)   # Addition
# print("x-1= ",x - 1)   # Subtraction
# print("x*2= ",x * 2)   # Multiplication
# print("x**2= ",x ** 2)  # Exponentiation
# x += 1
# print("x+=1 ",x)
# x *= 2
# print("x*=1 ",x)
# y = 2.5
# print(type(y))
# print(y, y + 1, y * 2, y ** 2)

'''boolean'''
# t, f = True, False
# print(type(t))
#
# print(t and f) # Logical AND;
# print(t or f)  # Logical OR;
# print(not t)   # Logical NOT;
# print(t != f)  # Logical XOR;

'''Strings'''
# hello = 'hello'   # String literals can use single quotes
# world = "world"   # or double quotes; it does not matter
# print(hello, len(hello))
#
# hw = hello + ' ' + world  # String concatenation
# print(hw)
#
# hw12 = '{} {} {}'.format(hello, world, 12)  # string formatting
# print(hw12)
#
# name = "Eric"
# age = 74
# print(f"Hello, {name}. You are {age}.")
#
# s = "hello"
# print(s.capitalize())  # Capitalize a string
# print(s.upper())       # Convert a string to uppercase; prints "HELLO"
# print(s.rjust(7))      # Right-justify a string, padding with spaces
# print(s.center(7))     # Center a string, padding with spaces
# print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another
# print('  world '.strip())  # Strip leading and trailing whitespace

'''Containers'''
# xs = [3, 1, 2]   # Create a list
# print(xs, xs[2])
# print(xs[-1])     # Negative indices count from the end of the list; prints "2"
#
# xs[2] = 'foo'    # Lists can contain elements of different types
# print(xs)
#
# xs.append('bar') # Add a new element to the end of the list
# print(xs)
#
# x = xs.pop()     # Remove and return the last element of the list
# print(x, xs)
#

'''Slicing'''
# nums = list(range(5))    # range is a built-in function that creates a list of integers
# print(nums)         # Prints "[0, 1, 2, 3, 4]"
# print(nums[2:4])    # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
# print(nums[2:])     # Get a slice from index 2 to the end; prints "[2, 3, 4]"
# print(nums[:2])     # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
# print(nums[:])      # Get a slice of the whole list; prints ["0, 1, 2, 3, 4]"
# print(nums[:-1])    # Slice indices can be negative; prints ["0, 1, 2, 3]"
# nums[2:4] = [8, 9] # Assign a new sublist to a slice
# print(nums)         # Prints "[0, 1, 8, 9, 4]"

'''Loops'''

# animals = ['cat', 'dog', 'monkey']
# for animal in animals:
#     print(animal)
#
# # If you want access to the index of each element within the body of a loop, use the built-in enumerate function:
# animals = ['cat', 'dog', 'monkey']
# for idx, animal in enumerate(animals):
#     print('#{}: {}'.format(idx + 1, animal))

'''List comprehensions:'''

# # When programming, frequently we want to transform one type of data into another.
# # As a simple example, consider the following code that computes square numbers:
# nums = [0, 1, 2, 3, 4]
# squares = []
# for x in nums:
#     squares.append(x ** 2)
# print(squares)
#
# # You can make this code simpler using a list comprehension:
# nums = [0, 1, 2, 3, 4]
# squares = [x ** 2 for x in nums]
# print(squares)
#
# # List comprehensions can also contain conditions:
# nums = [0, 1, 2, 3, 4]
# even_squares = [x ** 2 for x in nums if x % 2 == 0]
# print(even_squares)

'''Dictionaries'''

# # A dictionary stores (key, value) pairs, similar to a Map in Java or an object in Javascript.
# # You can use it like this:
# d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
# print(d['cat'])       # Get an entry from a dictionary; prints "cute"
# print('cat' in d)     # Check if a dictionary has a given key; prints "True"
#
# d['fish'] = 'wet'    # Set an entry in a dictionary
# print(d['fish'])      # Prints "wet"
#
# # error as its not a key
# # print(d['monkey'])  # KeyError: 'monkey' not a key of d
#
# print(d.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
# print(d.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
#
# del d['fish']        # Remove an element from a dictionary
# print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"
#
# # It is easy to iterate over the keys in a dictionary:
# d = {'person': 2, 'cat': 4, 'spider': 8}
# for animal, legs in d.items():
#     print('A {} has {} legs'.format(animal, legs))
#
# # Dictionary comprehensions:
# # These are similar to list comprehensions, but allow you to easily construct dictionaries.
# # For example:
# nums = [0, 1, 2, 3, 4]
# even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
# print(even_num_to_square)

'''Sets'''
# # A set is an unordered collection of distinct elements. As a simple example, consider the following:
# animals = {'cat', 'dog'}
# print('cat' in animals)   # Check if an element is in a set; prints "True"
# print('fish' in animals)  # prints "False"
#
# animals.add('fish')      # Add an element to a set
# print('fish' in animals)
# print(len(animals))       # Number of elements in a set;
#
# animals.add('cat')       # Adding an element that is already in the set does nothing
# print(len(animals))
# animals.remove('cat')    # Remove an element from a set
# print(len(animals))
#
# # Loops: Iterating over a set has the same syntax as iterating over a list;
# # however since sets are unordered,
# # you cannot make assumptions about the order in which you visit the elements of the set:
# animals = {'cat', 'dog', 'fish'}
# for idx, animal in enumerate(animals):
#     print('#{}: {}'.format(idx + 1, animal))
#
# # Set comprehensions: Like lists and dictionaries, we can easily construct sets using set comprehensions
# from math import sqrt
# print({int(sqrt(x)) for x in range(30)})

'''Tuples'''
# A tuple is an (immutable) ordered list of values.
# A tuple is in many ways similar to a list;
# one of the most important differences is that tuples can be used as keys in dictionaries and as elements of sets,
# while lists cannot.
# Here is a trivial example:

# d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
# t = (5, 6)       # Create a tuple
# print(type(t))
# print(d[t])
# print(d[(1, 2)])

# t[0] = 1  # TypeError: 'tuple' object does not support item assignment


'''Functions'''
# # Python functions are defined using the def keyword. For example:
# def sign(x):
#     if x > 0:
#         return 'positive'
#     elif x < 0:
#         return 'negative'
#     else:
#         return 'zero'
#
# for x in [-1, 0, 1]:
#     print(sign(x))
#
# # We will often define functions to take optional keyword arguments, like this:
# def hello(name, loud=False):
#     if loud:
#         print('HELLO, {}'.format(name.upper()))
#     else:
#         print('Hello, {}!'.format(name))
#
# hello('Bob')
# hello('Fred', loud=True)

'''Classes'''
# The syntax for defining classes in Python is straightforward:
# class Greeter:
#     # Constructor
#     def __init__(self, name):
#         self.name = name  # Create an instance variable
#
#     # Instance method
#     def greet(self, loud=False):
#         if loud:
#           print('HELLO, {}!'.format(self.name.upper()))
#         else:
#           print('Hello, {}'.format(self.name))
#
# g = Greeter('Fred')  # Construct an instance of the Greeter class
# g.greet()            # Call an instance method; prints "Hello, Fred"
# g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"
#
