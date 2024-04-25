from response import get_response
import numpy as np


site = [[False,  True, False, False,  True,  True, False, False,  True, True],
 [False, False,  True,  True, False,  True, False,  True, False, False],
 [ True,  True, False,  True, False, False,  True, False, False, False],
 [ True, False,  True,  True, False,  True,  True,  True, False, False],
 [False, False, False, False,  True,  True,  True, False,  True, True],
 [ True, False, False, False, False,  True, False,  True, False, True],
 [False,  True, False,  True, False, False, False,  True, False, True],
 [ True, False,  True,  True,  True, False, False,  True, False, False],
 [False, False,  True,  True, False, False,  True,  True, False, True],
 [ True,  True,  True, False,  True,  True, False,  True,  True, True]]

F = 0.5
array1 = np.random.randint(0, 11, size=(10, 10))
array2 = np.random.randint(0, 11, size=(10, 10))
array3 = np.random.randint(0, 11, size=(10, 10))
my_string1 = "[" + ",".join(str(element) for element in site[1]) + "]"
prompt_content_crossover = "I have one lists S1:" + my_string1  + \
                             "Converts this string to a Boolean list B1" \
                             "Please return the final value of list B1 directly." \
                             "Do not give additional explanations."
# prompt_content_crossover = "Give a random two-dimensional array"
Offspring_one = get_response(prompt_content_crossover)
print(Offspring_one)
print(Offspring_one[1])
print(type(Offspring_one))