
from functools import partial

def sum(a,b,c):
    return (a+b)*c

new_sum_1 = partial(sum, a=10)
new_sum_2 = partial(sum, b=10)
new_sum_3 = partial(sum, c=10)

print(new_sum_1(b=2,c=3)) # necessary to give key words for passing arguments
print(new_sum_2(a=2,c=3)) # necessary to give key words for passing arguments
print(new_sum_3(2,3)) # optional to give key words for passing arguments
