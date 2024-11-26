import numpy as np
x = (1,2)
y = (3,4)
z = tuple(np.array(x) + np.array(y))
print(z)