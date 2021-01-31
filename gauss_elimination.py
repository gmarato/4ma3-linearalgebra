# imports
import numpy as np

# Ab combined array
a = np.array( [[1, 2, 1, -1,  5],
               [3, 2, 4,  4, 16],
               [4, 4, 3,  4, 22],
               [2, 0, 1,  5, 15] ])

# length of array
n = len(a)
mp = np.zeros(shape=(4,5))

# gauss elimination
for k in range(n):
    if a[k][k] == 0:
        break

    for i in range(k+1, n):
        mp[i][k] = a[i][k]/a[k][k]
    
        for j in range(n+1):
            a[i][j] = a[i][j] - mp[i][k] * a[k][j]

u = np.delete(a,n,1)
b = np.delete(a,range(n),1)


x=np.zeros(n)

# back substitution
for j in reversed(range(n)):
    if u[j][j] == 0:
        break
    
    x[j] = b[j] / u[j][j]

    for i in range(j):
        b[i] = b[i] - u[i][j]*x[j] 