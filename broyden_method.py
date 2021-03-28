import numpy as np
import math 
import pprint

np.set_printoptions(precision=3)

def gauss_elim_solve(a, x):
    # gauss elimination algorithm
    n = len(a)
    a = np.concatenate((a,x),axis=1) # concatenate A and x
    mp = np.zeros(shape=(n,n+1)) # init empty array for multiplier matrix

    for k in range(n):
        if a[k][k] == 0:
            break
        
        for i in range(k+1, n):
            mp[i][k] = a[i][k]/a[k][k]
    
            for j in range(n+1):
                a[i][j] = a[i][j] - mp[i][k] * a[k][j]

    # back subs algorithm
    u = np.delete(a,n,1) # splitting u
    b = np.delete(a,range(n),1) # splitting b

    x=np.zeros(n) # array for x results

    for j in reversed(range(n)):
        if u[j][j] == 0:
            break
        
        x[j] = b[j] / u[j][j]

        for i in range(j):
            b[i] = b[i] - u[i][j]*x[j]

    x = x.reshape(n,1)

    return x

def broyden_method(b,x):
    k = 1
    tol = np.array([[1000],[1000]])
    tol_stop = np.array([[0.1],[0.1]])
    
    print("k", "     Bk     ","   xk   ", "    F(xk)    ")

    while tol_stop[1][0] < tol[1][0]:
        b_k = b
        x_k = x
        fofx_k = fx(x_k)
        
        s = gauss_elim_solve(b, -fx(x))
        x = x_k + s
        
        fofx = fx(x)
        
        y = fofx - fofx_k
        b = b_k + np.dot((y - np.dot(b_k, s)),s.transpose()) / (np.dot(s.transpose(), s))

        tol = abs((x - x_k)*100)
        print(k, b ,"   ", x.transpose(),"   ", fofx.transpose())

        k+=1

def fx(x):
    return np.array( [[1*x[0][0]+2*x[1][0]-2],
                      [1*pow(x[0][0],2)+4*pow(x[1][0],2)-4]])

b = np.array( [ [1, 2], 
                [2, 16] ] )

x = np.array( [[1], [2]])

broyden_method(b, x)