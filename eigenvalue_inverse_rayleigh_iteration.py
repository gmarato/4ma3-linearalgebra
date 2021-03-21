import numpy as np 
from random import randint

MIN_ITERATIONS = 5

def norm(vector, type):
    # calculate the infinity norm of a vector
    if type == "inf":
        norm = max(abs(vector))

    return norm

def identity_matrix(a):
    # find an identity matrix of a given matrix
    n = len(a)
    I = np.zeros(shape=(n,n))

    for i in range(n):
        I[i][i] = 1.000

    return I

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

def inverse_iteration(A,x):
    print("INVERSE ITERATION \n")
    print("Random X used:")
    print(x,"\n")

    iteration = 1 

    eigen_tol_stop = 0.01 
    eigen_tol = 100
    ynorm_old = 1

    while eigen_tol_stop < eigen_tol:
        
        y = gauss_elim_solve(A, x)

        ynorm = norm(y,"inf") # inf norm calculation
        
        if iteration > MIN_ITERATIONS: # perform at least few iterations before stopping
            eigen_tol = abs( (ynorm - ynorm_old) / ynorm ) * 100
            ynorm_old = ynorm

        x = y / ynorm # normalize

        print("Iteration","x","Ynorm")
        print(iteration,  x.transpose(), ynorm, "\n")
        
        iteration += 1

def rayleigh_quotient_iteration(A,x): 
    print("RAYLEIGH QUOTIENT ITERATION \n") 
    print("Random X used:")
    print(x,"\n")

    iteration = 1 

    eigen_tol_stop = 0.01 
    eigen_tol = 100
    ynorm_old = 1
    I = identity_matrix(A)

    while eigen_tol_stop < eigen_tol:
        
        x_t = x.transpose()
        
        sigma = ( np.dot(np.dot(x_t,A),x) )/( np.dot(x_t,x) )

        Acalc = A - sigma*I

        y = gauss_elim_solve(Acalc, x)
        #y = np.linalg.solve(Acalc, x)
        
        ynorm = norm(y,"inf") # inf norm calculation
        
        if iteration > MIN_ITERATIONS: # perform at least few iterations before stopping
            eigen_tol = abs( (ynorm - ynorm_old) / ynorm ) * 100
            ynorm_old = ynorm

        if ynorm!=0:
            x = y / ynorm # normalize
        else:
            break
   
        print("Iteration","x","Ynorm", "sigma")
        print(iteration, x.transpose(), ynorm, sigma,"\n")
        
        iteration += 1

A = np.array( [[1, -1, 0], [0, -4, 2], [0, 0, -2]] ) # input array
x = np.random.randint(abs(A).max(), size=(len(A),1)) # generate a random vector x 

inverse_iteration(A,x)
rayleigh_quotient_iteration(A,x)



