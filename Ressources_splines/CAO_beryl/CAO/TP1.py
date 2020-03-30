# -*- coding: utf-8 -*-

############
#Exercise 1#
############

import numpy as np

# 5 6 7 8 9 10
v1 = np.arange(5,11) 

# 0 0 0 5 6 7 8 9 10 1 1 1 1
v2 = np.concatenate((np.zeros(3), v1, np.ones(4)))

# 0 1 2 3 4 9 7 5 3 1 
v3 = np.concatenate((np.arange(5),np.arange(9,0,-2)))

# 1 1 1 1 0 0 0
# 1 1 1 1 0 0 0
m1 = np.concatenate((np.ones((2,4)), np.zeros((2,3))), axis = 1)

# 1 3 5 7 9
# 8 6 4 2 0
# 8 6 4 2 0
m2 = np.vstack((np.arange(1,10,2),np.tile(np.arange(8,-1,-2),[2,1])))

#0 0 1 1 1 1 1
#0 0 1 1 1 1 1
#0 0 1 2 3 4 5
m3 = np.hstack((np.zeros((3,2)), np.vstack((np.ones((2,5)),np.arange(1,6)))))
    

############
#Exercise 2#
############

# 2 1 0 0 0 0 0 0
# 1 4 1 0 0 0 0 0
# 0 1 4 1 0 0 0 0
# 0 0 1 4 1 0 0 0
# 0 0 0 1 4 1 0 0
# 0 0 0 0 1 4 1 0
# 0 0 0 0 0 1 4 1
# 0 0 0 0 0 0 1 2

#Première méthode :
m4 = np.eye(8)*2 + np.diag(np.concatenate([np.zeros(1), np.repeat(2,6), np.zeros(1)])) + np.diag(np.repeat(1,7), k = 1) + np.diag(np.repeat(1,7), k = 1)
# Autre méthode, moins efficace :
# m4 = np.eye(8)*4 + np.vstack((np.hstack((np.zeros((7,1)),np.eye(7))),np.hstack((np.zeros(7),[-2])))) + np.hstack((np.vstack((np.hstack(([-2],np.zeros(6))), np.eye(7))),np.zeros((8,1))))


############
#Exercise 3#
############

n_max = int(input("Enter N_max : "))
triangle = []
for n in range(n_max+1):
    if (n == 0 or n == 1) :
        triangle.append(1)
    else :
        newTriangle = []
        for i in range(n+1) :
            if (i-1 < 0) :
                t1 = 0
            else :
                t1 = triangle[i-1]
            if (i >= n) :
                t2 = 0
            else :
                t2 = triangle[i]
            newTriangle.append(t1+t2)
        triangle = newTriangle.copy()
            
    print("n = ",n," : ",triangle)
    
    
############
#Exercise 4#
############
    
n_max = int(input("Enter N_max : "))
numbers = list(range(2,n_max+1))
for elem in numbers :
    if elem <= int(np.sqrt(n_max)) :
        n = elem**2
        while (n <= n_max) :
            if (n in numbers) :
                numbers.remove(n);
            n += elem            
    else :
        break
print(numbers)


############
#Exercise 5#
############

import matplotlib.pyplot as plt
import exoLissajousDrawing as exLD


t = np.arange(0.0, 2*np.pi+0.1, 0.001)
nCouple = 1
for couple in [(2,3), (2,1), (2,5), (3,4), (3,5), (3,7)]:
    plt.subplot(2,3,nCouple)
    x,y = exLD.sinatsinbt(t,float(couple[0]),float(couple[1]))
    plt.plot(x,y)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    title = "a ="+str(couple[0])+", b ="+str(couple[1])
    plt.title(title)
    nCouple += 1


        
    