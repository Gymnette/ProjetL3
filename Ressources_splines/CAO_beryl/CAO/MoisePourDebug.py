# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 15:35:29 2019

@author: Béryl
"""

def MatriceN(n,h):
    M = np.zeros((n-2,n))
    for i in range(n-2):
        M[i][i] = 1
        M[i][i+2] =-1
    return (1.0/h)*M

def MatriceH03(N,n,uk,xi,h):
    M = np.zeros((N,n))
    j = 0
    for i in range(n-1):
        for ki in range(N):
            if xi[i] <= uk[ki] and uk[ki] <= xi[i+1]:
                M[j][i]=H0((uk[ki]-xi[i])/h)
                M[j][i+1] = H3((uk[ki]-xi[i])/h)
                j+=1
    return M

def MatriceH12(N,n,uk,xi,h):
    M=np.zeros((N,n))
    j = 0
    for i in range(n-1):
        for ki in range(N):
            if xi[i]<=uk[ki] and uk[ki] <= xi[i+1]:
                M[j][i] = H1((uk[ki]-xi[i])/h)
                M[j][i+1] = H2((uk[ki]-xi[i])/h)
                j+=1
    return h*M