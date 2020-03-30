# -*- coding: utf-8 -*-
from math import sinh
import numpy as np
import matplotlib.pyplot as plt

def umu(x):
    return 1-x

def phi(x,omegai):
    return (sinh(omegai*x) - x * sinh(omegai))/(sinh(omegai)-omegai)

def phiumu(x,omegai):
    return phi(umu(x),omegai)

def 

if __name__ == "__main__":
    u = np.arange(0,1.005,0.01)
    liste_omegai = [0.1,1,5,10,25,50]
    plt.subplot(231)
    for i in range(6):
        plt.subplot(int("23"+str(i+1)))
        omegai = liste_omegai[i]
        titre = str(omegai)
        plt.title(titre)
        plt.plot(u,u,label="u")
        plt.plot(u,list(map(umu,u)),label="1-u")
        plt.plot(u,list(map(lambda y:phi(y,omegai),u)),label="phi(u)")
        plt.plot(u,list(map(lambda y:phiumu(y,omegai),u)),label="phi(1-u)")
        plt.legend()
    plt.show()
    
    
