# -*- coding: utf-8 -*-
import numpy as np

def sinatsinbt(t,a,b):
    x = np.sin(a*t)
    y = np.sin(b*t)
    return x,y