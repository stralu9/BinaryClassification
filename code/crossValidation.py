# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:58:04 2021

@author: LucaS
"""
import numpy

def k_fold(i, D, L, n):
    samples_per_fold = int(numpy.ceil(D.shape[1]/n))
    DTR = numpy.hstack([D[:,0:i*samples_per_fold], D[:,((i+1)*samples_per_fold):]])
    DTE = D[:, (i*samples_per_fold):((i+1)*samples_per_fold)]
    LTR = numpy.hstack([L[0:i*samples_per_fold], L[((i+1)*samples_per_fold):]])
    LTE = L[(i*samples_per_fold):((i+1)*samples_per_fold)]
    return DTR, LTR, DTE, LTE

