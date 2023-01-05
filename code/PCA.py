# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:40:00 2021

@author: LucaS
"""

import numpy
    
def PCAmatrix(C,m):
    #U, s, Vh = numpy.linalg.svd(C);
    #P = U[:, 0:m];
    s,U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:,0:m]
    return P

def dataset_with_PCA(DTR, m):
    C = numpy.cov(DTR, ddof=0)
    P = PCAmatrix(C, m)
    return numpy.dot(P.T, DTR)
    

def apply_pca_training_and_test(DTR, DTE, m):
    C = numpy.cov(DTR, ddof=0)
    P = PCAmatrix(C, m)
    return numpy.dot(P.T, DTR), numpy.dot(P.T, DTE)