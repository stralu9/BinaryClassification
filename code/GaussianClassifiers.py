# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:34:34 2021

@author: LucaS
"""

import numpy
import evaluationModel as em
import crossValidation as cv
import PCA as pca

def computemuAndC(DTR, LTR):
    mu = []
    C = []
    
    for x in range(2):
        mu.append(DTR[:, LTR==x].mean(axis=1).reshape((DTR.shape[0],1)))
        C.append(numpy.cov(DTR[:, LTR==x], ddof=0))
    return mu,C

def computemuAndC_naive(DTR, LTR):
    mu = [];
    C = [];
    identityMatrix = numpy.eye(DTR.shape[0])
    
    for x in range(2):
        mu.append(DTR[:, LTR == x].mean(axis=1).reshape((DTR.shape[0],1)))
        C.append(numpy.cov(DTR[:,LTR==x], ddof=0)*identityMatrix)
    return mu,C

def computemuAndC_tied(DTR, LTR):
    mu = []
    C = numpy.zeros((DTR.shape[0],DTR.shape[0]))

    for x in range(2):
        mu.append(DTR[:, LTR == x].mean(axis=1).reshape((DTR.shape[0],1)))
        C+= numpy.cov(DTR[:,LTR==x], ddof=0)
    return mu,C/DTR.shape[1]

def computemuAndC_tied_naive(DTR, LTR):
    mu = [];
    C = numpy.zeros((DTR.shape[0],DTR.shape[0]))
    identityMatrix = numpy.eye(DTR.shape[0])

    for x in range(2):
        mu.append(DTR[:, LTR == x].mean(axis=1).reshape((DTR.shape[0],1)))
        C += numpy.cov(DTR[:,LTR==x], ddof=0)
        
    return mu,C*identityMatrix/DTR.shape[1]

def GAU_logpdf_ND(x, mu, C):
    _, det = numpy.linalg.slogdet(C)
    return numpy.diag(-0.5*x.shape[0]*numpy.log(2*numpy.pi) - 0.5*det - 0.5*numpy.dot((x-mu).T, numpy.dot(numpy.linalg.inv(C), x-mu)))

def getllr(DTE, mu, C):  
    Slog = numpy.zeros((2, DTE.shape[1]));
    for i in range(2):
        Slog[i] = GAU_logpdf_ND(DTE, mu[i], C[i])  
    return Slog[1,:]-Slog[0,:]

def getllr_tied(DTE, mu, C):
    
    Slog = numpy.zeros((2, DTE.shape[1]));
    for i in range(2):
        Slog[i] = GAU_logpdf_ND(DTE, mu[i], C)
    
    return Slog[1,:]-Slog[0,:]

def tied_naive_cov_evaluation(DTR, LTR, DTE, LTE, working_points):
    mu, C = computemuAndC_tied_naive(DTR, LTR)
    llr = getllr_tied(DTE, mu, C)
    dcf = numpy.zeros(len(working_points))
    
    for i in range(len(working_points)):
        dcf[i] += em.compute_min_DCF(llr, LTE, working_points[i])
    
    print(dcf)

def tied_cov_evaluation(DTR, LTR, DTE, LTE, working_points):
    mu, C = computemuAndC_tied(DTR, LTR)
    llr = getllr_tied(DTE, mu, C)
    dcf = numpy.zeros(len(working_points))
    
    for i in range(len(working_points)):
        dcf[i] += em.compute_min_DCF(llr, LTE, working_points[i])
    
    print(dcf)
    
    
def full_cov_evaluation(DTR, LTR, DTE, LTE, working_points):
    mu, C = computemuAndC(DTR, LTR)
    llr = getllr(DTE, mu, C)
    dcf = numpy.zeros(len(working_points))
        
    for i in range(len(working_points)):
        dcf[i] += em.compute_min_DCF(llr, LTE, working_points[i])
    
    print(dcf)
    

def naive_cov_evaluation(DTR, LTR, DTE, LTE, working_points):
    mu, C = computemuAndC_naive(DTR, LTR)
    llr = getllr(DTE, mu, C)
    dcf = numpy.zeros(len(working_points))
    
    for i in range(len(working_points)):
        dcf[i] += em.compute_min_DCF(llr, LTE, working_points[i])
    
    print(dcf)
    
def full_cov_model(D,L,n, working_points, m):
    scores = numpy.array([])
    Lf = numpy.array([], dtype=numpy.int32)
    dcf = numpy.zeros((len(working_points)))

    for i in range(n):        
        DTR, LTR, DTE, LTE = cv.k_fold(i, D, L, n)
        if m != DTR.shape[0]:
            DTR, DTE = pca.apply_pca_training_and_test(DTR, DTE, m)
        mu, C = computemuAndC(DTR, LTR)
        scores = numpy.append(scores, getllr(DTE, mu, C))
        Lf = numpy.append(Lf, LTE)
        
    for j in range(len(working_points)):
        dcf[j] = em.compute_min_DCF(scores, Lf, working_points[j])
    
    print(dcf)
    
def naive_cov_model(D,L,n, working_points, m):
    scores = numpy.array([])
    Lf = numpy.array([], dtype=numpy.int32)
    dcf = numpy.zeros(len(working_points))
    for i in range(n):
        DTR, LTR, DTE, LTE = cv.k_fold(i, D, L, n)
        if m != DTR.shape[0]:
            DTR, DTE = pca.apply_pca_training_and_test(DTR, DTE, m)
        
        mu, C = computemuAndC_naive(DTR, LTR)
        scores = numpy.append(scores, getllr(DTE, mu, C))
        Lf = numpy.append(Lf, LTE)
        
    for j in range(len(working_points)):
        dcf[j] = em.compute_min_DCF(scores, Lf, working_points[j])
    
    print(dcf)
    
def tied_cov_model(D,L,n, working_points, m):
    scores = numpy.array([])
    Lf = numpy.array([], dtype=numpy.int32)
    dcf = numpy.zeros((len(working_points)))
    for i in range(n):
        DTR, LTR, DTE, LTE = cv.k_fold(i, D, L, n)
        if m != DTR.shape[0]:
            DTR, DTE = pca.apply_pca_training_and_test(DTR, DTE, m)
        
        mu, C = computemuAndC_tied(DTR, LTR)
        scores = numpy.append(scores, getllr_tied(DTE, mu, C))
        Lf = numpy.append(Lf, LTE)
        
    for j in range(len(working_points)):
        dcf[j] = em.compute_min_DCF(scores, Lf, working_points[j])
    
    print(dcf)

def tied_naive_cov_model(D,L,n, working_points, m):
    scores = numpy.array([])
    Lf = numpy.array([], dtype=numpy.int32)
    dcf = numpy.zeros((len(working_points)))
    for i in range(n):
        DTR, LTR, DTE, LTE = cv.k_fold(i, D, L, n)
        if m != DTR.shape[0]:
            DTR, DTE = pca.apply_pca_training_and_test(DTR, DTE, m)
        
        mu, C = computemuAndC_tied_naive(DTR, LTR)
        scores = numpy.append(scores, getllr_tied(DTE, mu, C))
        Lf = numpy.append(Lf, LTE)
        
    for j in range(len(working_points)):
        dcf[j] = em.compute_min_DCF(scores, Lf, working_points[j])
    
    print(dcf)
    
    