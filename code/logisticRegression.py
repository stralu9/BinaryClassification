# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:03:04 2021

@author: LucaS
"""
import math

import numpy
import crossValidation as cv
import evaluationModel as em
import scipy
import matplotlib.pyplot as plt

def logreg_obj(v, DTR, LTR, l, prior = None):
    w = numpy.array(v[0:-1]).reshape(DTR.shape[0],1) 
    b = v[-1]
    value_T = value_F = 0
    
    if prior == None:
        prior = LTR.sum() / LTR.size
    
    value_T += LTR*numpy.logaddexp(0,-numpy.dot(w.T,DTR)-b)
    value_F += (1-LTR)*numpy.logaddexp(0,(numpy.dot(w.T,DTR)+b))

    value_T = value_T.sum() * (prior/LTR.sum())
    value_F = value_F.sum() * ((1-prior)/(LTR.size-LTR.sum()))
    first = l/2*numpy.linalg.norm(w)**2
    J = first + value_T + value_F
    
    return J

def compute_fi(DTR):
    dim = int(math.pow(DTR.shape[0],2))
    D = numpy.zeros((dim,DTR.shape[1]))
    for i in range(DTR.shape[1]):
        D[:,i] = numpy.dot(DTR[:, i].reshape(DTR.shape[0],1),DTR[:,i].T.reshape(1,DTR.shape[0])).reshape((-1, 1), order="F").reshape(dim,)
    
    return numpy.vstack([D, DTR])

def compute_lr_parameters(DTR, LTR, l, prior = None):
    x0 = numpy.zeros(DTR.shape[0]+1)

    if prior == None:
        x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, args=(DTR,LTR,l), approx_grad=1, factr=1)
    else:
        x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, args=(DTR,LTR,l, prior), approx_grad=1, factr=1)
    w = numpy.array(x[0:-1]).reshape(DTR.shape[0],1) 
    b = x[-1]
    return w, b


def compute_scores(DTE, w, b):
    S = numpy.zeros(DTE.shape[1])
    for i in range(DTE.shape[1]):
        S[i] = numpy.dot(w.T,DTE[:,i]) + b
        
    return S

def evaluate_lambda(D, L, n, wp):
    x_axis = numpy.logspace(-6,1,8)
    y_axis = numpy.zeros((len(wp), 8))
    
    for l in range(len(x_axis)):
        S = numpy.array([])
        Lf = numpy.array([], dtype=numpy.int32)
        for i in range(n):
            DTR, LTR, DTE, LTE = cv.k_fold(i, D, L, n)
            w, b = compute_lr_parameters(DTR, LTR, x_axis[l]) 

            S = numpy.append(S, compute_scores(DTE, w, b))
            Lf = numpy.append(Lf, LTE)
        
        for k in range(len(wp)):
            y_axis[k][l] = em.compute_min_DCF(S, Lf, wp[k])
    
    print(y_axis)
    plt.xlabel("lambda", fontsize=15)
    plt.ylabel("minDCF", fontsize=15)
    plt.xscale('log')
    plt.plot(x_axis, y_axis[0,:], linewidth=1, label="minDCF(pi=0.5)", )
    plt.plot(x_axis, y_axis[1,:], linewidth=1, label="minDCF(pi=0.7)")
    plt.plot(x_axis, y_axis[2,:], linewidth=1, label="minDCF(pi=0.1)")
    plt.legend(prop={'size': 15})
    plt.show()

    
def evaluate_prior(D, L, n, l, wp):
    dcf = numpy.zeros((len(wp)+1, len(wp)))
    S = []
    Lf = []
    for i in range(len(wp)+1):
        S.append(numpy.array([]))
        Lf.append(numpy.array([], dtype=numpy.int32))

    for i in range(n):
        DTR, LTR, DTE, LTE = cv.k_fold(i, D, L, n)
        for k in range(len(wp)):
            w, b = compute_lr_parameters(DTR, LTR, l, wp[k]["prior"])
            S[k] = numpy.append(S[k], compute_scores(DTE, w, b))
            Lf[k] = numpy.append(Lf[k], LTE)
        
        w, b = compute_lr_parameters(DTR, LTR, l)
        S[len(wp)] = numpy.append(S[len(wp)], compute_scores(DTE, w, b))
        Lf[len(wp)] = numpy.append(Lf[len(wp)], LTE)

    for k in range(len(wp)):
        for j in range(len(wp)+1):
            dcf[j][k] = em.compute_min_DCF(S[j], Lf[j], wp[k])
        
            
    print("EVALUATE PRIOR")
    print(dcf)


def linear_logistic_regression_model(D, L, n, wp):
    evaluate_lambda(D,L,n,wp)
    l = 10**-4
    evaluate_prior(D,L,n,l,wp)

def quadratic_logistic_regression_model(D, L, n, wp):
    D = compute_fi(D)
    evaluate_lambda(D,L,n,wp)
    l = 10**-6
    evaluate_prior(D,L,n,l,wp)

def linear_logistic_regression_evaluation(DTR, LTR, DTE, LTE, wp, l):   
    x_axis = numpy.logspace(-6,1,8)
    y_axis = numpy.zeros((len(wp), 8))
    
    for j in range(len(x_axis)):      
        w, b = compute_lr_parameters(DTR, LTR, x_axis[j])
        S = compute_scores(DTE, w, b)
        
        for k in range(len(wp)):
            y_axis[k][j] = em.compute_min_DCF(S, LTE, wp[k])
            
    print("EVALUATE LAMBDA")
    print(y_axis)
    plt.xlabel("lambda", fontsize=15)
    plt.ylabel("minDCF", fontsize=15)
    plt.xscale('log')
    plt.plot(x_axis, y_axis[0,:], linewidth=1, label="minDCF(pi=0.5)", )
    plt.plot(x_axis, y_axis[1,:], linewidth=1, label="minDCF(pi=0.7)")
    plt.plot(x_axis, y_axis[2,:], linewidth=1, label="minDCF(pi=0.1)")
    plt.legend(prop={'size': 15})
    plt.show()
    
    dcf = numpy.zeros((len(wp)+1, len(wp)))    
    for j in range(len(wp)):
        w, b = compute_lr_parameters(DTR, LTR, l, wp[j]["prior"])
        S = compute_scores(DTE, w, b)
        for k in range(len(wp)):        
            dcf[j][k] = em.compute_min_DCF(S, LTE, wp[k])
        
    w, b = compute_lr_parameters(DTR, LTR, l)
    S = compute_scores(DTE, w, b)
    for k in range(len(wp)):        
        dcf[len(wp)][k] = em.compute_min_DCF(S, LTE, wp[k])   
        
            
    print("EVALUATE PRIOR")
    print(dcf)


def quadratic_logistic_regression_evaluation(DTR, LTR, DTE, LTE, wp, l):
    DTR = compute_fi(DTR)
    DTE = compute_fi(DTE)
    x_axis = numpy.logspace(-6,1,8)
    y_axis = numpy.zeros((len(wp), 8))
    
    for j in range(len(x_axis)):
        w, b = compute_lr_parameters(DTR, LTR, x_axis[j])
        S = compute_scores(DTE, w, b)
        
        for k in range(len(wp)):
            y_axis[k][j] = em.compute_min_DCF(S, LTE, wp[k])
            
    print("EVALUATE LAMBDA")
    print(y_axis)
    plt.xlabel("lambda", fontsize=15)
    plt.ylabel("minDCF", fontsize=15)
    plt.xscale('log')
    plt.plot(x_axis, y_axis[0,:], linewidth=1, label="minDCF(pi=0.5)", )
    plt.plot(x_axis, y_axis[1,:], linewidth=1, label="minDCF(pi=0.7)")
    plt.plot(x_axis, y_axis[2,:], linewidth=1, label="minDCF(pi=0.1)")
    plt.legend(prop={'size': 15})
    plt.show()
    
    dcf = numpy.zeros((len(wp)+1, len(wp)))
    
    for j in range(len(wp)):
        w, b = compute_lr_parameters(DTR, LTR, l, wp[j]["prior"])
        S = compute_scores(DTE, w, b)
        for k in range(len(wp)):        
            dcf[j][k] = em.compute_min_DCF(S, LTE, wp[k])
        
    w, b = compute_lr_parameters(DTR, LTR, l)
    S = compute_scores(DTE, w, b)
    for k in range(len(wp)):        
        dcf[len(wp)][k] = em.compute_min_DCF(S, LTE, wp[k])  
            
    print("EVALUATE PRIOR")
    print(dcf)
    

        