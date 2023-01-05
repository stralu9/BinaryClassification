# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 16:56:52 2021

@author: LucaS
"""
import numpy
import matplotlib.pyplot as plt
import crossValidation as cv
import scipy
import evaluationModel as em

def svm(H):
    def f(a):
        a = a.reshape((len(a),1))
        gradient = (numpy.dot(H, a) - numpy.ones((H.shape[1],1))).reshape((H.shape[0],))
        return 0.5*numpy.dot(a.T,numpy.dot(H, a)) - numpy.dot(a.T,numpy.ones((H.shape[0],1))),gradient
    return f   


def evaluate_C(D, L, n, wp):
    x_axis = numpy.logspace(-3,3,7)
    y_axis = numpy.zeros((len(wp),7))
    
    for l in range(len(x_axis)):
        S = numpy.array([])
        Lf = numpy.array([], dtype=numpy.int32)
        for i in range(n):
            DTR, LTR, DTE, LTE = cv.k_fold(i, D, L, n)
                
            w = compute_svm_parameters(DTR, LTR, x_axis[l]) 
            S = numpy.append(S, compute_scores(DTE, w))
            Lf = numpy.append(Lf, LTE)
        
        for k in range(len(wp)):
            y_axis[k][l] = em.compute_min_DCF(S, Lf, wp[k])
    
    print(y_axis)
    plt.xlabel("C", fontsize=15)
    plt.ylabel("minDCF", fontsize=15)
    plt.xscale('log')
    plt.plot(x_axis, y_axis[0,:], linewidth=1, label="minDCF(pi=0.5)")
    plt.plot(x_axis, y_axis[1,:], linewidth=1, label="minDCF(pi=0.7)")
    plt.plot(x_axis, y_axis[2,:], linewidth=1, label="minDCF(pi=0.1)")
    plt.legend(prop={'size': 15})
    plt.show()

def evaluate_C_c_pk(D, L, n, wp):
    x_axis = numpy.logspace(-2,2,5)
    c = [0, 0.1, 1]
    y_axis = numpy.zeros((len(c),len(x_axis)))
    for l in range(len(x_axis)):
        for k in range(len(c)):
            S = numpy.array([])
            Lf = numpy.array([], dtype=numpy.int32)
            for i in range(n):
                DTR, LTR, DTE, LTE = cv.k_fold(i, D, L, n)
                w = compute_svm_kf_parameters(DTR, LTR, x_axis[l], c[k], True) 
                S = numpy.append(S, compute_scores_kf(DTR, LTR, DTE, w, c[k], True))
                Lf = numpy.append(Lf, LTE)
            
            y_axis[k][l] = em.compute_min_DCF(S, Lf, wp)
    print(y_axis)          
    plt.xlabel("C", fontsize=15)
    plt.ylabel("minDCF", fontsize=15)
    plt.xscale('log')
    plt.plot(x_axis, y_axis[0,:], linewidth=1, label="c=0")
    plt.plot(x_axis, y_axis[1,:], linewidth=1, label="c=0.1")
    plt.plot(x_axis, y_axis[2,:], linewidth=1, label="c=1")
    plt.legend(prop={'size': 15})
    plt.show()
    
def evaluate_C_psi_rbfk(D, L, n, wp):
    x_axis = numpy.array((0.01, 0.1, 1, 5, 10, 15, 20, 100))
    psi = [0.1, 0.01, 0.001]
    y_axis = numpy.zeros((len(psi),len(x_axis)))
    
    for l in range(len(x_axis)):
        for k in range(len(psi)):
            S = numpy.array([])
            Lf = numpy.array([], dtype=numpy.int32)
            for i in range(n):
                DTR, LTR, DTE, LTE = cv.k_fold(i, D, L, n)
                    
                w = compute_svm_kf_parameters(DTR, LTR, x_axis[l], psi[k], False) 
                S = numpy.append(S, compute_scores_kf(DTR, LTR, DTE, w, psi[k], False))
                Lf = numpy.append(Lf, LTE) 
            y_axis[k][l] = em.compute_min_DCF(S, Lf, wp)
    
    print(y_axis)      
    plt.xlabel("C", fontsize=15)
    plt.ylabel("minDCF", fontsize=15)
    plt.xscale('log')
    plt.plot(x_axis, y_axis[0,:], linewidth=1, label="log(γ)=-3")
    plt.plot(x_axis, y_axis[1,:], linewidth=1, label="log(γ)=-2")
    plt.plot(x_axis, y_axis[2,:], linewidth=1, label="log(γ)=-1")
    plt.legend(prop={'size': 15})
    plt.show()
    
def linear_SVM_model(D, L, wp, n):
    print("EVALUATE C")
    evaluate_C(D,L,n,wp)
    C=0.01
    print("EVALUATE PRIOR")
    evaluate_prior_linear(D, L, n, wp, C)
    
  
def compute_svm_parameters(DTR, LTR, C, prior=None):
    z = 2*LTR - 1
    
    DTR_extended = numpy.vstack([DTR,numpy.ones(DTR.shape[1])])
    
    #compute H
    H = numpy.dot((z*DTR_extended).T, z*DTR_extended)

    alpha = numpy.zeros((DTR.shape[1],1))
    
    mf = numpy.ones(H.shape[0])
    if prior != None :
        pi_emp = LTR.sum() / LTR.size
        mf[LTR == 1] = prior / pi_emp
        mf[LTR == 0] = (1-prior) / (1-pi_emp)
          
    bound = [(0,C) for i in range(H.shape[0])]
    bound = bound*mf.reshape((len(mf),1))
    x, f, d = scipy.optimize.fmin_l_bfgs_b(svm(H), alpha, bounds=bound, factr=1)
    
    return (x*z*DTR_extended).sum(axis=1).reshape((DTR.shape[0]+1,1))

def compute_scores(DTE,w):
    DTE_extended = numpy.vstack([DTE,numpy.ones(DTE.shape[1])])
    s = numpy.dot(w.T,DTE_extended)
    s = s.reshape((s.size,))
    return s


def kf_polynomial(x, y, c):
    return ((numpy.dot(x.T, y) +c)**2)

def kf_rbf(x,y,l):
    return numpy.exp(-l*(numpy.linalg.norm(x-y))**2)

def compute_svm_kf_parameters(DTR,LTR, C, c, flag, prior=None):
    z = (2*LTR - 1).reshape((len(LTR), 1)) 
    H = numpy.zeros((DTR.shape[1], DTR.shape[1]))
    if flag:                
        H = numpy.dot(z, z.T)*kf_polynomial(DTR, DTR, c)
    else:
        for i in range(DTR.shape[1]):
            for j in range(DTR.shape[1]):
                H[i][j] = z[i]*z[j]*kf_rbf(DTR[:,i], DTR[:,j], c)
                
    
    #minimize L
    alpha = numpy.zeros((DTR.shape[1],1))
    
    mf = numpy.ones(H.shape[0]).reshape((H.shape[0],1))
    if prior != None :
        pi_emp = LTR.sum() / LTR.size
        mf[LTR == 1] = prior / pi_emp
        mf[LTR == 0] = (1-prior) / (1-pi_emp)
            
    bound = [(0,C) for i in range(H.shape[0])]
    bound = bound*mf.reshape((len(mf),1))
    x, f, d = scipy.optimize.fmin_l_bfgs_b(svm(H), alpha, bounds=bound, factr=1)
    
    return x


def compute_scores_kf(DTR, LTR, DTE, x, c, flag):
    z = 2*LTR - 1  
    s = numpy.zeros((DTE.shape[1]))
    if flag:
        for j in range(DTE.shape[1]):
            for i in range(DTR.shape[1]):     
                s[j] += x[i]*z[i]*kf_polynomial(DTR[:,i], DTE[:,j],c)
    else:
        for j in range(DTE.shape[1]):
            for i in range(DTR.shape[1]):
                s[j] += x[i]*z[i]*kf_rbf(DTR[:,i], DTE[:,j], c)             
    return s

def evaluate_prior_linear(D, L, n, wp, C):
    dcf = numpy.zeros((len(wp)+1,len(wp)))
    
    for l in range(len(wp)+1):
        S = numpy.array([])
        Lf = numpy.array([], dtype=numpy.int32)
        for i in range(n):
            DTR, LTR, DTE, LTE = cv.k_fold(i, D, L, n)
            if l==(len(wp)):
                w = compute_svm_parameters(DTR, LTR, C) 
                S = numpy.append(S, compute_scores(DTE, w))
                Lf = numpy.append(Lf, LTE) 
            else:
                w = compute_svm_parameters(DTR, LTR, C, wp[l]["prior"]) 
                S = numpy.append(S, compute_scores(DTE, w))
                Lf = numpy.append(Lf, LTE)          
        
        for k in range(len(wp)):
            dcf[l][k] = em.compute_min_DCF(S, Lf, wp[k])
    print(dcf)
            
def evaluate_prior_kernel(D, L, n, wp, C, c, flag):
    dcf = numpy.zeros((len(wp)+1,len(wp)))
    
    for l in range(len(wp)+1):
        S = numpy.array([])
        Lf = numpy.array([], dtype=numpy.int32)
        for i in range(n):
            DTR, LTR, DTE, LTE = cv.k_fold(i, D, L, n)
            if l==(len(wp)):
                w = compute_svm_kf_parameters(DTR, LTR, C, c, flag) 
                S = numpy.append(S, compute_scores_kf(DTR, LTR, DTE, w, c, flag))
                Lf = numpy.append(Lf, LTE) 
            else:
                w = compute_svm_kf_parameters(DTR, LTR, C, c, flag, wp[l]["prior"]) 
                S = numpy.append(S, compute_scores_kf(DTR, LTR, DTE, w, c, flag))
                Lf = numpy.append(Lf, LTE)          
        
        for k in range(len(wp)):
            dcf[l][k] = em.compute_min_DCF(S, Lf, wp[k])
            
    print(dcf)
    
def kernel_svm_models(D, L, wp, n):
    polynomial_kernel_svm_model(D,L,n,wp)
    rbf_svm_model(D,L,n,wp)
    
def polynomial_kernel_svm_model(D,L,n,wp):
    print("EVALUATE C and psi")
    evaluate_C_c_pk(D, L, n, wp[0])
    C=0.1
    c=100
    print("EVALUATE prior")
    evaluate_prior_kernel(D,L,n,wp, C, c, True)
    
def rbf_svm_model(D,L,n,wp):
    print("EVALUATE C and psi")
    evaluate_C_psi_rbfk(D, L, n, wp[0])
    C=10
    psi = 0.01
    print("EVALUATE prior")
    evaluate_prior_kernel(D, L,n, wp, C, psi, False)
    
def linear_svm_evaluation(DTR, LTR, DTE, LTE, wp, C):
    x_axis = numpy.logspace(-3,3,7)
    y_axis = numpy.zeros((len(wp),7))
    
    for l in range(len(x_axis)):                
        w = compute_svm_parameters(DTR, LTR, x_axis[l]) 
        S = compute_scores(DTE, w)
        
        for k in range(len(wp)):
            y_axis[k][l] = em.compute_min_DCF(S, LTE, wp[k])
    
    print(y_axis)
    plt.xlabel("C", fontsize=15)
    plt.ylabel("minDCF", fontsize=15)
    plt.xscale('log')
    plt.plot(x_axis, y_axis[0,:], linewidth=1, label="minDCF(pi=0.5)")
    plt.plot(x_axis, y_axis[1,:], linewidth=1, label="minDCF(pi=0.7)")
    plt.plot(x_axis, y_axis[2,:], linewidth=1, label="minDCF(pi=0.1)")
    plt.legend(prop={'size': 15})
    plt.show()
    dcf = numpy.zeros((len(wp)+1,len(wp)))
    
    for l in range(len(wp)+1):
        
        if l==(len(wp)):
            w = compute_svm_parameters(DTR, LTR, C) 
            S = compute_scores(DTE, w)
        else:
            w = compute_svm_parameters(DTR, LTR, C, wp[l]["prior"]) 
            S = compute_scores(DTE, w)          
        
        for k in range(len(wp)):
            dcf[l][k] = em.compute_min_DCF(S, LTE, wp[k])
            
    print(dcf)

def kernel_svm_evaluation(DTR,LTR,DTE,LTE,wp, C_q, c, C_rbf, psi):
    polynomial_kernel_svm_evaluation(DTR,LTR,DTE,LTE,wp,C_q,c)
    rbf_kernel_svm_evaluation(DTR,LTR,DTE,LTE,wp,C_rbf,psi)

def polynomial_kernel_svm_evaluation(DTR,LTR,DTE,LTE,wp,C,psi):
    x_axis = numpy.logspace(-2,2,5)
    c = [0, 0.1, 1]
    y_axis = numpy.zeros((len(c),len(x_axis)))
    for l in range(len(x_axis)):
        for k in range(len(c)):            
            w = compute_svm_kf_parameters(DTR, LTR, x_axis[l], c[k], True) 
            S = compute_scores_kf(DTR, LTR, DTE, w, c[k], True)
            
            y_axis[k][l] = em.compute_min_DCF(S, LTE, wp[0])
            
    print(y_axis)          
    plt.xlabel("C", fontsize=15)
    plt.ylabel("minDCF", fontsize=15)
    plt.xscale('log')
    plt.plot(x_axis, y_axis[0,:], linewidth=1, label="c=0")
    plt.plot(x_axis, y_axis[1,:], linewidth=1, label="c=0.1")
    plt.plot(x_axis, y_axis[2,:], linewidth=1, label="c=1")
    plt.legend(prop={'size': 15})
    plt.show()
    
    dcf = numpy.zeros((len(wp)+1,len(wp)))
    
    for l in range(len(wp)+1):
        
        if l==(len(wp)):
            w = compute_svm_kf_parameters(DTR, LTR, C, psi, True) 
            S = compute_scores_kf(DTR, LTR, DTE, w, psi, True)
        else:
            w = compute_svm_kf_parameters(DTR, LTR, C, psi, True, wp[l]["prior"]) 
            S = compute_scores_kf(DTR, LTR, DTE, w, psi, True)         
        
        for k in range(len(wp)):
            dcf[l][k] = em.compute_min_DCF(S, LTE, wp[k])
            
    print(dcf)

def rbf_kernel_svm_evaluation(DTR,LTR,DTE,LTE,wp,C,psi):
    x_axis = numpy.array((0.01, 0.1, 1, 5, 10, 15, 20, 100))
    c = [0.1, 0.01, 0.001]
    y_axis = numpy.zeros((len(c),len(x_axis)))
    
    for l in range(len(x_axis)):
        for k in range(len(c)):
        
            w = compute_svm_kf_parameters(DTR, LTR, x_axis[l], c[k], False) 
            S = compute_scores_kf(DTR, LTR, DTE, w, c[k], False)
                
            y_axis[k][l] = em.compute_min_DCF(S, LTE, wp[0])
    
    print(y_axis)      
    plt.xlabel("C", fontsize=15)
    plt.ylabel("minDCF", fontsize=15)
    plt.xscale('log')
    plt.plot(x_axis, y_axis[0,:], linewidth=1, label="log(γ)=-3")
    plt.plot(x_axis, y_axis[1,:], linewidth=1, label="log(γ)=-2")
    plt.plot(x_axis, y_axis[2,:], linewidth=1, label="log(γ)=-1")
    plt.legend(prop={'size': 15})
    plt.show()
    
    dcf = numpy.zeros((len(wp)+1,len(wp)))
    
    for l in range(len(wp)+1):        
        if l==(len(wp)):
            w = compute_svm_kf_parameters(DTR, LTR, C, psi, False) 
            S = compute_scores_kf(DTR, LTR, DTE, w, psi, False)
        else:
            w = compute_svm_kf_parameters(DTR, LTR, C, psi, False, wp[l]["prior"]) 
            S = compute_scores_kf(DTR, LTR, DTE, w, psi, False)         
        
        for k in range(len(wp)):
            dcf[l][k] = em.compute_min_DCF(S, LTE, wp[k])
            
    print(dcf)
    
def assess_quality_of_scoresRBF(D,L,wp,n):
    C = 10
    psi = 0.01
    prior = 0.5
    minDCF = numpy.zeros(len(wp))
    actDCF = numpy.zeros(len(wp))
    
    S = numpy.array([])
    Lf = numpy.array([], dtype=numpy.int32)
    for i in range(n):
        DTR, LTR, DTE, LTE = cv.k_fold(i, D, L, n)
                    
        w = compute_svm_kf_parameters(DTR, LTR, C, psi, False, prior) 
        S = numpy.append(S, compute_scores_kf(DTR, LTR, DTE, w, psi, False))
        Lf = numpy.append(Lf, LTE)
    
    for i in range(len(wp)):
        minDCF[i] = em.compute_min_DCF(S, Lf, wp[i])
        actDCF[i] = em.compute_act_DCF(S, Lf, wp[i])
    
    return minDCF, actDCF, S

def assess_quality_of_scoresRBF_eval(DTR,LTR, DTE, LTE, wp, C, psi, prior):
    minDCF = numpy.zeros(len(wp))
    actDCF = numpy.zeros(len(wp))
                    
    w = compute_svm_kf_parameters(DTR, LTR, C, psi, False, prior) 
    S = compute_scores_kf(DTR, LTR, DTE, w, psi, False)

    for i in range(len(wp)):
        minDCF[i] = em.compute_min_DCF(S, LTE, wp[i])
        actDCF[i] = em.compute_act_DCF(S, LTE, wp[i])
    
    return minDCF, actDCF, S
    