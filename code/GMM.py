# -*- coding: utf-8 -*-
"""
Created on Fri May 28 21:12:44 2021

@author: LucaS
"""
import numpy, scipy.special
import crossValidation as cv
import evaluationModel as em
import matplotlib.pyplot as plt


def logpdf_GMM(X, gmm):
    S = numpy.zeros((len(gmm), X.shape[1]))
    for i in range(len(gmm)):       
        S[i, :] = GAU_logpdf_ND(X, gmm[i][1], gmm[i][2])
            
    for g in range(len(gmm)):
        S[g,:] += numpy.log(gmm[g][0])
    return S, scipy.special.logsumexp(S,axis=0)
    

def GAU_logpdf_ND(x, mu, C):
    _, det = numpy.linalg.slogdet(C)
    return numpy.diag(-0.5*x.shape[0]*numpy.log(2*numpy.pi) - 0.5*det - 0.5*numpy.dot((x-mu).T, numpy.dot(numpy.linalg.inv(C), x-mu)))

def mcol(v):
    return v.reshape((v.size, 1))

def compute_GMM(X, gmm, psi):
    scores, prev = logpdf_GMM(X,gmm)
    while 1:            
        responsibilities = numpy.exp(scores-prev)
            
        Z = numpy.sum(responsibilities, axis=1)
        F = numpy.zeros((len(gmm), X.shape[0]))
        for g in range(len(gmm)):
            for i in range(X.shape[1]):
                F[g] += responsibilities[g][i]*X[:,i]
        
        S = numpy.zeros((len(gmm), X.shape[0], X.shape[0]))
        for g in range(len(gmm)):
            for i in range(X.shape[1]):
                S[g] += responsibilities[g][i]*(X[:,i:i+1].dot(X[:,i:i+1].T))
        for i in range(len(gmm)):
            gmm[i] = (Z[i]/numpy.sum(Z), 
                      ((F[i]/Z[i]).reshape((X.shape[0],1))), 
                      S[i]/Z[i] - ((F[i]/Z[i]).reshape((X.shape[0],1))).dot(((F[i]/Z[i]).reshape((X.shape[0],1))).T))
            
            U, s, _ = numpy.linalg.svd(gmm[i][2])
            s[s<psi] = psi
            
            gmm[i] = (gmm[i][0], gmm[i][1], numpy.dot(U, mcol(s)*U.T))
            
        scores, actual = logpdf_GMM(X,gmm)
        if (actual.mean() - prev.mean()) < 10**-6:
            break
        prev = actual   
    return gmm

def compute_GMM_diag(X, gmm, psi):
    scores, prev = logpdf_GMM(X,gmm)
    
    while 1:        
        responsibilities = numpy.exp(scores-prev)
            
        Z = numpy.sum(responsibilities, axis=1)
        F = numpy.zeros((len(gmm), X.shape[0]))
        for g in range(len(gmm)):
            for i in range(X.shape[1]):
                F[g] += responsibilities[g][i]*X[:,i]
        
        S = numpy.zeros((len(gmm), X.shape[0], X.shape[0]))
        for g in range(len(gmm)):
            for i in range(X.shape[1]):
                S[g] += responsibilities[g][i]*(X[:,i:i+1].dot(X[:,i:i+1].T))
        for i in range(len(gmm)):
            gmm[i] = (Z[i]/numpy.sum(Z), 
                      ((F[i]/Z[i]).reshape((X.shape[0],1))), 
                      (S[i]/Z[i] - ((F[i]/Z[i]).reshape((X.shape[0],1))).dot(((F[i]/Z[i]).reshape((X.shape[0],1))).T))*numpy.eye(gmm[i][2].shape[0]))   
            U, s, _ = numpy.linalg.svd(gmm[i][2])
            s[s<psi] = psi
            gmm[i] = (gmm[i][0], gmm[i][1], numpy.dot(U, mcol(s)*U.T))
        
        scores, actual = logpdf_GMM(X,gmm)
        if (actual.mean() - prev.mean()) < 10**-6:
            break
        prev = actual
    return gmm

def split_GMM(GMM):
    alpha = 0.1
    tmp=[]
    for j in range(len(GMM)):
        U, s, _ = numpy.linalg.svd(GMM[j][2])
        d = U[:, 0:1] * s[0]**0.5 * alpha
        tmp.append((GMM[j][0]/2, GMM[j][1] + d, GMM[j][2]))
        tmp.append((GMM[j][0]/2, GMM[j][1] - d, GMM[j][2]))
    
    return tmp

def compute_GMM_tied(X, gmm, psi):
    scores, prev = logpdf_GMM(X,gmm)
    while 1:  
        responsibilities = numpy.exp(scores-prev)
            
        Z = numpy.sum(responsibilities, axis=1)
        F = numpy.zeros((len(gmm), X.shape[0]))
        for g in range(len(gmm)):
            for i in range(X.shape[1]):
                F[g] += responsibilities[g][i]*X[:,i]
        
        S = numpy.zeros((len(gmm), X.shape[0], X.shape[0]))
        for g in range(len(gmm)):
            for i in range(X.shape[1]):
                S[g] += responsibilities[g][i]*(X[:,i:i+1].dot(X[:,i:i+1].T))
        for i in range(len(gmm)):
            gmm[i] = (Z[i]/numpy.sum(Z), 
                      ((F[i]/Z[i]).reshape((X.shape[0],1))), 
                      (S[i]/Z[i] - ((F[i]/Z[i]).reshape((X.shape[0],1))).dot(((F[i]/Z[i]).reshape((X.shape[0],1))).T)))   
        diag = numpy.zeros((X.shape[0], X.shape[0]))
        for i in range(len(gmm)):
            diag += Z[i]*gmm[i][2]
            
        diag = diag/X.shape[1]
        U, s, _ = numpy.linalg.svd(diag)
        s[s<psi] = psi
        for i in range(len(gmm)):            
            gmm[i] = (gmm[i][0], gmm[i][1], numpy.dot(U, mcol(s)*U.T))
            
        scores, actual = logpdf_GMM(X,gmm)
        if (actual.mean() - prev.mean()) < 10**-6:
            break
        prev = actual
    return gmm
       
def compute_scores(DTE, GMM):
    prob = numpy.zeros((len(GMM), DTE.shape[1]));
    for c in range(len(GMM)):
        _, prob[c,:] = logpdf_GMM(DTE, GMM[c])
    return prob[1,:]-prob[0,:]

def compute_initialGMM(DTR, LTR, psi):

    D0 = DTR[:,LTR==0]
    initialGMM0 = (1, D0.mean(axis=1).reshape(D0.shape[0], 1), numpy.cov(D0, ddof=0))
    
    D1 = DTR[:,LTR==1]
    initialGMM1 = (1, D1.mean(axis=1).reshape(D1.shape[0], 1), numpy.cov(D1, ddof=0))
    
    return [[initialGMM0], [initialGMM1]]
    
   
def gmm_model(D, L, n, wp):
    dim = 3
    K = [2, 4, 8, 16, 32, 64, 128]
    dcf = numpy.zeros((dim, len(K)))
    S = [[],[],[]]
    Lf = numpy.array([], dtype=numpy.int32)
    for i in range(dim):
        for j in range(len(K)):
            S[i].insert(j,numpy.array([]))

    psi = 0.01    
        
    for j in range(n):
        DTR, LTR, DTE, LTE = cv.k_fold(j, D, L, n)
        GMM = compute_initialGMM(DTR, LTR, psi)
        GMM_diag = compute_initialGMM(DTR, LTR, psi)
        GMM_tied = compute_initialGMM(DTR, LTR, psi)
        Lf = numpy.append(Lf, LTE)
        for i in range(2):
            GMM[i] = compute_GMM(DTR[:,LTR==i], GMM[i], psi)
            GMM_diag[i] = compute_GMM_diag(DTR[:,LTR==i], GMM_diag[i], psi)
            GMM_tied[i] = compute_GMM_tied(DTR[:,LTR==i], GMM_tied[i], psi)
        
        for components in range(len(K)):          
            for i in range(2):
                GMM[i] = split_GMM(GMM[i]) 
                GMM_diag[i] = split_GMM(GMM_diag[i])
                GMM_tied[i] = split_GMM(GMM_tied[i])
                GMM[i] = compute_GMM(DTR[:,LTR==i], GMM[i], psi)                
                GMM_diag[i] = compute_GMM_diag(DTR[:,LTR==i], GMM_diag[i], psi)
                GMM_tied[i] = compute_GMM_tied(DTR[:,LTR==i], GMM_tied[i], psi)

            S[0][components] = numpy.append(S[0][components], compute_scores(DTE, GMM))
            S[1][components] = numpy.append(S[1][components], compute_scores(DTE, GMM_diag))
            S[2][components] = numpy.append(S[2][components], compute_scores(DTE, GMM_tied))
          
    for k in range(dim):
        for j in range(len(K)):
            dcf[k][j] = em.compute_min_DCF(S[k][j], Lf, wp)
            
    print(dcf)
    
    labels = ['2', '4', '8', '16', '32', '64', '128']
    x = numpy.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width , dcf[0, :], width, label='full cov')
    ax.bar(x + width, dcf[1,:], width, label='diag cov')
    ax.bar(x , dcf[2, :], width, label='tied cov')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('minDCF', fontsize=15)
    ax.set_xlabel('components', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15)
    ax.legend(prop={'size': 15})
    fig.tight_layout()
    plt.show()

def gmm_evaluation(DTR, LTR, DTE, LTE, wp):
    dim = 3
    K = [2, 4, 8, 16, 32, 64, 128]
    dcf = numpy.zeros((dim, len(K)))   

    psi = 0.01    
        
    GMM = compute_initialGMM(DTR, LTR, psi)
    GMM_diag = compute_initialGMM(DTR, LTR, psi)
    GMM_tied = compute_initialGMM(DTR, LTR, psi) 
    for i in range(max(LTR)+1):
        GMM[i] = compute_GMM(DTR[:,LTR==i], GMM[i], psi)
        GMM_diag[i] = compute_GMM_diag(DTR[:,LTR==i], GMM_diag[i], psi)
        GMM_tied[i] = compute_GMM_tied(DTR[:,LTR==i], GMM_tied[i], psi)
    
    for components in range(len(K)):          
        for i in range(2):
            GMM[i] = split_GMM(GMM[i])                
            GMM_diag[i] = split_GMM(GMM_diag[i])
            GMM_tied[i] = split_GMM(GMM_tied[i])
            GMM[i] = compute_GMM(DTR[:,LTR==i], GMM[i], psi)                
            GMM_diag[i] = compute_GMM_diag(DTR[:,LTR==i], GMM_diag[i], psi)
            GMM_tied[i] = compute_GMM_tied(DTR[:,LTR==i], GMM_tied[i], psi)
            

        scores = compute_scores(DTE, GMM)
        scores_diag = compute_scores(DTE, GMM_diag)
        scores_tied = compute_scores(DTE, GMM_tied)
        dcf[0][components] = em.compute_min_DCF(scores, LTE, wp)
        dcf[1][components] = em.compute_min_DCF(scores_diag, LTE, wp)
        dcf[2][components] = em.compute_min_DCF(scores_tied, LTE, wp)
            
    print(dcf)
    
    labels = ['2', '4', '8', '16', '32', '64', '128']
    x = numpy.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width , dcf[0, :], width, label='full cov')
    ax.bar(x + width, dcf[1,:], width, label='diag cov')
    ax.bar(x , dcf[2, :], width, label='tied cov')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('minDCF', fontsize=15)
    ax.set_xlabel('components', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15)
    ax.legend(prop={'size': 15})
    fig.tight_layout()
    plt.show()