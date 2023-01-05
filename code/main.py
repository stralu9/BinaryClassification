# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 09:36:45 2021

@author: LucaS
"""

import numpy
import seaborn as sns;
from matplotlib import pyplot as plt
import GaussianClassifiers as gc
import evaluationModel as em
import PCA as pca
import logisticRegression as lr
import SVM as svm
import GMM as gmm
from scipy.stats import norm

def load(file):
    quality_wine = []
    vett = []
    with open(file) as f:
        for line in f:
            try:
                v = line.split(",")[0:11]
                v = numpy.array([float(i) for i in v])
                v = v.reshape(v.size, 1)
                vett.append(v)
                quality_wine.append(int(line.split(",")[-1].strip()))
            except:

                pass
    quality_wine = numpy.array(quality_wine)
    vett = numpy.hstack(vett)
    return vett, quality_wine

def plot_hist(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]

    features = {
        0: 'fixed acidity',
        1: 'volatile acidity',
        2: 'citric acid',
        3: 'residual sugar',
        4: 'chlorides',
        5: 'free sulfur dioxide',
        6: 'total sulfur dioxide',
        7: 'density',
        8: 'pH',
        9: 'sulphates',
        10: 'alcohol',
    }
    for graf_n in range(D.shape[0]):
        plt.figure()
        plt.xlabel(features[graf_n], fontsize=15)
        plt.hist(D0[graf_n, :], bins=50, density=True, alpha=0.5, label="GOOD QUALITY")
        plt.hist(D1[graf_n, :], bins=50, density=True, alpha=0.5, label="BAD QUALITY")


        plt.legend(prop={'size': 15})
        plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
        plt.savefig('../hist_%d.pdf' % graf_n)

    plt.show()
def preliminary_analysis(D,L):
    heat_map(D, L)
    plot_hist(D, L)

def heat_map(D,L):
    cov_matrix = numpy.corrcoef(D)
    cov_matrix0 =  numpy.corrcoef(D[:, L==0]) #bad_quality
    cov_matrix1 = numpy.corrcoef(D[:, L==1]) #good_quality
    
    plt.title("General Pearson coefficient", fontsize=15)
    sns.heatmap(cov_matrix, vmin=-1, vmax=1)
    plt.figure()
    plt.title("Class0 Pearson coefficient", fontsize=15)
    sns.heatmap(cov_matrix0, vmin=-1, vmax=1)
    plt.figure()
    plt.title("Class1 Pearson coefficient", fontsize=15)
    sns.heatmap(cov_matrix1, vmin=-1, vmax=1)
    
def Gaussianization(DTR, DTE=numpy.array([])):
    if DTE.shape[0] == 0:
        D = numpy.zeros((DTR.shape[0], DTR.shape[1]))
        for i in range(DTR.shape[0]):
            for j in range(DTR.shape[1]):
                selected = numpy.array(DTR[i, :] > DTR[i][j], dtype=int)
                D[i][j] += selected.sum()
        D = (D + 1) / (DTR.shape[1] + 2)
        return norm.ppf(D)
    else:
        D = numpy.zeros((DTE.shape[0], DTE.shape[1]))
        for i in range(DTR.shape[0]):
            for j in range(DTE.shape[1]):
                selected = numpy.array(DTR[i, :] > DTE[i][j], dtype=int)
                D[i][j] += selected.sum()
        D = (D + 1) / (DTR.shape[1] + 2)
        return norm.ppf(D)

def Gaussian_models(D,L, working_points,nfolds):

    #WITHOUT_PCA
    print("WITHOUT PCA")
    gc.full_cov_model(D, L, nfolds, working_points, D.shape[0])
    gc.naive_cov_model(D, L, nfolds, working_points, D.shape[0])
    gc.tied_cov_model(D, L, nfolds, working_points, D.shape[0])
    gc.tied_naive_cov_model(D, L, nfolds, working_points, D.shape[0])   

    for dim in range(9,D.shape[0]):
        print("WITH PCA m="+dim.__str__())
        gc.full_cov_model(D, L, nfolds, working_points, dim)
        gc.naive_cov_model(D, L, nfolds, working_points, dim)
        gc.tied_cov_model(D, L, nfolds, working_points, dim)
        gc.tied_naive_cov_model(D, L, nfolds, working_points, dim)
        
def logistic_regression_models(D,L, wp, n):
    print("LOGISTIC REGRESSION")
    print("LINEAR MODEL")
    linear_logistic_regression_model(D, L, wp, n)
    print("QUADRATIC MODEL")
    quadratic_logistic_regression_model(D, L, wp, n)
    

def linear_logistic_regression_model(D, L, working_points, nfolds):   
    lr.linear_logistic_regression_model(D, L, nfolds, working_points)


def quadratic_logistic_regression_model(D, L, working_points, nfolds):   
    lr.quadratic_logistic_regression_model(D, L, nfolds, working_points)
 
    
def logistic_regression_evaluation(DTR,LTR,DTE,LTE, wp, l1, l2):    
    print("LINEAR REGRESSION")
    lr.linear_logistic_regression_evaluation(DTR,LTR,DTE,LTE,wp,l1)
    print("QUADRATIC REGRESSION")
    lr.quadratic_logistic_regression_evaluation(DTR,LTR,DTE,LTE,wp,l2)
    
def Gaussian_evaluation(DTR, LTR, DTE, LTE, working_points):
    print("WITHOUT PCA")
    gc.full_cov_evaluation(DTR, LTR, DTE, LTE, working_points)
    gc.naive_cov_evaluation(DTR, LTR, DTE, LTE, working_points)
    gc.tied_cov_evaluation(DTR, LTR, DTE, LTE, working_points)
    gc.tied_naive_cov_evaluation(DTR, LTR, DTE, LTE, working_points)  
    
    print("WITH_PCA")
    for dim in range(9,DTR.shape[0]):
        print("WITH PCA m="+dim.__str__())
        DTR, DTE = pca.apply_pca_training_and_test(DTR, DTE, dim)
        gc.full_cov_evaluation(DTR, LTR, DTE, LTE, working_points)
        gc.naive_cov_evaluation(DTR, LTR, DTE, LTE, working_points)
        gc.tied_cov_evaluation(DTR, LTR, DTE, LTE, working_points)
        gc.tied_naive_cov_evaluation(DTR, LTR, DTE, LTE, working_points)

def svm_evaluation(DTR, LTR, DTE, LTE, wp, C, C_q, c, C_rbf, psi):
    svm.linear_svm_evaluation(DTR, LTR, DTE, LTE, wp, C)
    svm.kernel_svm_evaluation(DTR, LTR, DTE, LTE, wp, C_q, c, C_rbf, psi)

def svm_models(D, L, wp, nfolds):
    print("LINEAR SVM")
    svm.linear_SVM_model(D, L, wp, nfolds)
    print("KERNEL SVM")
    svm.kernel_svm_models(D, L, wp, nfolds)
    
def GMM_models(D, L, wp, nfolds):
    gmm.gmm_model(D, L, nfolds, wp[0])

def GMM_evaluation(D,L,DE,LE,wp):
    gmm.gmm_evaluation(D,L,DE,LE,wp[0])
    
def score_quality_validation(D, L, wp, n):
    minDCF_svmrbf, actDCF_svmrbf, scores_svmrbf = svm.assess_quality_of_scoresRBF(D, L, working_points, nfolds)
    
    print(minDCF_svmrbf)
    print(actDCF_svmrbf)
    
    em.plot_bayes_error(scores_svmrbf, L)

def score_quality_evaluation(D, L, DE, LE, wp, C, psi, prior):
    minDCF, actDCF, scores_svmrbf = svm.assess_quality_of_scoresRBF_eval(D, L, DE, LE, working_points, C, psi, prior)
    print(minDCF)
    print(actDCF)
    em.plot_bayes_error(scores_svmrbf, LE)
    
if __name__ == '__main__':
    
    D, L = load("train.txt")
    
    DE, LE = load("test.txt")
    
    plt.rc('xtick', labelsize=15) 
    plt.rc('ytick', labelsize=15) 
    working_points = [{"prior":0.5, "Cfn": 1, "Cfp":1}, {"prior":0.7, "Cfn": 1, "Cfp":1},
              {"prior":0.1, "Cfn": 1, "Cfp":1}]
    nfolds = 5
    #preliminary_analysis(D,L)
    
    print("MODEL")
    #Gaussian_models(D,L, working_points, nfolds)
    logistic_regression_models(D, L, working_points, nfolds)
    #svm_models(D, L, working_points, nfolds)
    #GMM_models(D, L, working_points, nfolds)

    
    DG = Gaussianization(D)
    DGE = Gaussianization(D, DE)
    print("MODEL WITH GAUSSIANIZATION")
    #Gaussian_models(DG,L, working_points, nfolds)
    logistic_regression_models(DG, L, working_points, nfolds)
    svm_models(DG, L, working_points, nfolds)
    GMM_models(DG, L, working_points, nfolds)
    
    
    score_quality_validation(DG, L, working_points, nfolds)    
    
    
    print("EVALUATION RAW")
    Gaussian_evaluation(D, L, DE, LE, working_points)
    l_r = 10**-4
    l_q = 10**-6
    logistic_regression_evaluation(D, L, DE, LE, working_points, l_r, l_q)
    C = 0.01
    C_q = 0.1
    c = 1
    C_rbf = 15
    psi = 0.1
    svm_evaluation(D, L, DE, LE, working_points, C, C_q, c, C_rbf, psi)
    GMM_evaluation(D, L, DE, LE, working_points)
    
    print("EVALUATION Gauss")
    Gaussian_evaluation(DG, L, DGE, LE, working_points)
    l_r = 10**-3
    l_q = 10**-3
    logistic_regression_evaluation(DG, L, DGE, LE, working_points, l_r, l_q)
    C = 0.1
    C_q = 100
    c = 0.1
    C_rbf = 10
    psi = 0.01
    prior_rbf = 0.5
    svm_evaluation(DG, L, DGE, LE, working_points, C, C_q, c, C_rbf, psi)
    GMM_evaluation(DG, L, DGE, LE, working_points)    
    score_quality_evaluation(DG,L,DGE,LE,working_points, C_rbf, psi, prior_rbf)
