# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:03:48 2021

@author: LucaS
"""

import numpy
import matplotlib.pyplot as plt

def compute_bayes_risk(m, wp):
    return wp["prior"]*wp["Cfn"]*m[0][1]/(m[0][1]+m[1][1])+(1-wp["prior"])*wp["Cfp"]*m[1][0]/(m[0][0]+m[1][0]) 

def compute_min_DCF(m, labels, wp):    
    minDCF = float('inf')
    for x in m:
        labelsA = numpy.zeros((m.shape[0]), dtype=numpy.int32)
        labelsA[m >= x] = 1
        confusionMatrix = numpy.zeros((2,2))
        for i in range(len(labels)):
            confusionMatrix[labelsA[i]][labels[i]] += 1;
            
        bayes_risk = compute_bayes_risk(confusionMatrix, wp)
        if bayes_risk < minDCF:
            minDCF = bayes_risk
    
    t = max(m)+10
    labelsA = numpy.zeros((m.shape[0]), dtype=numpy.int32)
    labelsA[m >= t] = 1
    confusionMatrix = numpy.zeros((2,2))
    for i in range(len(labels)):
        confusionMatrix[labelsA[i]][labels[i]] += 1;      
    bayes_risk = compute_bayes_risk(confusionMatrix, wp)
    if bayes_risk < minDCF:
        minDCF = bayes_risk  
        
    return minDCF/min(wp["prior"]*wp["Cfn"], (1-wp["prior"])*wp["Cfp"])  


def compute_act_DCF(scores, LTE, wp):    
    confusionMatrix = numpy.zeros((2,2))
    t = -numpy.log(wp["prior"]/(1-wp["prior"]))
    actual = numpy.zeros((len(LTE)), dtype=numpy.int32)
    actual[scores >= t] = 1
    for i in range(len(LTE)):
        confusionMatrix[actual[i]][LTE[i]] += 1;
                        
    return compute_bayes_risk(confusionMatrix, wp)/min(wp["prior"]*wp["Cfn"], (1-wp["prior"])*wp["Cfp"])

def compute_DCF_with_threshold(scores, LTE, wp, t):    
    confusionMatrix = numpy.zeros((2,2))
    actual = numpy.zeros((len(LTE)), dtype=numpy.int32)
    actual[scores >= t] = 1
    for i in range(len(LTE)):
        confusionMatrix[actual[i]][LTE[i]] += 1;
                        
    return compute_bayes_risk(confusionMatrix, wp)/min(wp["prior"]*wp["Cfn"], (1-wp["prior"])*wp["Cfp"])


def plot_bayes_error(m_svm, labels):
    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    
    actdcf_svm = numpy.zeros((21))
    mindcf_svm = numpy.zeros((21))    
    j=0
    for t1 in effPriorLogOdds:
        piEff = 1/(1+numpy.exp(t1))
        wp={"prior":piEff, "Cfn":1, "Cfp":1}
            
        actdcf_svm[j] = compute_act_DCF(m_svm, labels, wp)
        mindcf_svm[j] = compute_min_DCF(m_svm, labels, wp)

        j+=1
        
        
    plt.figure()
    plt.xlabel('log(π/(1-π))', fontsize=15)
    plt.ylabel('DCF', fontsize=15)
    plt.plot(effPriorLogOdds, actdcf_svm, 'b', label='RBF SVM - act DCF')
    plt.plot(effPriorLogOdds, mindcf_svm,  'b--', label='RBF SVM - min DCF')  
    plt.legend(prop={'size': 15})
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()
    