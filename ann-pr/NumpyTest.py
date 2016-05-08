#!/usr/bin/python
# encoding: utf-8

'''
Created on Nov 14, 2015

@author: yanruibo
'''
import numpy as np
import os
import cv2
if __name__ == '__main__':
    
#     a = np.array([[1, 2], [3, 4]])
#     b = np.array([[5, 6]])
#     c = np.concatenate((a, b.T), axis=1)
#     print(c)
    pos_matrix = np.array([[1.0,2.0,1.0],[1.0,1.0,1.0],[2.0,1.0,1.0]])
    print pos_matrix.shape
    print pos_matrix
    column_sums = pos_matrix.sum(axis=0)
    print column_sums.shape
    print column_sums
    print column_sums[np.newaxis,:].shape
    print column_sums[np.newaxis,:]
    pos_new_matrix = pos_matrix / column_sums[np.newaxis,:]
    print pos_new_matrix
#     
#     mat = cv2.imread('../Database_PR_02/%s/%s' % ("pos","pos05000.png"))
#     
#     gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
#     print gray.shape
#     mat = cv2.imread('../Database_PR_02/%s/%s' % ("neg","neg11000.png"))
#     
#     gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
#     print gray.shape
#     
#     negFileList = os.listdir('../Database_PR_02/neg')
#     #9 150 155 192 307
#     print negFileList[8]
#     print negFileList[149]
#     print negFileList[154]
#     print negFileList[191]
#     print negFileList[306]
#     #175 239 340 584 599
#     print "weird:"
#     print negFileList[174]
#     print negFileList[238]
#     print negFileList[339]
#     print negFileList[583]
#     print negFileList[598]
#     
#     