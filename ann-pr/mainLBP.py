#! /usr/bin/env python
# coding=utf-8

'''
Created on 2015年10月16日

@author: yanruibo
'''
import sys
import time
import numpy as np
import cv2
from os import listdir
import numpy as np
'''
mat：二维矩阵
x，y （x，y）坐标

获得二维矩阵mat的(x,y)处的值，因为涉及边界问题，
且重复使用度高，故将其写作一个函数，将超出图像边界的值设置为0
'''
def getValue(mat, x, y):
    try:
        return mat[x, y]
    except IndexError:
        return 0
'''
binarySequence:八位的二进制数列表 eg: [1,0,1,1,1,0,1,1]
输入一个八位的二进制数列表，从不同位置循环右移八次得到八个数值，
并取得其中的最小值返回
'''
def minRORValue(binarySequence):
    # eightValues存储循环右移得到的八个数值
    eightValues = []
    # 循环右移算法
    for i in range(8):
        sum = 0
        pos = i
        for j in range(8):
            sum += int(binarySequence[pos % 8]) * pow(2, 7 - j)
            pos = pos + 1
        eightValues.append(sum)
    # 从八个数值中找到最小的数值
    minValue = min(eightValues)
    return minValue
'''
通过循环计算得出36种编码的值
'''
def get36Values():
    ans = []
    for i in range(256):
        # 将0-255转成二进制字符串 注：python中没有直接转换成二进制数值的函数，没有搜索到
        binStr = '{0:b}'.format(i)
        binList = []
        # 将二进制字符串放在binList列表中
        for i in range(len(binStr)):
            binList.append(binStr[i])
        # 0-255数值中转化为二进制之后不足八位在前面补0
        if(len(binList) < 8):
            for i in range(8 - len(binList)):
                binList.insert(0, '0')
        minValue = minRORValue(binList)
        if minValue not in ans:
            ans.append(minValue)
    return ans
    
'''
mat:二维矩阵

输入一个二维矩阵，经过LBP处理得到36个值，将36个值从小到大排序，
返回每个值的数量的列表，如果一个值没有出现，则在相应的位置返回0

'''
def LBP(mat):
    
    afterRORValues = []
    # 两重循环遍历每个像素点
    for x in range(0, len(mat)):
        for y in range(0, len(mat[0])):
            center = mat[x, y]
            topLeft = getValue(mat, x - 1, y + 1)
            topUp = getValue(mat, x, y + 1)
            topRight = getValue(mat, x + 1, y + 1)
            right = getValue(mat, x + 1, y)
            bottomRight = getValue(mat, x + 1, y - 1)
            bottomCenter = getValue(mat, x, y - 1)
            bottomLeft = getValue(mat, x - 1, y - 1)
            left = getValue(mat, x - 1, y)
            # 获得一个center像素点周围八个点的列表
            aroundCenter = [topLeft, topUp, topRight, right, bottomRight, bottomCenter, bottomLeft, left]
            # 将八个点的值转为八位二进制数，如果大于中间数记为1，如果小于中间数记为0
            binarySequence = []
            for i in range(len(aroundCenter)):
                if(aroundCenter[i] > center):
                    flag = 1
                else:
                    flag = 0
                binarySequence.append(flag)
            # 获得最小的值，并加入到afterRORValues列表中
            minValue = minRORValue(binarySequence)
            afterRORValues.append(minValue)
    # 对经过ROR处理之后的列表进行排序，此时afterRORValues列表中的取值为36中或者少于36个
    sortedValues = sorted(afterRORValues)
    fixedValues = get36Values()
    # 统计36个值（从小到大排列），每个值的取值个数，将每个值出现的次数放在returnValues列表中
    returnValues = []
    for i in range(len(fixedValues)):
        counter = 0
        for item in sortedValues:
            if(item == fixedValues[i]):
                counter = counter + 1
        returnValues.append(counter)
    
    '''
    这里提供了另外一种方法可以计算直方图的取值
    fixedValues.append(256)
    hist, bins = np.histogram(afterRORValues,fixedValues)
    print hist
    print type(hist)
    print "len of hist",len(hist) 
    print bins
    '''
    return returnValues
'''
filename:写的文件名
fileList:待写的图片的文件名的列表
type:类型　pos or neg
将处理后的36维的实例写入文件中
'''
def writeToFile(filename, fileList,typename):
    fp = open(filename, "w")
    for i in range(len(fileList)):
        print "processing ",i
        mat = cv2.imread('../Database_PR_02/%s/%s' % (typename,fileList[i]))
        gray = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
        listData = LBP(gray)
        for i in range(len(listData)):
            if(i == (len(listData) - 1)):
                fp.write(str(listData[i]) + "\n")
            else:
                fp.write(str(listData[i]) + " ")
    fp.flush()
    fp.close()
'''
对pos.txt和neg.txt中的数据按行进行归一化
'''
def normalize():
    pos_matrix = np.loadtxt("pos.txt", dtype=float, delimiter=' ')
    print pos_matrix.shape
    neg_matrix = np.loadtxt("neg_remove_noise.txt", dtype=float, delimiter=' ')
    print neg_matrix.shape
    #column_sums = pos_matrix.sum(axis=0)
    #pos_new_matrix = pos_matrix / column_sums[np.newaxis,:]
    row_sums = pos_matrix.sum(axis=1)
    pos_new_matrix = pos_matrix / row_sums[:,np.newaxis]
    np.savetxt("pos_normalized_row_remove_noise.txt", pos_new_matrix)
    #column_sums = neg_matrix.sum(axis=0)
    #neg_new_matrix = neg_matrix / column_sums[np.newaxis,:]
    row_sums = neg_matrix.sum(axis=1)
    neg_new_matrix = neg_matrix / row_sums[:,np.newaxis]
    np.savetxt("neg_normalized_row_remove_noise.txt", neg_new_matrix)
    
if __name__ == '__main__':

#     startTimeStamp = time.time()
#     
#     posFileList = listdir('../Database_PR_02/pos')
#     negFileList = listdir('../Database_PR_02/neg')
#     
#     # 写入文件
#     writeToFile("pos.txt", posFileList,"pos")
#     writeToFile("neg.txt", negFileList,"neg")
#     # 统计写入时间
#     print "begin normalization"
#     normalize()
#     endTimeStamp = time.time()
#     total_time = endTimeStamp - startTimeStamp
#     ft = open("total_time.txt", "w")
#     ft.write("Normal Total Time : " + str(total_time) + "\n")
#     ft.close()
#     
    normalize()
