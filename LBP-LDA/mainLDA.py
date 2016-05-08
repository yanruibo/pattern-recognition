#! /usr/bin/env python
# coding=utf-8

'''
Created on Oct 18, 2015

@author: yanruibo
'''


import numpy as np
import random
import matplotlib.pyplot as plt
'''
trainX：训练样本集X的取值，是一个矩阵 一行一个实例
trainY：训练样本集Y的取值
输入训练样本集x和y的值，输出w投影方向和 w0阀值
w1为neg w2为pos
'''
def LDA(trainX, trainY):
    
    negIndexes = []  # 训练集负性样本的索引
    negCounts = 0  # 训练集负性样本数量
    posIndexes = []  # 训练集正性样本的索引
    posCounts = 0  # 训练集正性样本数量
    for i in range(len(trainY)):
        # 如果y值为0，则为负性样本，如果y值为1，则为正性样本
        if(trainY[i] == 0):
            negIndexes.append(i)
            negCounts = negCounts + 1
        else:
            posIndexes.append(i)
            posCounts = posCounts + 1
    # 求负性样本的类均值向量 mNeg
    negSum = np.zeros([1, len(trainX[0])])
    for negIndex in negIndexes:
        negSum += trainX[negIndex]
    mNeg = negSum.T / negCounts
    
    # 求负性样本的类内散度矩阵S
    sNeg = np.dot((trainX[negIndexes[0]].T - mNeg), (trainX[negIndexes[0]].T - mNeg).T)
    for i in range(1, len(negIndexes)):
        sNeg += np.dot((trainX[negIndexes[i]].T - mNeg), (trainX[negIndexes[i]].T - mNeg).T)
    
    # 求正性样本的类均值向量 mPos
    posSum = np.zeros([1, len(trainX[0])])
    for posIndex in posIndexes:
        posSum += trainX[posIndex]
    mPos = posSum.T / posCounts
    
    # 求正性样本的类内散度矩阵S
    sPos = np.dot((trainX[posIndexes[0]].T - mPos), (trainX[posIndexes[0]].T - mPos).T)
    for i in range(1, len(posIndexes)):
        sPos += np.dot((trainX[posIndexes[i]].T - mPos), (trainX[posIndexes[i]].T - mPos).T)
    # 求类内总散度矩阵
    sw = sNeg + sPos
    inv_sw = np.linalg.inv(sw)
    # 求投影方向
    w = np.dot(inv_sw, (mNeg - mPos))
    
    pPos = float(posCounts) / len(trainY)
    pNeg = float(negCounts) / len(trainY)
    # 求阀值 w0
    w0 = -0.5 * (np.dot(np.dot((mNeg + mPos).T, inv_sw), mNeg - mPos)) - np.log(pPos / pNeg)
    
    #************************************#
    '''
    绘图，绘出训练样本的投影分布ｗＴ*ｘ+w0和分界线０
    该功能是后来添加的，没有把它单独写成一个函数。
    '''
    w0_value = w0[0][0]
    neg_projects = []
    for i in range(len(negIndexes)):
        project_value = np.dot(w.T, trainX[negIndexes[i]].T)[0] + w0_value
        neg_projects.append(project_value)
        
    pos_projects = []
    for i in range(len(posIndexes)):
        project_value = np.dot(w.T, trainX[posIndexes[i]].T)[0] + w0_value
        pos_projects.append(project_value)
    # print min(neg_projects),max(neg_projects)
    # print min(pos_projects),max(pos_projects)
    plt.plot(range(len(neg_projects)), neg_projects)
    plt.plot(range(len(pos_projects)), pos_projects)
    
    zero_values = []
    for i in range(len(neg_projects)):
        zero_values.append(0)
    plt.plot(range(len(neg_projects)), zero_values)
    plt.show()
    
    #************************************#
    
   

    return w, w0
'''
该函数从所有的正性和负性样本中随机去ｋ个正性样本和ｋ个负性样本进行训练，形成分类模型，对剩下的数据进行测试。

'''
def test(k):
    # 将所有的正性和负性数据加载进内存
    pos_matrix = np.loadtxt("pos.txt", dtype=float, delimiter=' ')
    print pos_matrix.shape
    neg_matrix = np.loadtxt("neg.txt", dtype=float, delimiter=' ')
    full_index = range(1000)
    # 从总的正性样本中随机选取k个作为测试样本，并将其索引放在train_pos_indexes中，
    # 同时从总的负性样本中随机选取k个作为测试样本，并将其索引放在train_neg_indexes中
    
    train_pos_indexes = random.sample(range(0, 1000), k)
    train_neg_indexes = random.sample(range(0, 1000), k)
    
    print train_neg_indexes
    print train_pos_indexes
    # np.savetxt("train_neg_indexes_10.txt", train_neg_indexes)
    # np.savetxt("train_pos_indexes_10.txt", train_pos_indexes)
    # 形成训练矩阵，作为LDA函数的输入
    trainX = np.zeros([2 * k, 36])  # 2k行36列
    trainY = np.zeros([2 * k])  # 2k行
    for i in range(k):
        trainX[i, :] = pos_matrix[train_pos_indexes[i]]
        trainY[i] = 1 
        trainX[i + k, :] = neg_matrix[train_neg_indexes[i]]
        trainY[i + k] = 0
    # print trainX
    # print trainY
    w, w0 = LDA(trainX, trainY)
    w0_value = w0[0][0]
    wT = w.T
    # print wT.shape
    # print w0.shape
    # 计算出除去训练样本剩下的样本 用来测试
    test_pos_indexes = list(set(full_index) - set(train_pos_indexes))
    test_neg_indexes = list(set(full_index) - set(train_neg_indexes))
    # 分类并计算精确度
    right_counter = 0
    for item in test_neg_indexes:
        gx = np.dot(wT, neg_matrix[item].T) + w0_value
        gx_value = gx[0]
        if(gx_value > 0):
            right_counter = right_counter + 1
            # print "real neg, predict neg"
        else:
            # print "real neg, predict pos"
            pass
    for item in test_pos_indexes:
        gx = np.dot(wT, pos_matrix[item].T) + w0_value
        gx_value = gx[0]
        if(gx_value < 0):
            right_counter = right_counter + 1
    precision = float(right_counter) / (2000 - 2 * k)
    
    
    #************************************#
    '''
    绘图，绘出训练样本的投影分布ｗＴ*ｘ+w0和分界线０
    该功能是后来添加的，没有把它单独写成一个函数。
    '''
    w0_value = w0[0][0]
    neg_projects = []
    for i in range(len(test_neg_indexes)):
        project_value = np.dot(w.T, neg_matrix[test_neg_indexes[i]].T)[0] + w0_value
        neg_projects.append(project_value)
        
    pos_projects = []
    for i in range(len(test_pos_indexes)):
        project_value = np.dot(w.T, pos_matrix[test_pos_indexes[i]].T)[0] + w0_value
        pos_projects.append(project_value)
    # print min(neg_projects),max(neg_projects)
    # print min(pos_projects),max(pos_projects)
    plt.plot(range(len(neg_projects)), neg_projects)
    plt.plot(range(len(pos_projects)), pos_projects)
    
    zero_values = []
    for i in range(len(neg_projects)):
        zero_values.append(0)
    plt.plot(range(len(neg_projects)), zero_values)
    plt.show()
    
    #************************************#
    print "precision", precision

'''
用提前找好的ｋ个训练正性样本和ｋ个负性样本建立线性模型，对剩下的数据进行分类测试
'''
def test_with_fixed_data(k, train_neg_indexes, train_pos_indexes):
    # 将所有的正性和负性数据加载进内存
    pos_matrix = np.loadtxt("pos.txt", dtype=float, delimiter=' ')
    neg_matrix = np.loadtxt("neg.txt", dtype=float, delimiter=' ')
    full_index = range(1000)
    
    # 形成训练矩阵，作为LDA函数的输入
    trainX = np.zeros([2 * k, 36])  # 2k行36列
    trainY = np.zeros([2 * k])  # 2k行
    for i in range(k):
        trainX[i, :] = pos_matrix[train_pos_indexes[i]]
        trainY[i] = 1 
        trainX[i + k, :] = neg_matrix[train_neg_indexes[i]]
        trainY[i + k] = 0
    # print trainX
    # print trainY
    w, w0 = LDA(trainX, trainY)
    w0_value = w0[0][0]
    print "w", w
    print "w0", w0_value
    wT = w.T
    # print wT.shape
    # print w0.shape
    # 计算出除去训练样本剩下的样本 用来测试
    test_pos_indexes = list(set(full_index) - set(train_pos_indexes))
    test_neg_indexes = list(set(full_index) - set(train_neg_indexes))
    # 分类并计算精确度
    right_counter = 0
    for item in test_neg_indexes:
        gx = np.dot(wT, neg_matrix[item].T) + w0_value
        gx_value = gx[0]
        if(gx_value > 0):
            right_counter = right_counter + 1
            # print "real neg, predict neg"
        else:
            # print "real neg, predict pos"
            pass
    for item in test_pos_indexes:
        gx = np.dot(wT, pos_matrix[item].T) + w0_value
        gx_value = gx[0]
        if(gx_value < 0):
            right_counter = right_counter + 1
    precision = float(right_counter) / (2000 - 2 * k)
    
    #************************************#
    '''
    绘图，绘出测试样本的投影分布ｗＴ*ｘ+w0和分界线０
    '''
    w0_value = w0[0][0]
    neg_projects = []
    for i in range(len(test_neg_indexes)):
        project_value = np.dot(w.T, neg_matrix[test_neg_indexes[i]].T)[0]+w0_value
        neg_projects.append(project_value)
        
    pos_projects = []
    for i in range(len(test_pos_indexes)):
        project_value = np.dot(w.T, pos_matrix[test_pos_indexes[i]].T)[0]+w0_value
        pos_projects.append(project_value)
    # print min(neg_projects),max(neg_projects)
    # print min(pos_projects),max(pos_projects)
    plt.plot(range(len(test_neg_indexes)), neg_projects)
    plt.plot(range(len(test_pos_indexes)), pos_projects)
    
    zero_values = []
    for i in range(len(neg_projects)):
        zero_values.append(0)
    plt.plot(range(len(neg_projects)), zero_values)
    plt.show()
    
    #************************************#
    print "precision", precision
if __name__ == '__main__':
    '''
    tesk(k)函数是从总样本中随机选取ｋ个正性样本和ｋ个负性样本，由于样本是随机选的，所以效果可能会不好
    test_with_fixed_data(20,np.loadtxt("train_neg_indexes_20.txt"),np.loadtxt("train_pos_indexes_20.txt"))是我
    用test(k)函数调出来的当k=20时,即取２０个正性样本和２０个负性样本时的分类效果，由于是我自己选的所以效果比较好
    相同的一下为ｋ取50 100 500 和　900时的情况
    test_with_fixed_data(50,np.loadtxt("train_neg_indexes_50.txt"),np.loadtxt("train_pos_indexes_50.txt"))
    test_with_fixed_data(100,np.loadtxt("train_neg_indexes_100.txt"),np.loadtxt("train_pos_indexes_100.txt"))
    test_with_fixed_data(500,np.loadtxt("train_neg_indexes_500.txt"),np.loadtxt("train_pos_indexes_500.txt"))
    test_with_fixed_data(900,np.loadtxt("train_neg_indexes_900.txt"),np.loadtxt("train_pos_indexes_900.txt"))
    
    运行test_with_fixed_data函数会先给出训练样本的投影分布和阀值，红线标示阀值，绿线和蓝线表示两类，然后给出测试样本的投影分布。
    最后在控制台会有投影方向w,阀值w0,准确率precison的打印。
    '''
    #test(100);
    # test_with_fixed_data(20,np.loadtxt("train_neg_indexes_20.txt"),np.loadtxt("train_pos_indexes_20.txt"))
    test_with_fixed_data(50,np.loadtxt("train_neg_indexes_50.txt"),np.loadtxt("train_pos_indexes_50.txt"))
    # test_with_fixed_data(100,np.loadtxt("train_neg_indexes_100.txt"),np.loadtxt("train_pos_indexes_100.txt"))
    # test_with_fixed_data(500,np.loadtxt("train_neg_indexes_500.txt"),np.loadtxt("train_pos_indexes_500.txt"))
    # test_with_fixed_data(900,np.loadtxt("train_neg_indexes_900.txt"),np.loadtxt("train_pos_indexes_900.txt"))
    
    
    
