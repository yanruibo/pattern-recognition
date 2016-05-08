#!/usr/bin/python
# encoding: utf-8

'''
Created on Nov 13, 2015

@author: yanruibo
反向传播三层神经网络

'''

import numpy as np
import random


class ANN:
    '''
    构造函数:初始化变量
    input_num是输入层的节点个数
    hidden_num是隐层节点个数
    output_num是输出层的节点个数
    '''
    def __init__(self, input_num, hidden_num, output_num):
        
        self.input_num = input_num + 1  # 增加一个偏差节点，相当于增加了偏置
        self.hidden_num = hidden_num
        self.output_num = output_num
        
        # input_values记录输入层节点的值
        # hidden_values记录隐层节点的值
        # output_values记录输出节点的值
        self.input_values = np.ones(self.input_num)
        self.hidden_values = np.ones(self.hidden_num)
        self.output_values = np.ones(self.output_num)

        # w1[i][j]是从输入层第i个节点到隐层第j个节点的权值
        # w2[j][k]是从隐层第j个节点到输出层第k个节点的权值
        self.w1 = np.random.uniform(-0.5, 0.5, (self.input_num, self.hidden_num))
        self.w2 = np.random.uniform(-0.5, 0.5, (self.hidden_num, self.output_num))
    
        # last_change1[i][j]是从输入层第i个节点到隐层第j个节点的上一次的权值修正项(记忆项，动量项或者惯性项)
        # last_change２[j][k]是从隐层第j个节点到输出层第k个节点的上一次的权值修正项(记忆项，动量项或者惯性项)
        self.last_change1 = np.zeros((self.input_num, self.hidden_num))
        self.last_change2 = np.zeros((self.hidden_num, self.output_num))
        # 输出一些log信息
        self.fp = open("log_info.txt", "a")
    
    '''
    关闭打开的文件描述符
    '''
    def __del__(self):
        self.fp.close()
    '''
    sigmoid函数输入自变量x,返回函数y的值
    '''
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    '''
    sigmoid函数的梯度
    '''
    def gradient_sigmoid(self, y):
        return y * (1.0 - y)
    '''
    前向传播函数
    x：输入向量
    '''    
    def forward(self, x):
        if(len(x) != self.input_num - 1):
            raise ValueError("输入个数不匹配")
        # 激活输入层节点
        for i in range(self.input_num - 1):
            # self.input_values[i] = sigmoid(x[i]) four layer networks
            self.input_values[i] = x[i]
        # 激活隐层节点
        for j in range(self.hidden_num):
            sum = 0.0
            for i in range(self.input_num):
                sum = sum + self.input_values[i] * self.w1[i][j]
            self.hidden_values[j] = self.sigmoid(sum)
        # 激活输出层节点
        for k in range(self.output_num):
            sum = 0.0
            for j in range(self.hidden_num):
                sum = sum + self.hidden_values[j] * self.w2[j][k]
            self.output_values[k] = self.sigmoid(sum)
        # 返回输出层的结果
        return self.output_values[:]
    '''
    反向传播函数1:只设定步长
    y：标签值
    eta：步长
    '''
    def back_propagation(self, y, eta):
        if(len(y) != self.output_num):
            raise ValueError("输出值个数不匹配")
        # 计算输出层和隐层的delta对应教材公式中的delta，下面更新权值需要使用
        output_deltas = np.zeros(self.output_num)
        for k in range(self.output_num):
            error = y[k] - self.output_values[k]
            output_deltas[k] = self.gradient_sigmoid(self.output_values[k]) * error
        
        hidden_deltas = np.zeros(self.hidden_num)
        for j in range(self.hidden_num):
            error = 0.0
            for k in range(self.output_num):
                error = error + output_deltas[k] * self.w2[j][k]
            hidden_deltas[j] = self.gradient_sigmoid(self.hidden_values[j]) * error
        # 更新隐层到输出层的权值
        for j in range(self.hidden_num):
            for k in range(self.output_num):
                change = output_deltas[k] * self.hidden_values[j]
                self.w2[j][k] = self.w2[j][k] + eta * change
        # 更新输入层到隐层的权值
        for i in range(self.input_num):
            for j in range(self.hidden_num):
                change = hidden_deltas[j] * self.input_values[i]
                self.w1[i][j] = self.w1[i][j] + eta * change
        # 计算平方误差
        error = 0.0
        for k in range(len(y)):
            error += 0.5 * (y[k] - self.output_values[k]) ** 2
        return error
    '''
    反向传播函数2:
    ｙ:标签值
    alpha：动量因子
    eta:学习率(步长)
    '''
    def back_propagation2(self, y, alpha, eta):
        if(len(y) != self.output_num):
            raise ValueError("输出值个数不匹配")
        
        # 计算输出层和隐层的delta对应教材公式中的delta，下面更新权值需要使用
        output_deltas = np.zeros(self.output_num)
        for k in range(self.output_num):
            output_deltas[k] = self.gradient_sigmoid(self.output_values[k]) * (y[k] - self.output_values[k])
        
        hidden_deltas = np.zeros(self.hidden_num)
        for j in range(self.hidden_num):
            error = 0.0
            for k in range(self.output_num):
                error = error + output_deltas[k] * self.w2[j][k]
            hidden_deltas[j] = self.gradient_sigmoid(self.hidden_values[j]) * error
        
        # 更新隐层到输出层的权值
        for j in range(self.hidden_num):
            for k in range(self.output_num):
                change = output_deltas[k] * self.hidden_values[j]
                self.w2[j][k] = self.w2[j][k] + alpha * self.last_change2[j][k] + eta * change
                self.last_change2[j][k] = change
            
        # 更新输入层到隐层的权值
        for i in range(self.input_num):
            for j in range(self.hidden_num):
                change = hidden_deltas[j] * self.input_values[i]
                self.w1[i][j] = self.w1[i][j] + alpha * self.last_change1[i][j] + eta * change
                self.last_change1[i][j] = change
        # 计算平方误差
        error = 0.0
        for k in range(len(y)):
            error += 0.5 * (y[k] - self.output_values[k]) ** 2
        return error
    '''
    训练函数
    train_matrix:训练向量组成的矩阵，最后一列是标签列
    iteration:对整个训练集的迭代次数
    threshhold：阀值
    alpha:动量因子
    eta：学习率(步长)
    '''    
    def train(self, train_matrix, iteration, threshold, alpha, eta):
        
        real_iteration = 0
        
        '''
        for i in range(iteration):
            current_error = 0.0
            last_error = 0.0
            train_num = len(train_matrix)
            
            line = train_matrix[i % train_num]
            
            x = line[:len(line) - 1]
            y = line[-1].reshape(1)
            
            self.forward(x)
            current_error = self.back_propagation2(y, alpha, eta)
            
            # error = self.back_propagation(y, eta)
            if(current_error - last_error < threshold):
                print "已收敛"
                break
            last_error = current_error
            real_iteration = real_iteration + 1
            if(i%100==0):
                print "current error : ", current_error
        '''
        
        for i in range(iteration):
            current_error = 0.0
            last_error = 0.0
            #对样本完全遍历完之后就进行shuffle一下
            np.random.shuffle(train_matrix)
            for line in train_matrix:
                x = line[:len(line) - 1]
                y = line[-1].reshape(1)
                self.forward(x)
                current_error += self.back_propagation2(y, alpha, eta)
                real_iteration = real_iteration + 1
            if(current_error - last_error < threshold):
                print "已收敛"
                break
            last_error = current_error
            print "current error : ", current_error / len(train_matrix)
        
        self.fp.write("iterations : " + str(real_iteration) + "\n")
        self.fp.write("threshold : " + str(threshold) + "\n")
        self.fp.write("alpha : " + str(alpha) + "\n")
        self.fp.write("eta : " + str(eta) + "\n")
    '''
    测试函数
    test_matrix:测试向量组成的矩阵，最后一列是标签列
    '''
    def test(self, test_matrix):
        fd = open("predict.txt", "w")
        right_counter = 0
        for line in test_matrix:
            x = line[:len(line) - 1]
            y = line[-1]
            predict = self.forward(x)
            fd.write("real : " + str(y) + " predict : " + str(predict) + "\n")
            predict_result = None
            #分类
            if (predict > 0.0 and predict < 0.5):
                predict_result = 0.0
            elif(predict <= 1.0):
                predict_result = 1.0
            if(predict_result == y):
                right_counter += 1
        accuracy = float(right_counter) / len(test_matrix)
        self.fp.write("accuracy : " + str(accuracy) + "\n")
        fd.close()
    
'''
类外测试函数
k:随机选取的正性样本的数目或者负性样本的数目
alpha:动量因子
eta:步长
'''
def test(k,alpha,eta):

    #pos_matrix = np.loadtxt("pos_normalized.txt", dtype=float, delimiter=' ')
    #neg_matrix = np.loadtxt("neg_normalized.txt", dtype=float, delimiter=' ')
    pos_matrix = np.loadtxt("pos_normalized_row_remove_noise.txt", dtype=float, delimiter=' ')
    neg_matrix = np.loadtxt("neg_normalized_row_remove_noise.txt", dtype=float, delimiter=' ')
    full_pos_index = range(len(pos_matrix))
    full_neg_index = range(len(neg_matrix))
    # 从总的正性样本中随机选取k个作为测试样本，并将其索引放在train_pos_indexes中，
    # 同时从总的负性样本中随机选取k个作为测试样本，并将其索引放在train_neg_indexes中
    train_pos_indexes = random.sample(range(0, len(pos_matrix)), k)
    train_neg_indexes = random.sample(range(0, len(neg_matrix)), k)
    print train_neg_indexes
    print train_pos_indexes
    # np.savetxt("train_neg_indexes_10.txt", train_neg_indexes)
    # np.savetxt("train_pos_indexes_10.txt", train_pos_indexes)
    
    trainX = np.zeros([2 * k, 36])  # 2k行36列
    trainY = np.zeros([2 * k])  # 2k行
    for i in range(k):
        trainX[i, :] = pos_matrix[train_pos_indexes[i]]
        trainY[i] = 1 
        trainX[i + k, :] = neg_matrix[train_neg_indexes[i]]
        trainY[i + k] = 0
    train_matrix = np.append(trainX, trainY.reshape(len(trainY), 1), axis=1)
    
    test_pos_indexes = list(set(full_pos_index) - set(train_pos_indexes))
    test_neg_indexes = list(set(full_neg_index) - set(train_neg_indexes))
    total_num = len(pos_matrix) + len(neg_matrix)
    testX = np.zeros([total_num - 2 * k, 36])
    testY = np.zeros([total_num - 2 * k])
    
    for i in range(len(test_pos_indexes)):
        testX[i, :] = pos_matrix[test_pos_indexes[i]]
        testY[i] = 1 
    for i in range(len(test_neg_indexes)):
        testX[i + len(test_pos_indexes), :] = neg_matrix[test_neg_indexes[i]]
        testY[i + len(test_pos_indexes)] = 0
    test_matrix = np.append(testX, testY.reshape(len(testY), 1), axis=1)
    #np.savetxt("train_matrix.txt", train_matrix)
    #np.savetxt("test_matrix.txt", test_matrix)
    #调用神经网络初始化
    ann = ANN(36, 36, 1)
    #训练，对整个训练样本迭代500次，实际迭代的次数为：500*训练样本的数量,剩下三个参数依次是阀值，动量因子,步长
    ann.train(train_matrix, 500, np.power(10, -6), alpha, eta)
    #测试
    ann.test(test_matrix)

    
if __name__ == '__main__':
    test(100,0.1,0.8)
    
