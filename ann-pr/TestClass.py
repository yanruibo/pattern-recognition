#!/usr/bin/python
# encoding: utf-8

'''
Created on Nov 18, 2015

@author: yanruibo
'''
class Test:
    def __init__(self):
        print "init"
    def __del__(self):
        print "del"
if __name__ == '__main__':
    test = Test()
    