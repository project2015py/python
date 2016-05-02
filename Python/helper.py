#!/usr/bin/env python
# helper.py
# Yunhan Wang (yw559)
# Oct 23, 2013
""" Description: Package of all helper methods such as printing, saving, etc.
Updated by Yunhan Wang 8/28/2014
"""
""" random seed """

def printf(std, a):
    """ use this instead of built-in print,
    in case we not only need printing but also need to
    record the text (etc. save()) """
#    print(a)
    std.write(a+'\n')

def disp(std, a):
    printf(std, str(a))
    
def errPrint(x):
    print(x)

def slct(l,n):
    # slct.m modified frofunction rt=slct(l,n)m slct.c
    # 2/21/96
    # Selection rule of icosahedral harmonics - if l-30*n is a non negative
    # integer other than 1,2,3,4,5,7,8,9,11,13,14,17,19,23,29 then it is a
    # valid order.
    # Example: slct(0,0)=1, slct(6,0)=1, slct(30,1)=1, slct(3,0)=0.
    rt, i = 0, l - n * 30
    if i >= 0 and n >= 0 and i not in [1,2,3,4,5,7,8,9,11,13,14,17,19,23,29]:
        rt = 1
    return rt