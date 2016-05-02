#!/usr/bin/env python

import numpy as np
"""
matlab Copyright 2013 Yili Zheng, Qiu Wang, Peter C. Doerschuk
python Copyright 2014 Shenghan Gao, Yayi Li, Yu Tang, Peter C. Doerschuk
Cornell University has not yet decided on the license for this software so no rights come with this file.
Certainly no warrenty of any kind comes with this file.
All of this will be corrected when Cornell University comes to a decision.
"""

def ty_nnz(a):
    """
    @type: np.matrix
    @param a matrix

    @rtype: int
    @return number of non-zero elements in the matrix
    """
    res = 0
    for x in np.nditer(a):
        if x != 0:
            res += 1

    return res