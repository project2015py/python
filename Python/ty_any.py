#!/usr/bin/env python

import numpy as np
"""
matlab Copyright 2013 Yili Zheng, Qiu Wang, Peter C. Doerschuk
python Copyright 2014 Shenghan Gao, Yayi Li, Yu Tang, Peter C. Doerschuk
Cornell University has not yet decided on the license for this software so no rights come with this file.
Certainly no warrenty of any kind comes with this file.
All
"""

def ty_any_smaller(a, target):
    for x in np.nditer(a):
        if x < target:
            return True
    return False

def ty_any_smaller_or_equal(a, target):
    for x in np.nditer(a):
        if x <= target:
            return True
    return False