#!/usr/bin/env python

import numpy as np
"""
matlab Copyright 2013 Yili Zheng, Qiu Wang, Peter C. Doerschuk
python Copyright 2014 Shenghan Gao, Yayi Li, Yu Tang, Peter C. Doerschuk
Cornell University has not yet decided on the license for this software so no rights come with this file.
Certainly no warrenty of any kind comes with this file.
All
"""

def ty_isnumeric(a):
    for x in np.nditer(a):
        if type(x) == type(True):
            return False
    return True
