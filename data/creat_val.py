# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:13:34 2017

@author: vcaadmin
"""
import os, shutil

path = '/data/huang/behaviour/data/data_photos/'
path1 = '/data/huang/behaviour/data/data_v1/'
VCA_LABELS = sorted(os.listdir(path))
for i in VCA_LABELS:
	if i.split("_")[0] == "1":
	    print(i)
	    li_all = sorted(os.listdir(os.path.join(path, i)))
	    l = 0
	    for li in li_all:
		if os.path.splitext(li)[1] == '.jpg':
			l += 1
			if 6 > l > 0:
				if not os.path.exists(os.path.join(path1, i)):
					os.makedirs(os.path.join(path1, i))
				shutil.copy(os.path.join(path, i, li), os.path.join(path1, i, li))
