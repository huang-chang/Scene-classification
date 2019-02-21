# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 19:00:49 2019

@author: universe
"""

label_path = '/home/universe/Desktop/label_371_12_20.txt'
label_name = []
signal = 1

with open(label_path, 'r') as f:
    for i in f.readlines():
        if i.strip().split(':')[-1] == '-' and i.strip().split(':')[1].split('_')[0] not in ['others']:
            print(signal, i.strip())
            signal += 1
            
        