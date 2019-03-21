# -*- coding: utf-8 -*-
import os 

class_name = []

for directory, folders, files in os.walk('/data/huang/behaviour/data/data_photos'):
    if len(folders) == 0:
        #if directory.split('/')[-1].split('_')[0] != '1':
        class_name.append([directory.split('/')[-1], len(files)])
class_name.sort()

with open('class_number.txt', 'w') as f:
    for i in class_name:
        f.write('{}:{}\n'.format(i[0],i[1]))
