# -*- coding: utf-8 -*-
import os 

action_name = []

for directory, folders, files in os.walk('data_photos'):
    action_name.extend(folders)
    action_name.sort()
    with open('name.txt', 'w') as f:
        for index, item in enumerate(action_name):
            f.write('{}:{}:\n'.format(index, item))
    break
