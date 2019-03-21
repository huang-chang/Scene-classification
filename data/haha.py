#import os
#
#bad_path = []
#number = 0
#for directory, folders, files in os.walk('/data/huang/behaviour/data/data_photos'):
#    for file in files:
#        number += 1
#        print(number, os.path.splitext(file)[1])
#        if os.path.splitext(file)[1] not in ['.jpg','.jpeg']:
#            number += 1
#            bad_path.append(os.path.join(directory, file))
#
#for index, item in enumerate(bad_path):
#    print(index, item)
    #os.remove(item)
import numpy as np
a = [1,2,3,4,5]
print(np.mean(a))