import time
import shutil
import os


def get_model_path():
    model_list = []
    for directory, folders, files in os.walk('/data/huang/behaviour/data/tfmodel'):
        for file in files:
            if file.split('.')[0] =='model':
                model_list.append(os.path.join(directory, file))
        break
    return model_list
    


signal = 0
while(1):
    write_time = time.ctime().split(' ')[3].split(':')
    if write_time[0] == '14' and write_time[1] == '00' and write_time[2] == '00':
        model_path = get_model_path()
        for path in model_path:
            shutil.copy(path, '/data/huang/behaviour/data/tfmodel/model_backup')
            
    if write_time[0] == '19' and write_time[1] == '00' and write_time[2] == '00':
        model_path = get_model_path()
        for path in model_path:
            shutil.copy(path, '/data/huang/behaviour/data/tfmodel/model_backup')
            
    if write_time[0] == '00' and write_time[1] == '00' and write_time[2] == '00':
        model_path = get_model_path()
        for path in model_path:
            shutil.copy(path, '/data/huang/behaviour/data/tfmodel/model_backup')
            
#    if write_time[0] == '11' and write_time[1] == '00' and write_time[2] == '00':
#        model_path = get_model_path()
#        for path in model_path:
#            shutil.copy(path, '/data/huang/behaviour/data/tfmodel/model_backup')
        
    if write_time[0] == '05' and write_time[1] == '00' and write_time[2] == '00':
        model_path = get_model_path()
        for path in model_path:
            shutil.copy(path, '/data/huang/behaviour/data/tfmodel/model_backup')
            
        break
    

        
