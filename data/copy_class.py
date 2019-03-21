import shutil
class_path = []
string = '/data/huang/behaviour/data/data_photos_backup_5_21'
with open('class_number_1.txt','r') as f:
    for i in f.readlines():
        class_path.append(string + '/' + i.strip().split(':')[1])
        
copy_path = '/data/huang/behaviour/data/temp/'

for i in class_path:
    copy_path_temp = copy_path + i.split('/')[-1]
    try:
        shutil.copytree(i, copy_path_temp)
    except:
        pass
    
