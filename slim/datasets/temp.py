import os

image_number = 0
for directory, folders, files in os.walk('/data/huang/behaviour/data/data_photos'):
    if len(folders) > 1:
        class_number = len(folders)
        
    if len(files) > 0:
        image_number += len(files)


print(class_number)
print(image_number)
