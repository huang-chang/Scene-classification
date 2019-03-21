import os
import cv2

path = '/data/huang/behaviour/data/data_photos'
image_path = []
format_set = set()

for directory, folders, files in os.walk(path):
    print(directory)
    if len(files) > 0:
        for file in files:
            image_path.append(os.path.join(directory,file))
            
for index, item in enumerate(image_path):
    img = cv2.imread(item)
    os.remove(item)
    dirname = os.path.dirname(item)
    basename = os.path.basename(item)
    filename = os.path.splitext(basename)[0]
    new_img = '{}.jpg'.format(os.path.join(dirname,filename))
    cv2.imwrite(new_img,img,[int(cv2.IMWRITE_JPEG_QUALITY),100])
    if index % 1000 == 0:
        print(index)
