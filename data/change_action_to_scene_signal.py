import os


for directory, folders, files in os.walk('/data/huang/behaviour/data/data_photos_left'):
    for folder in folders:
        os.rename(folder, '{}_{}'.format(1,folder))
    break

