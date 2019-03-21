import os

xml = []
signal = 0
for directory, folders, files in os.walk('/data/huang/behaviour/data/temp'):
    for file in files:
        if os.path.splitext(file)[1] in ['.xml', '.XML']:
            xml.append(os.path.join(directory, file))
for item in xml:
    os.remove(item)
        
    