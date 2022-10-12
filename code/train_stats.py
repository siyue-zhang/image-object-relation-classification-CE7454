from PIL import Image, ImageFile
import io
from os import walk
import numpy as np 


folder = './data/coco/train2017'

R = []
G = []
B = []

f = []
for (dirpath, dirnames, filenames) in walk(folder):
    f.append(filenames)
    break

for i,x in enumerate(f[0]):
    path = folder+'/'+x

    with open(path, 'rb') as f:
        content = f.read()
        filebytes = content
        buff = io.BytesIO(filebytes)
        image = Image.open(buff).convert('RGB')
        image = np.array(image)
        R.extend(list(image[0].flatten()/255))
        G.extend(list(image[1].flatten()/255))
        B.extend(list(image[2].flatten()/255))

# Original array
for x in [R,G,B]:
    array = np.array(x)
    r1 = np.mean(array)
    print("\nMean: ", r1)
  
    r2 = np.std(array)
    print("\nstd: ", r2)


