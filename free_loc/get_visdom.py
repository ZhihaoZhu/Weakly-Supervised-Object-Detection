
from __future__ import print_function


from PIL import Image

import numpy as np

import os
import visdom
import numpy as np
vis = visdom.Visdom(server='http://localhost',port='8097')
currentDirectory = os.getcwd()


for i in range(11, 15):
        img_path =   os.path.join(currentDirectory, "free_loc/tmp/f{fname}.png".format(fname = str(i)))
        img = Image.open(img_path)
        img = np.array(img)
        img = img.transpose(2,0,1)

        vis.image(img, opts = dict(title="index_{}".format(i)))

for i in range(18, 14, -1):
        img_path =   os.path.join(currentDirectory, "free_loc/tmp/f{fname}.png".format(fname = str(i)))
        img = Image.open(img_path)
        img = np.array(img)
        img = img.transpose(2,0,1)

        vis.image(img, opts = dict(title="index_{}".format(i)))
