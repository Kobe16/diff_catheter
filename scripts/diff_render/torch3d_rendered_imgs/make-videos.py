import os
import sys
import numpy as np
import imageio
import argparse
import scipy.misc
import matplotlib.pyplot
import imageio
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', default='')
parser.add_argument('--st_idx', type=int, default=0)
parser.add_argument('--ed_idx', type=int, default=99)
parser.add_argument('--height', type=int, default=480)
parser.add_argument('--width', type=int, default=640)

args = parser.parse_args()
st_x = 1
ed_x = 100

images = []

num = 0


writer = imageio.get_writer('diff_curve_ok1.gif', fps=10)
for i in range(st_x, ed_x, 1):

    filename = os.path.join(args.src_dir, 'render_'+'%d.jpg' % i)
    print(filename)
    img = imageio.imread(filename)

    #images.append(img)
    writer.append_data(img)
    num = num + 1

print("total frame num: ", num)
writer.close()



#imageio.mimsave(args.src_dir + 'case1_sim_end_to_emit_render.mp4', images, duration=10/num)



