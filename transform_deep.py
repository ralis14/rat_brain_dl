from PIL import Image
import pylab as plt
import numpy as np
import os

WIDTH = 200
HEIGHT = 300

# Read Swanson's images and store them using Numpy with compression
# atlas_sw = np.empty((73, HEIGHT, WIDTH), dtype=np.uint8)
# atlas_sw_labels = np.empty(73, dtype=np.uint8)
# for plate in range(0, 73):
#     filename = 'Level-' + str(plate+1).zfill(2) + '.jpg'
#     filename = os.path.join('atlas_sw', filename)
#
#     if not os.path.exists(filename):
#         print("Couldn't find ", filename, ", skipping")
#         continue
#
#     # Grayscale conversion and resizing
#     im = Image.open(filename).convert("L")
#     im = im.resize((WIDTH, HEIGHT))
#     im = np.array(im, dtype=np.uint8)
#
#     atlas_sw[plate] = im
#     atlas_sw_labels[plate] = (plate + 1)
#
# np.savez_compressed('atlas_sw', images=atlas_sw, labels=atlas_sw_labels)

# 89 of 161 use Nissl
# atlas_pw = np.empty((89, HEIGHT, WIDTH), dtype=np.uint8)
# atlas_pw_labels = np.empty(89, dtype=np.uint8)
# index = 0
# for plate in range(0, 161):
#     filename = 'RBSC7-' + str(plate+1).zfill(3) + '.jpg'
#     filename = os.path.join('atlas_pw', filename)
#
#     if not os.path.exists(filename):
#          print("Couldn't find ", filename, ", skipping")
#          continue
#
#     im = Image.open(filename).convert("L")
#
#     # Split them in half
#     box = (0, 0, im.width / 2, im.height)
#     im = im.crop(box)
#     # Resize
#     im = im.resize((WIDTH, HEIGHT))
#     im = np.array(im, dtype=np.uint8)
#
#     atlas_pw[index] = im
#     atlas_pw_labels[index] = (plate + 1)
#     index += 1
#
# np.savez_compressed('atlas_pw', images=atlas_pw, labels=atlas_pw_labels)
#
# with np.load('atlas_pw.npz') as data:
#     images = data['images']
#     labels = data['labels']
#
#     plt.imshow(images[25])
#     print(labels[25])

print("Datasets processed")
