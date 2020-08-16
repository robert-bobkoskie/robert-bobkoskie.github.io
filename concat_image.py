#! /usr/bin/env python

import numpy as np
from PIL import Image

# im = np.array(Image.open('/home/rb868x/PROJECTS/python/AVAYA.png').resize((256, 256)))
im1 = Image.open('/home/rb868x/PROJECTS/python/AVAYA.png')
im2 = Image.open('/home/rb868x/PROJECTS/python/ATT.png')
im3 = Image.open('/home/rb868x/PROJECTS/python/TATA.png')
# print(im1.size, im2.size, im3.size)
# print(np.array(im1).shape)
# print(np.array(im2).shape)
# print(np.array(im3).shape)

img1_w, img1_h = im1.size
img2_w, img2_h = im2.size
img3_w, img3_h = im3.size

img2_scale_w = round(float(img1_h) / float(img2_h) * float(img2_w), 0)
img3_scale_w = round(float(img1_h) / float(img3_h) * float(img3_w), 0)
# print(img1_h, img2_scale_w)
# print(img1_h, img3_scale_w)

im2_resized = im2.resize((int(img2_scale_w), int(img1_h)))
im3_resized = im3.resize((int(img3_scale_w), int(img1_h)))
# print(im1.size, im2_resized.size, im3_resized.size)
# im2_resized.save('im2_resized.png')
# im3_resized.save('im3_resized.png')

concat_img = Image.new('RGB', (im1.width + im2_resized.width, im1.height))
concat_img.paste(im1, (0, 0))
concat_img.paste(im2_resized, (im1.width, 0))
# print(concat_img.size)
# concat_img.save('concat_img.png')

concat_img3 = Image.new('RGB', (concat_img.width + im3_resized.width, im1.height))
concat_img3.paste(concat_img, (0, 0))
concat_img3.paste(im3_resized, (concat_img.width, 0))
# print(concat_img3.size)
concat_img3.save('concat_img.png')

im = np.array(concat_img3)
im_hc = 255 - im
Image.fromarray(im_hc).save('im_hc.png')
