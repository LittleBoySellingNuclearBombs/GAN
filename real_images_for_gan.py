import numpy as np
import cv2
import json
IMGs=[]
# D:\Develop\JetBrains\PycharmProjects\GAN_DATA\extra_data\images is the path of data
# the data u can download form https://drive.google.com/file/d/1tpW7ZVNosXsIAWu8-f5EpwtF3ls3pb79/view
# reference https://github.com/CodePlay2016/GAN_learn/tree/master/HW3

for i in range(36740):
    img=cv2.imread('D:/Develop/JetBrains/PycharmProjects/GAN_DATA/extra_data/images/{}.jpg'.format(i)) 
    img=cv2.resize(img, (64,64), interpolation=cv2.INTER_CUBIC) 
    IMGs.append(img)
IMGs=np.array(IMGs)
np.save('real_images.npy', IMGs)
