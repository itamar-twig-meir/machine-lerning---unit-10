import numpy as np
from numpy.ma.core import reshape
from PIL import Image
import matplotlib.pyplot as plt
import time



#region red - green resources
raccoon = Image.open(r'C:/Users/miche/PycharmProjects/Raccoon.png')
plt.imshow(raccoon)
Array = np.array(raccoon)
#endregion

# region no vectorization red - green
tic = time.time()
Red_value_array = []
Green_value_array = []
Blue_value_array = []

for height in Array:
    for width in height:
        Red_value_array.append(width[0])
        Green_value_array.append(width[1])
        Blue_value_array.append(width[2])


green_score = 0
red_score = 0

for i in range(0, len(Red_value_array), 1):
    if Red_value_array[i] > Green_value_array[i] :
        red_score += 1
         # print(red_score)
    elif Red_value_array[i] < Green_value_array[i] :
        green_score += 1
        # print(green_score)

toc = time.time()
print ("Non Vectorized version: red = " + str(red_score) + ", green = " + str(green_score) + ". It took " + str(1000*(toc-tic)) + "ms")
# endregion red

# region vectorization red - green
tic = time.time()
red_array = Array[:,:,0]
green_array = Array[:,:,1]
red_score2 = np.sum(red_array > green_array)
green_score2 = np.sum(green_array > red_array)
toc = time.time()
print ("Vectorized version: red = " + str(red_score2) + ", green = " + str(green_score2) + ". It took " + str(1000*(toc-tic)) + "ms")
# endregion


# region picture flattening
array = np.array([[[1,2,3],[4,5,7],[8,9,10]], [[11,12,13],[14,15,16],[17,18,19]]])

flattened_array = array.flatten()
reshaped_array = reshape(array,-1, order='A')
print(flattened_array)
print(reshaped_array)

# endregion
# numpy flattening

