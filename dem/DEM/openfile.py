import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.animation as animation
import time
import scipy.io
#image = scipy.io.loadmat('bolder/data_0.mat')
#image = scipy.io.loadmat('terrain_generation/perlin_bolder/data_10.mat')
#image = scipy.io.loadmat('two_craters_label/data_10.mat')
#image = scipy.io.loadmat('64pix_(0deg)_dem(noisy)_ver2\image\image(t-70)_1178')
#image = scipy.io.loadmat(r'64pix_(0deg)_dem(noisy)_evaluate_1124\5deg\image\image_1.mat')
#image = scipy.io.loadmat(r'64pix_(0-3deg)_dem(lidar_noisy)_boulder\image(t-10)\image_52.mat')
image = scipy.io.loadmat(r'C:\Users\aki\Documents\GitHub\deep\DEM\simple_crater\image\data_5.mat')

#print(type(image))
#print(image.keys())
#print(image.values())
#print(image.items())
#print(image['time_data'])

print(image['time_data'].shape)


fig, ax = plt.subplots()
N = 11
def update(i):
    if i == 1:
        print("===========")
    img = image['time_data'][:,:,i]
    #print(i)
    
    plt.clf()
    
    plt.imshow(img,cmap='gray')
hoge = animation.FuncAnimation(fig, update, np.arange(1,  N), interval=10)  # 代入しないと消される
hoge.save('20_anim.gif', writer='PillowWriter')
plt.show()

"""
### ラベル可視化
np.set_printoptions(threshold=10000000)
label =  scipy.io.loadmat('64pix_(0deg)_dem(noisy)/label/label_151.mat')
print(label.keys())
print(label.values())
print(label.items())
label_img = label['label_data']
print(label_img)
plt.imshow(label_img)
plt.show()


print(image['image'])
print(type(image['image']))
print(image['image'][:,:,1])
img = image['image'][:,:,1]
N = 100
fig, ax = plt.subplots()

def update(i):
    img = image['image'][:,:,i]
    
    plt.clf()
    
    plt.imshow(img,cmap='gray')
hoge = animation.FuncAnimation(fig, update, np.arange(1,  N), interval=25)  # 代入しないと消される
plt.show()
"""