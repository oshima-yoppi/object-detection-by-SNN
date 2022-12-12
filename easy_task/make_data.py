import numpy as np
import cv2
import torch 
import h5py
import pandas as pd
import torchvision
import random
import pandas as pd
import os, shutil


def draw_circle(center, radius,time=10, pixel=128):
    img = np.zeros((pixel, pixel, 3), dtype=np.uint8)
    cv2.circle(img, center, radius, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_8, shift=0)
    return img
def youtube(events, path):
    images = []
    events = events.to(torch.uint8)
    events *= 255
    for i in range(time):
        p_ = torchvision.transforms.functional.to_pil_image(events[i,:,:])
        images.append(p_)
    images[0].save(path, duration = 100, save_all=True, append_images=images[1:], loop = 50)


def SaveEvents(path, events, label):
    with h5py.File(path, 'w') as f:
        f.create_dataset('label', data = label)
        f.create_dataset('input', data = events)
    return


if __name__ == "__main__":
    pixel = 64
    time = 10
    noize_rate = 0.1
    number_of_data = 300
    numer_list = []
    radius_list, x_list, y_list = [],[],[]
    label_list = []
    path_list = []
    youtube_dir = "youtube"
    dataset_dir = "dataset"
    csv_path = f"dataset/info.csv"

    if os.path.exists(youtube_dir):
        shutil.rmtree(youtube_dir)
        os.makedirs(youtube_dir)
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
        os.makedirs(dataset_dir)
    for i in range(number_of_data):
        x, y, r, label = None, None, None, 0
        youtube_path = os.path.join(youtube_dir, f"{str(i).zfill(4)}.gif")
        dataset_path = os.path.join(dataset_dir, f"{str(i).zfill(4)}.h5")


        events = torch.where(torch.rand(time, pixel, pixel)<= noize_rate, 1, 0)
        if random.random() <= 0.5:
            x = random.randint(0,pixel)
            y = random.randint(0, pixel)
            r = random.randint(4, pixel//2)
            label = 1
            for t in range(time):
                img = draw_circle((x, y), r + t, pixel=pixel)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = torch.from_numpy(img.astype(np.float32)).clone()
                events[t,:,:] = torch.logical_or(img, events[t,:,:])
                events[t,:,:] = torch.where(events[t,:,:] == True, 1, 0)
        
        youtube(events, youtube_path)
        SaveEvents(dataset_path, events, label)
        numer_list.append(i)
        radius_list.append(r)
        x_list.append(x)
        y_list.append(y)
        label_list.append(label)
        path_list.append(dataset_path)
    
    df = pd.DataFrame(
        data={
            'number':numer_list,
            'radius':radius_list,
            'x':x_list,
            'y':y_list,
            'label':label_list,
            'path':path_list
        }
    )
    df.to_csv(csv_path, index=False)
    
        
