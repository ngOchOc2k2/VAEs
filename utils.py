import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import random
import numpy as np
import imageio
import torch
from PIL import Image

DATASET_PATH ="/home/luungoc/Paper/TrainLAB/VAEs/lfw-deepfunneled/lfw-deepfunneled/"
ATTRIBUTES_PATH = "/home/luungoc/Paper/TrainLAB/VAEs/lfw_attributes.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Training on',DEVICE)


def explore_data(num_head : int):
    dataset = []
    for path in glob.iglob(os.path.join(DATASET_PATH, "**", "*.jpg")):
        person = path.split("/")[-2]
        dataset.append({"person":person, "path": path})
        
    dataset = pd.DataFrame(dataset)
    dataset = dataset.groupby("person").filter(lambda x: len(x) < 25 )
    return dataset.head(num_head)



def show_image(dataset):
    plt.figure(figsize=(20,10))
    for i in range(20):
        idx = random.randint(0, len(dataset))
        img = plt.imread(dataset.path.iloc[idx])
        plt.subplot(4, 5, i+1)
        plt.imshow(img)
        plt.title(dataset.person.iloc[idx])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()



def fetch_dataset(dx=80, dy=80, dimx=45,dimy=45):
    df_attrs = pd.read_csv(ATTRIBUTES_PATH, sep='\t', skiprows=1,) 
    df_attrs = pd.DataFrame(df_attrs.iloc[:,:-1].values, columns = df_attrs.columns[1:])
    
    photo_ids = []
    for dirpath, dirnames, filenames in os.walk(DATASET_PATH):
        for fname in filenames:
            if fname.endswith(".jpg"):
                fpath = os.path.join(dirpath,fname)
                photo_id = fname[:-4].replace('_',' ').split()
                person_id = ' '.join(photo_id[:-1])
                photo_number = int(photo_id[-1])
                photo_ids.append({'person':person_id,'imagenum':photo_number,'photo_path':fpath})

    photo_ids = pd.DataFrame(photo_ids)
    df = pd.merge(df_attrs,photo_ids,on=('person','imagenum'))

    assert len(df)==len(df_attrs),"lost some data when merging dataframes"
    
    all_photos = df['photo_path'].apply(imageio.imread)\
                                .apply(lambda img:img[dy:-dy,dx:-dx])\
                                .apply(lambda img: np.array(Image.fromarray(img).resize([dimx,dimy])) )

    all_photos = np.stack(all_photos.values).astype('uint8')
    all_attrs = df.drop(["photo_path","person","imagenum"],axis=1)
    
    return all_photos,all_attrs
