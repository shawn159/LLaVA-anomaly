import os
import sys
from tqdm import tqdm
from PIL import Image
import numpy as np


sys.path.insert(1, "/root/LLaVA-anomaly/llava/train")
from nsa import concatenate_images_side_by_side


data_path = "/data/anomaly/mvtec/"

# for training
categoriesA = [
     "bottle",
     "cable",
     "capsule",
     "carpet",
     "grid",
     "hazelnut",
     "leather"
]

#categoriesA = [
#    "bottle"
#]

# for testing only (unseen classes)
categoreisB = [
    "cable",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


def get_nsa_dataset(categories=categoriesA, train_ids=[1,2,3]):
    l_d = []
    for category in tqdm(categories):
        
        # set paths
        train_path = f'{data_path}/{category}/train/good/'
        test_dir = os.path.join(data_path, f'{category}', 'test')
        l_class = os.listdir(test_dir)
        assert 'good' in l_class
        l_normal_img_path = [os.path.join(test_dir, 'good', s) for s in os.listdir(os.path.join(test_dir, 'good'))]

        l_anomaly_class = [c for c in l_class if c != 'good']
        l_anomaly_img_path = []
        for c in l_anomaly_class:
            l_anomaly_img_path += [os.path.join(test_dir, c, s) for s in os.listdir(os.path.join(test_dir, c))]

        l_test_img_path = l_normal_img_path + l_anomaly_img_path
        l_test_img_label = np.concatenate([np.zeros(len(l_normal_img_path)), np.ones(len(l_anomaly_img_path),)])
        
        
        # set training images
        train_images = [train_path + f'{i:03d}.png' for i in train_ids]
        
        # actual gpt run
        d = {'train_image_ids':[], 'test_image_class': [], 'test_image_id':[], 
                'label':[], 'logit': [], 'category': [], 'image': []}
        for test_image_path in tqdm(l_test_img_path):
            images = [Image.open(path) for path in train_images + [test_image_path]]
            input_image = concatenate_images_side_by_side(images)

            d['image'].append(input_image)
            d['train_image_ids'].append(train_ids)
            test_cls, test_id = test_image_path.split('/')[-2], int(test_image_path.split('/')[-1].strip('.png'))
            d['test_image_class'].append(test_cls)
            d['test_image_id'].append(test_id)
            d['category'].append(category)
            if test_cls == 'good':
                d['label'].append(0)
            else:
                d['label'].append(1)

        l_d.append(d)
    return l_d


if __name__ == '__main__':
    get_nsa_dataset()
