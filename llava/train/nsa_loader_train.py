import os
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from nsa import NSA_transform, concatenate_images_side_by_side


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



"""
The shape of datadict
  {
    "id": "i",
    "image": image_obj,
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nThe corresponding images are arranged in the form of four. The left three represent normal sample images and the rightmost one represents the test image. Compared to the three sample images, is there an anomaly in the test image? Answer 0 if not, and 1 if there is."
      },
      {
        "from": "gpt",
        "value": "0"
      },
    ]
  }
"""

prompt =  "<image>\nThe corresponding images are arranged in the form of four. The left three represent normal sample images and the rightmost one represents the test image. Compared to the three sample images, is there an anomaly in the test image? Answer 0 if not, and 1 if there is."

class MVTecConcatSyntheticAnomaly(Dataset):
    def __init__(self, categories, data_path, anomaly_fn, K=4, transform=None, grid_size=2):
        """
        Read MVTec training set, and then randomly concat K images within a object category.
        The rightmost image is randomly transformed using "anomaly_fn".
        
        categories: a list containing mvtec object categories
        data_path: path to the mvtec dataset
        K: the number of images in a single instance"""
        self.categories = categories
        self.data_path = data_path
        self.transform = transform
        self.anomaly_fn = anomaly_fn
        self.grid_size = grid_size
        # prepare image paths

        l_tuple_img = []  # contains image paths
        l_tuple_label = []
        for cat in self.categories:
            # read normal image file list
            train_dir = os.path.join(data_path, cat, "train", "good")
            l_train_img = os.listdir(train_dir)
            l_train_img = [os.path.join(train_dir, img) for img in l_train_img]

            # random shuffle
            np.random.shuffle(l_train_img)
            
            # group by K images. drop last ones
            for i in range(len(l_train_img)//K):
                l_tuple_img.append(l_train_img[i*K:i*K+K])

            # randomly assign label
            l_tuple_label += list(np.random.randint(0, 2, len(l_tuple_img)))

        self.l_tuple_img = l_tuple_img
        self.l_tuple_label = l_tuple_label
 
    def create_datadict(self, i, catimg, mask):
        datadict = {}
        datadict["id"] = f"{i}"
        datadict["image"] = catimg
        datadict["conversations"] = [{
                                       "from": "human",
                                       "value": prompt
                                     },
                                     {
                                        "from": "gpt",
                                        "value": f"{mask}"
                                     }]
        return datadict
    
    def __getitem__(self, i):
        l_img_path, label = self.l_tuple_img[i], self.l_tuple_label[i]
        images = [Image.open(path) for path in l_img_path]
        if label == 1:  # synthetic anomaly
            # do anomaly
            category = [word for word in self.categories if word in l_img_path[0]]
            images[-1], mask = self.anomaly_fn[category[0]].transform(images[-1])
            mask = 1
            
        else:
            mask = 0
        
        catimg = concatenate_images_side_by_side(images)
        if self.transform is not None:
            catimg = self.transform(catimg)
        return self.create_datadict(i, catimg, mask)

    def __len__(self):
        return len(self.l_tuple_img)


def get_nsa_dataset():
    nsa_by_categories = {category: NSA_transform(data_dir=data_path, class_name=category) for category in categoriesA}
    ds = MVTecConcatSyntheticAnomaly(categoriesA, data_path=data_path, anomaly_fn=nsa_by_categories, transform=None)

    return ds


if __name__ == '__main__':
    ds = get_nsa_dataset()
    print(ds["conversations"])
