import os
import json
from sklearn.model_selection import train_test_split

root = os.path.dirname(__file__)
images_dir = os.path.join(root, 'images')
all_imgs = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

train, val = train_test_split(all_imgs, test_size=0.2, random_state=42)

def make_dict(img_list):
    # data format: {img_path: []} (empty list of boxes)
    d = {}
    for p in img_list:
        d[p] = []
    return d

with open(os.path.join(root, 'train.json'), 'w') as f:
    json.dump(make_dict(train), f)

with open(os.path.join(root, 'val.json'), 'w') as f:
    json.dump(make_dict(val), f)

print('Created train.json and val.json with', len(train), 'train and', len(val), 'val images')
