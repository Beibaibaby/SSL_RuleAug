import os
import random
import glob
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


class PGM(Dataset):
    # def __init__(self, root, cache_root, dataset_type=None, regime='neutral', image_size=80, transform=None,
    #              use_cache=False, save_cache=False, in_memory=False, subset=None, flip=False, permute=False):
    #     self.root = root
    #     self.cache_root = cache_root if cache_root is not None else root
    #     self.dataset_type = dataset_type
    #     self.regime = regime if regime is not None else 'neutral'
    #     print([self.dataset_type, self.regime])
    #     if self.dataset_type != 'train' and self.regime.startswith('augment'):
    #         self.regime = 'neutral'
    #     print([self.dataset_type, self.regime])
    #     self.image_size = image_size
    #     self.transform = transform
    #     self.use_cache = use_cache
    #     self.save_cache = save_cache
    #     self.flip = flip
    #     self.permute = permute

    def __init__(self, dataset_dir, data_split=None, image_size=80, transform=None, 
                 load_in_memory=False, subset=None, flip=False, permute=False):
        self.dataset_dir = dataset_dir
        self.data_split = data_split
        self.image_size = image_size
        self.transform = transform
        self.flip = flip
        self.permute = permute

        # def set_paths():
        #     if self.root is not None:
        #         if os.path.isdir(os.path.join(self.root, 'data')):
        #             self.data_dir = os.path.join(self.root, 'data', self.regime)
        #         else:
        #             self.data_dir = os.path.join(self.root, self.regime)
        #     else:
        #         self.data_dir = None
        #     if self.use_cache:
        #         self.cached_dir = os.path.join(self.cache_root, 'cache', self.regime,
        #                                        f'{self.dataset_type}_{self.image_size}')
        # set_paths()

        # if subset is not None:
        #     position_file_names_place = os.path.join('files', 'pgm', f'{subset}_{dataset_type}.txt')
        #     assert os.path.isfile(position_file_names_place), f'Subset file {position_file_names_place} not found'
        #     with open(position_file_names_place, "r") as file:
        #         contents = file.read()
        #         self.file_names = contents.splitlines()

        #     self.file_names = [os.path.basename(f) for f in self.file_names]
        # else:
        # data_dir = self.dataset_dir if self.dataset_dir is not None else self.cached_dir

        # self.file_names = [f for f in os.listdir(self.dataset_dir) if self.data_split in f]
        self.file_names = [os.path.basename(f) for f in glob.glob(os.path.join(self.dataset_dir, "*_" + self.data_split + "_*.npz"))]
        self.file_names.sort()

        # Sanity
        assert subset != 'train' or len(self.file_names) == 1200000, f'Train length = {len(self.file_names)}'
        assert subset != 'val' or len(self.file_names) == 20000, f'Validation length = {len(self.file_names)}'
        assert subset != 'test' or len(self.file_names) == 200000, f'Test length = {len(self.file_names)}'

        # print(f'Dataset {self.dataset_type} size {len(self.file_names)} ')

        
        self.memory = None
        if load_in_memory:
            self.load_all_data()

    def load_all_data(self):
        if os.path.exists(self.dataset_dir + "/" + self.data_split + str(self.image_size) + "_.npz"):
            print("Loading %s cache into memory" % self.data_split)
            loader = np.load(self.dataset_dir + "/" + self.data_split + "_.npz", allow_pickle=True)
            self.memory = loader['data']
        else:
            self.memory = [None] * len(self.file_names)
            from tqdm import tqdm
            for idx in tqdm(range(len(self.file_names)), 'Loading into memory'):
                image, data, _ = self.get_data(idx)
                d = {'target': data["target"],
                     'meta_target': data["meta_target"],
                     'structure': data["structure"],
                     'meta_structure': data["meta_structure"],
                     'meta_matrix': data["meta_matrix"]
                    }
                self.memory[idx] = (image, d)
                del data

            np.savez_compressed(self.dataset_dir + "/" + self.data_split + str(self.image_size), data=self.memory)

        
        # self.memory = None
        # if in_memory:
        #     self.memory = [None] * len(self.file_names)
        #     from tqdm import tqdm
        #     for idx in tqdm(range(len(self.file_names)), 'Loading Memory'):
        #         image, data, _ = self.get_data(idx)
        #         d = {'target': data["target"],
        #              'meta_target': data["meta_target"],
        #              'relation_structure': data["relation_structure"],
        #              'relation_structure_encoded': data["relation_structure_encoded"]
        #              }
        #         self.memory[idx] = (image, d)
        #         del data

        

    # def save_image(self, image, file):
    #     image = image.numpy()
    #     os.makedirs(os.path.dirname(file), exist_ok=True)
    #     image_file = os.path.splitext(file)[0] + '.png'
    #     skimage.io.imsave(image_file, image.reshape(self.image_size, self.image_size))

    # def load_image(self, file):
    #     image_file = os.path.splitext(file)[0] + '.png'
    #     gen_image = skimage.io.imread(image_file).reshape(1, self.image_size, self.image_size)
    #     if self.transform:
    #         gen_image = self.transform(gen_image)
    #     gen_image = to_tensor(gen_image)
    #     return gen_image

    # def load_cached_file(self, file):
    #     try:
    #         data = np.load(file)
    #         image = data['image']
    #         return image, data
    #     except:
    #         raise ValueError(f'Error - Could not open existing file {file}')

    # def save_cached_file(self, file, image, data):
    #     os.makedirs(os.path.dirname(file), exist_ok=True)
    #     data['image'] = image
    #     np.savez_compressed(file, **data)

    def __len__(self):
        return len(self.file_names)

    def get_data(self, idx):
        data_file = self.file_names[idx]
        if self.memory is not None and self.memory[idx] is not None:
            resize_image, data = self.memory[idx]
        else:
            data_path = os.path.join(self.dataset_dir, data_file)
            data = np.load(data_path)

            image = data["image"].reshape(16, 160, 160)
            if self.image_size != 160:
                resize_image = np.zeros((16, self.image_size, self.image_size))
                for idx in range(0, 16):
                    resize_image[idx] = cv2.resize(
                        image[idx], (self.image_size, self.image_size), interpolation = cv2.INTER_NEAREST
                    )
                # resize_image = resize_image.astype(np.uint8)
            else:
                resize_image = image
                # resize_image = image.astype(np.uint8)

            # print(resize_image.shape)

        return resize_image, data, data_file

    def __getitem__(self, idx):
        resize_image, data, data_file = self.get_data(idx)

        # Get additional data
        target = data["target"]
        meta_target = data["meta_target"]
        structure_encoded = data["relation_structure_encoded"]
        del data

        if self.transform:
            resize_image = self.transform(resize_image)
        # resize_image = to_tensor(resize_image)

        if self.flip:
            if random.random() > 0.5:
                resize_image[[0, 1, 2, 3, 4, 5, 6, 7]] = resize_image[[0, 3, 6, 1, 4, 7, 2, 5]]

        if self.permute:
            new_target = random.choice(range(8))
            if new_target != target:
                resize_image[[8 + new_target, 8 + target]] = resize_image[[8 + target, 8 + new_target]]
                target = new_target

        target = torch.tensor(target, dtype=torch.long)
        meta_target = torch.tensor(meta_target, dtype=torch.float32)
        structure_encoded = torch.tensor(structure_encoded, dtype=torch.float32)

        return resize_image, target, meta_target, structure_encoded, data_file
