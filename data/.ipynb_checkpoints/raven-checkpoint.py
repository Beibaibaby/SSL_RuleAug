# import os
# import random
# import glob
# import numpy as np
# import cv2

# import torch
# from torch.utils.data import Dataset



# class RAVEN(Dataset):
#     def __init__(
#         self, 
#         dataset_dir, 
#         data_split=None, 
#         image_size=80, 
#         transform=None, 
#         subset=None, 
#         num_samples=None,
#         ssl=False,
#     ):
#         self.dataset_dir = dataset_dir
#         self.data_split = data_split
#         self.image_size = image_size
#         self.transform = transform
#         self.ssl = ssl

#         subsets = os.listdir(self.dataset_dir)

#         self.file_names = []
#         for i in subsets:
#             file_names = [os.path.basename(f) for f in glob.glob(os.path.join(self.dataset_dir, i, "*_" + self.data_split + ".npz"))]
#             file_names.sort()
#             self.file_names += [os.path.join(i, f) for f in file_names]
        
        
#         if num_samples != None and data_split == "train":
#             permutation_indices = np.random.permutation(len(self.file_names))
            
#             if ssl:
#                 sel = permutation_indices[:num_samples]
#                 self.file_names_labeled = [self.file_names[s] for s in sel]
                
#                 sel = permutation_indices[num_samples:]
#                 self.file_names = [self.file_names[s] for s in sel]
#             else:
#                 sel = permutation_indices[:num_samples]
#                 self.file_names = [self.file_names[s] for s in sel]


#     def __len__(self):
#         return len(self.file_names)

#     def get_data(self, data_file):
#         data_path = os.path.join(self.dataset_dir, data_file)
#         data = np.load(data_path)

#         image = data["image"].reshape(16, 160, 160)
#         if self.image_size != 160:
#             resize_image = np.zeros((16, self.image_size, self.image_size))
#             for idx in range(0, 16):
#                 resize_image[idx] = cv2.resize(
#                     image[idx], (self.image_size, self.image_size), interpolation = cv2.INTER_NEAREST
#                 )
#         else:
#             resize_image = image

#         return resize_image, data, data_file
    
    
#     def __getitem_sl__(self, idx):
#         image, data, data_file = self.get_data(self.file_names[idx])

#         # Get additional data
#         target = data["target"]
#         meta_target = data["meta_target"]
#         structure = data["structure"]
#         structure_encoded = data["meta_matrix"]
#         del data

#         if self.transform:
#             image = self.transform(image)

#         # if self.permute:
#         #     new_target = random.choice(range(8))
#         #     if new_target != target:
#         #         resize_image[[8 + new_target, 8 + target]] = resize_image[[8 + target, 8 + new_target]]
#         #         target = new_target

#         target = torch.tensor(target, dtype=torch.long)
#         meta_target = torch.tensor(meta_target, dtype=torch.float32)
#         structure_encoded = torch.tensor(structure_encoded, dtype=torch.float32)
        
#         return image, target, meta_target, structure_encoded, data_file
    
    
#     def __getitem_ssl__(self, idx):
#         # unlabeled data
#         image_unlabeled, _, data_file = self.get_data(self.file_names[idx])
        
#         # labeled data and its other information
#         idx = idx % len(self.file_names_labeled)
#         image_labeled, data, data_file = self.get_data(self.file_names_labeled[idx])
            
#         target_labeled = data["target"]
#         # meta_target = data["meta_target"]
#         # structure = data["structure"]
#         # structure_encoded = data["meta_matrix"]
        
#         if self.transform:
#             image_unlabeled = self.transform(image_unlabeled)
#             image_labeled = self.transform(image_labeled)
            
#         # if self.permute:
#         #     new_target = random.choice(range(8))
#         #     if new_target != target:
#         #         resize_image[[8 + new_target, 8 + target]] = resize_image[[8 + target, 8 + new_target]]
#         #         target = new_target

#         target_labeled = torch.tensor(target_labeled, dtype=torch.long)
#         target_unlabeled = torch.tensor(-100, dtype=torch.long)
#         # meta_target = torch.tensor(meta_target, dtype=torch.float32)
#         # structure_encoded = torch.tensor(structure_encoded, dtype=torch.float32)
        
#         return image_labeled, target_labeled, image_unlabeled, target_unlabeled

#     def __getitem__(self, idx):
#         if self.ssl:
#             return self.__getitem_ssl__(idx)
#         else:
#             return self.__getitem_sl__(idx)



import os
import random
import glob
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset



class RAVEN(Dataset):
    def __init__(
        self, 
        dataset_dir, 
        data_split=None, 
        image_size=80, 
        transform=None, 
        subset=None, 
        num_samples=None,
        ssl=False,
    ):
        self.dataset_dir = dataset_dir
        self.data_split = data_split
        self.image_size = image_size
        self.transform = transform
        self.ssl = ssl

        subsets = os.listdir(self.dataset_dir)

        self.file_names = []
        for i in subsets:
            file_names = [os.path.basename(f) for f in glob.glob(os.path.join(self.dataset_dir, i, "*_" + self.data_split + ".npz"))]
            file_names.sort()
            self.file_names += [os.path.join(i, f) for f in file_names]
        
        
        if num_samples != None and data_split == "train":
            permutation_indices = np.random.permutation(len(self.file_names))
            
            if ssl:
                self.labeled_indices = permutation_indices[:num_samples]
                self.unlabeled_indices = permutation_indices[num_samples:]
                                
                self.labeled_indicators = torch.zeros(len(self.file_names), dtype=torch.bool)
                self.labeled_indicators[self.labeled_indices] = True
            else:
                sel = permutation_indices[:num_samples]
                self.file_names = [self.file_names[s] for s in sel]

    def __len__(self):
        return len(self.file_names)

    def get_data(self, data_file):
        data_path = os.path.join(self.dataset_dir, data_file)
        data = np.load(data_path)

        image = data["image"].reshape(16, 160, 160)
        if self.image_size != 160:
            resize_image = np.zeros((16, self.image_size, self.image_size))
            for idx in range(0, 16):
                resize_image[idx] = cv2.resize(
                    image[idx], (self.image_size, self.image_size), interpolation = cv2.INTER_NEAREST
                )
        else:
            resize_image = image

        return resize_image, data, data_file
    
    
    def __getitem__(self, idx):
        image, data, data_file = self.get_data(self.file_names[idx])

        # Get additional data
        target = data["target"]
        meta_target = data["meta_target"]
        structure = data["structure"]
        structure_encoded = data["meta_matrix"]
        del data

        if self.transform:
            image = self.transform(image)

        # if self.permute:
        #     new_target = random.choice(range(8))
        #     if new_target != target:
        #         resize_image[[8 + new_target, 8 + target]] = resize_image[[8 + target, 8 + new_target]]
        #         target = new_target

        target = torch.tensor(target, dtype=torch.long)
        meta_target = torch.tensor(meta_target, dtype=torch.float32)
        structure_encoded = torch.tensor(structure_encoded, dtype=torch.float32)
        
        if self.ssl:
            if self.labeled_indicators[idx]:
                return image, target
            else:
                return image, torch.tensor(-100, dtype=torch.long)
        else:
            return image, target, meta_target, structure_encoded, data_file