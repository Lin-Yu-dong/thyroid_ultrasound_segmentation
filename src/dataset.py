import os
import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None, return_id=False): 
        
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.return_id = return_id 
        
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
            return_id (bool, optional): Whether to return image ID in __getitem__. Defaults to False.
        
        Note:
            Make sure you have 6 folders:
            <dataset name>
            ├── trainging image
            ├── trainging mask
            ├── validation image
            ├── validation mask
            ├── test image
            └── test mask
        """

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        # print(img.shape)

        mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext) 
        if not os.path.exists(mask_path):
            print(f" Mask file not found: {mask_path}")
            raise IndexError(f"Missing mask: {mask_path}")
        
        mask = cv2.imread(mask_path)
        if mask is None:
            print(f" Failed to read mask: {mask_path}")
            raise IndexError(f"Unreadable mask: {mask_path}")
        
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)                 

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255

        if self.return_id: 
            return img, mask, {'img_id': img_id}
        else:
            return img, mask