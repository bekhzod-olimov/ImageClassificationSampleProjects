import os
import torch
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import random_split, Dataset, DataLoader

torch.manual_seed(2025)

class CustomDataset(Dataset):
    def __init__(self, data_turgan_yolak, ds_nomi, tfs=None, im_files=[".png", ".jpg", ".jpeg", ".bmp"]):
        
        self.tfs, self.ds_nomi = tfs, ds_nomi
        self.data_turgan_yolak = data_turgan_yolak 
        self.im_files          = im_files      

        self.get_root(); self.get_files(); self.get_info()                

    def get_root(self): 
        if self.ds_nomi == "pet_disease": self.root = f"{self.data_turgan_yolak}/{self.ds_nomi}/{self.ds_nomi}/data"
        if self.ds_nomi == "geo_scene": self.root = f"{self.data_turgan_yolak}/{self.ds_nomi}/{self.ds_nomi}/GeoSceneNet16K"
        if self.ds_nomi == "lentils": self.root = f"{self.data_turgan_yolak}/{self.ds_nomi}/lentils/data"        
        elif self.ds_nomi == "rice_leaf_disease": self.root = f"{self.data_turgan_yolak}/{self.ds_nomi}/rice_leaf_disease/Rice Leaf  Disease Dataset"
        elif self.ds_nomi == "car_brands": self.root = f"{self.data_turgan_yolak}/{self.ds_nomi}/car_brands"
        elif self.ds_nomi == "dog_breeds": self.root = f"{self.data_turgan_yolak}/{self.ds_nomi}/dog_breeds/Dog Breed Classification"        
        elif self.ds_nomi == "apple_disease": self.root = f"{self.data_turgan_yolak}/{self.ds_nomi}/{self.ds_nomi}/{self.ds_nomi}/images"
    
    def get_files(self): 
        if self.ds_nomi in ["dog_breeds"]: self.im_paths = [path for im_file in self.im_files for path in glob(f"{self.root}/*/*/*{im_file}")]        
        elif self.ds_nomi in ["lentils", "apple_disease"]: self.im_paths = [path for im_file in self.im_files for path in glob(f"{self.root}/*{im_file}")]
        else: self.im_paths = [path for im_file in self.im_files for path in glob(f"{self.root}/*/*{im_file}")]        

    def get_info(self):

        self.cls_names, self.cls_counts = {}, {}
        count = 0
        for im_path in self.im_paths:
            class_name = self.get_class(im_path)
            if class_name not in self.cls_names:
                self.cls_names[class_name] = count
                self.cls_counts[class_name] = 1
                count += 1
            else: self.cls_counts[class_name] += 1
    
    def get_class(self, path): 
        if self.ds_nomi in ["lentils", "apple_disease"]: return os.path.basename(path).split("_")[0]
        else: return os.path.dirname(path).split("/")[-1]

    def __len__(self): return len(self.im_paths)

    def __getitem__(self, idx):
        
        im_path = self.im_paths[idx]
        im = Image.open(im_path)
        if im.mode != "RGB": im = im.convert("RGB")        
        gt = self.cls_names[self.get_class(im_path)]

        if self.tfs: im = self.tfs(im)

        return im, gt

    @classmethod
    def get_dls(cls, data_turgan_yolak, ds_nomi, tfs, bs, split=[0.8, 0.1, 0.1], ns=4):
        
        ds = cls(data_turgan_yolak=data_turgan_yolak, ds_nomi=ds_nomi, tfs=tfs)

        total_len = len(ds)
        tr_len = int(total_len * split[0])
        vl_len = int(total_len * split[1])
        ts_len = total_len - (tr_len + vl_len)

        tr_ds, vl_ds, ts_ds = random_split(ds, [tr_len, vl_len, ts_len])

        tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=ns)
        val_dl = DataLoader(vl_ds, batch_size=bs, shuffle=False, num_workers=ns)
        ts_dl = DataLoader(ts_ds, batch_size=1, shuffle=False, num_workers=ns)

        return tr_dl, val_dl, ts_dl, ds.cls_names, ds.cls_counts
