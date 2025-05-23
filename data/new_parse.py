import os
import torch
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import random_split, Dataset, DataLoader
from sklearn.model_selection import train_test_split

torch.manual_seed(2025)

class CustomDataset(Dataset):
    def __init__(self, data_turgan_yolak, ds_nomi, rasm_yolaklari=None, rasm_javoblari=None, tfs=None, data_type=None, rasm_fayllari=[".png", ".jpg", ".jpeg", ".bmp"]):
        
        self.tfs, self.ds_nomi = tfs, ds_nomi
        self.data_type = data_type
        self.data_turgan_yolak = data_turgan_yolak 
        self.rasm_fayllari          = rasm_fayllari      

        if rasm_yolaklari and rasm_javoblari: self.rasm_yolaklari = rasm_yolaklari; self.im_lbls = rasm_javoblari
        else:
            self.get_root(); self.get_files()
            
        self.get_info() 

    def get_root(self): 

        if self.ds_nomi == "pet_disease": self.root = f"{self.data_turgan_yolak}/{self.ds_nomi}/{self.ds_nomi}/data"
        elif self.ds_nomi == "facial_expression": self.root = f"{self.data_turgan_yolak}/{self.ds_nomi}/{self.ds_nomi}"        
        elif self.ds_nomi == "geo_scene": self.root = f"{self.data_turgan_yolak}/{self.ds_nomi}/{self.ds_nomi}/GeoSceneNet16K"
        elif self.ds_nomi == "lentils": self.root = f"{self.data_turgan_yolak}/{self.ds_nomi}/lentils/data"        
        elif self.ds_nomi == "rice_leaf_disease": self.root = f"{self.data_turgan_yolak}/{self.ds_nomi}/rice_leaf_disease/Rice Leaf  Disease Dataset"
        elif self.ds_nomi == "car_brands": self.root = f"{self.data_turgan_yolak}/{self.ds_nomi}/car_brands"
        elif self.ds_nomi == "dog_breeds": self.root = f"{self.data_turgan_yolak}/{self.ds_nomi}/dog_breeds/Dog Breed Classification"        
        elif self.ds_nomi == "apple_disease": self.root = f"{self.data_turgan_yolak}/{self.ds_nomi}/{self.ds_nomi}/{self.ds_nomi}/images"
        elif self.ds_nomi == "animals": self.root = f"{self.data_turgan_yolak}/{self.ds_nomi}/{self.ds_nomi}/animal_dataset/animal_dataset/{self.ds_nomi}/{self.ds_nomi}"        
        elif self.ds_nomi == "pokemon": self.root = f"{self.data_turgan_yolak}/{self.ds_nomi}/{self.ds_nomi}/dataset"        
        
    
    def get_files(self): 
        if self.ds_nomi in ["dog_breeds"]: self.rasm_yolaklari = [path for im_file in self.rasm_fayllari for path in glob(f"{self.root}/*/*/*{im_file}")]        
        elif self.ds_nomi in ["lentils", "apple_disease"]: self.rasm_yolaklari = [path for im_file in self.rasm_fayllari for path in glob(f"{self.root}/*{im_file}")]
        elif self.ds_nomi in ["facial_expression"]: self.rasm_yolaklari = [path for im_file in self.rasm_fayllari for path in glob(f"{self.root}/{self.data_type}/*/*{im_file}")]
        elif self.ds_nomi in ["pokemon"]: self.rasm_yolaklari = [path for im_file in self.rasm_fayllari for path in glob(f"{self.root}/{self.data_type}/*/*{im_file}")]
        else: self.rasm_yolaklari = [path for im_file in self.rasm_fayllari for path in glob(f"{self.root}/*/*{im_file}")] 

    def get_info(self):

        self.cls_names, self.cls_counts = {}, {}
        count = 0
        for im_path in self.rasm_yolaklari:
            class_name = self.get_class(im_path)
            if class_name not in self.cls_names:
                self.cls_names[class_name] = count
                self.cls_counts[class_name] = 1
                count += 1
            else: self.cls_counts[class_name] += 1
    
    def get_class(self, path): 
        if self.ds_nomi in ["lentils", "apple_disease"]: return os.path.basename(path).split("_")[0]
        else: return os.path.dirname(path).split("/")[-1]

    def __len__(self): return len(self.rasm_yolaklari)

    def __getitem__(self, idx):
        
        im_path = self.rasm_yolaklari[idx]
        im = Image.open(im_path)
        if im.mode != "RGB": im = im.convert("RGB")    
        if self.ds_nomi in ["pokemon", "facial_expression"]: gt = self.cls_names[self.get_class(im_path)]
        else: gt = self.im_lbls[idx]

        if self.tfs: im = self.tfs(im)

        return im, gt

    @classmethod
    def get_dls(cls, data_turgan_yolak, ds_nomi, tfs, bs, split=[0.8, 0.1, 0.1], ns=4):
        
        if ds_nomi in ["facial_expression"]:

            tr_ds = cls(data_turgan_yolak=data_turgan_yolak, data_type = "train", ds_nomi=ds_nomi, tfs=tfs)
            vl_ds = cls(data_turgan_yolak=data_turgan_yolak, data_type = "val", ds_nomi=ds_nomi, tfs=tfs)
            ts_ds = cls(data_turgan_yolak=data_turgan_yolak, data_type = "test", ds_nomi=ds_nomi, tfs=tfs)

            cls_names, cls_counts = tr_ds.cls_names, [tr_ds.cls_counts, vl_ds.cls_counts, ts_ds.cls_counts]

        elif ds_nomi in ["pokemon"]:

            tr_ds = cls(data_turgan_yolak=data_turgan_yolak, data_type = "train", ds_nomi=ds_nomi, tfs=tfs)            
            ts_ds = cls(data_turgan_yolak=data_turgan_yolak, data_type = "test", ds_nomi=ds_nomi, tfs=tfs)

            cls_names, cls_counts = tr_ds.cls_names, [tr_ds.cls_counts, ts_ds.cls_counts]           

            total_len = len(tr_ds)
            tr_len = int(total_len * split[0])
            vl_len = total_len - tr_len        

            tr_ds, vl_ds = random_split(tr_ds, [tr_len, vl_len])            

        else: 

            ds = cls(data_turgan_yolak=data_turgan_yolak, ds_nomi=ds_nomi, tfs=tfs)

            rasm_yolaklari = ds.rasm_yolaklari
            rasm_javoblari = [ds.cls_names[ds.get_class(rasm_yolagi)] for rasm_yolagi in rasm_yolaklari]

            train_paths, temp_paths, train_lbls, temp_lbls = train_test_split( rasm_yolaklari, rasm_javoblari, test_size=(split[1] + split[2]), stratify=rasm_javoblari, random_state=2025 )
            
            val_ratio = split[1] / (split[1] + split[2])
            val_paths, test_paths, val_lbls, test_lbls = train_test_split(
                temp_paths, temp_lbls, test_size=(1 - val_ratio), stratify=temp_lbls, random_state=2025
            )

            tr_ds = cls(data_turgan_yolak=data_turgan_yolak, ds_nomi = ds_nomi, tfs=tfs, rasm_yolaklari = train_paths, rasm_javoblari = train_lbls)
            vl_ds = cls(data_turgan_yolak=data_turgan_yolak, ds_nomi = ds_nomi, tfs=tfs, rasm_yolaklari = val_paths, rasm_javoblari = val_lbls)
            ts_ds = cls(data_turgan_yolak=data_turgan_yolak, ds_nomi = ds_nomi, tfs=tfs, rasm_yolaklari = test_paths, rasm_javoblari = test_lbls)            

            cls_names = ds.cls_names; cls_counts = [tr_ds.cls_counts, vl_ds.cls_counts, ts_ds.cls_counts]

        tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=ns)
        val_dl = DataLoader(vl_ds, batch_size=bs, shuffle=False, num_workers=ns)
        ts_dl = DataLoader(ts_ds, batch_size=1, shuffle=False, num_workers=ns)

        return tr_dl, val_dl, ts_dl, cls_names, cls_counts
