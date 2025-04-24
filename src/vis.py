import numpy as np, os
from matplotlib import pyplot as plt
from torchvision import transforms as T
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

class Visualization:

    def __init__(self, vis_datas, vis_dir, ds_nomi, n_ims, rows, cmap=None, cls_names=None, cls_counts=None, t_type="rgb"):
        self.n_ims, self.rows = n_ims, rows
        self.t_type, self.cmap = t_type, cmap
        self.cls_names = cls_names
        self.vis_dir = vis_dir
        self.ds_nomi = ds_nomi
        os.makedirs(vis_dir, exist_ok = True)
        
        data_names = ["train", "val", "test"]
        self.vis_datas = {data_names[i]: vis_datas[i] for i in range(len(vis_datas))}
        if isinstance(cls_counts, list): 
            self.analysis_datas = {data_names[i]: cls_counts[i] for i in range(len(cls_counts))}
        else: 
            self.analysis_datas = {"all": cls_counts}

    def tn2np(self, t):
        gray_tfs = T.Compose([T.Normalize(mean=[0.], std=[1/0.5]), T.Normalize(mean=[-0.5], std=[1])])
        rgb_tfs = T.Compose([T.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]), 
                             T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])])
        
        invTrans = gray_tfs if self.t_type == "gray" else rgb_tfs
        
        return (invTrans(t) * 255).detach().squeeze().cpu().permute(1, 2, 0).numpy().astype(np.uint8) if self.t_type == "gray" \
               else (invTrans(t) * 255).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)

    def plot(self, rows, cols, count, im, title="Original Image"):
        plt.subplot(rows, cols, count)
        plt.imshow(self.tn2np(im))
        plt.axis("off")
        plt.title(title)
        return count + 1

    def vis(self, data, save_name):
        print(f"{save_name.upper()} Data Visualization is in process...\n")
        assert self.cmap in ["rgb", "gray"], "Please choose rgb or gray cmap"
        cmap = "viridis" if self.cmap == "rgb" else None
        cols = self.n_ims // self.rows
        count = 1

        plt.figure(figsize=(25, 20))
        indices = [np.random.randint(low=0, high=len(data) - 1) for _ in range(self.n_ims)]

        for idx, index in enumerate(indices):
            if count == self.n_ims + 1: break
            image, label = data.dataset[index]            
            plt.subplot(self.rows, self.n_ims // self.rows, idx + 1)
            image = self.tn2np(image)
            if Image.fromarray(image).mode != "RGB": image = image.convert("RGB")

            if cmap: plt.imshow(image, cmap=cmap)
            else: plt.imshow(image)

            plt.axis('off')
            if self.cls_names is not None:
                plt.title(f"GT -> {self.cls_names[int(label)]}")
            else:
                plt.title(f"GT -> {label}")
        
        plt.savefig(f"{self.vis_dir}/{self.ds_nomi}_{save_name}_data_vis.png")

    def data_analysis(self, cls_counts, save_name):
        print("Data analysis is in process...\n")
        width, text_width, text_height = 0.7, 0.05, 2
        cls_names = list(cls_counts.keys())
        counts = list(cls_counts.values())
        _, ax = plt.subplots(figsize=(20, 10))
        indices = np.arange(len(counts))
        ax.bar(indices, counts, width, color="darkorange")
        ax.set_xlabel("Class Names", color="black")        
        ax.set_xticks(np.arange(len(cls_names)), labels=cls_names, rotation=90)
        ax.set(xticks=indices, xticklabels=cls_names)
        ax.set_ylabel("Data Counts", color="black")
        ax.set_title("Dataset Class Imbalance Analysis")
        for i, v in enumerate(counts):
            ax.text(i - text_width, v + text_height, str(v), color="royalblue")
        plt.savefig(f"{self.vis_dir}/{self.ds_nomi}_data_analysis.png")
    
    def plot_pie_chart(self, cls_counts):
        print("Generating pie chart...\n")
        labels = list(cls_counts.keys())
        sizes = list(cls_counts.values())
        explode = [0.1] * len(labels)
        
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)
        plt.title("Class Distribution")
        plt.axis("equal")
        plt.savefig(f"{self.vis_dir}/{self.ds_nomi}_pie_chart.png")

    def visualization(self): [self.vis(data.dataset, save_name) for (save_name, data) in self.vis_datas.items()]
        
    def analysis(self): [self.data_analysis(data, save_name) for (save_name, data) in self.analysis_datas.items()]

    def pie_chart(self): [self.plot_pie_chart(data) for data in self.analysis_datas.values()]