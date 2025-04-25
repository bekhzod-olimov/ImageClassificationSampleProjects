import os
import torch
import argparse
import timm
import pickle
from src.train import TrainValidation
from src.vis import Visualization
from src.plot import PlotLearningCurves
from src.infer import ModelInferenceVisualizer
from src.transform import get_tfs
from data.parse import CustomDataset
from data.fetch import DatasetDownloader

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a classification model")

    parser.add_argument('--dataset_name', type=str, default="lentils", help="Name of the dataset")
    parser.add_argument('--device', type=str, help="GPU/CPU for training")
    parser.add_argument('--dataset_root', type=str, default="/home/bekhzod/Desktop/backup/image_classification_project_datasets", help="Root folder for datasets")
    parser.add_argument('--cls_root', type=str, default="saved_cls_names", help="Root folder for class names")
    parser.add_argument('--vis_dir', type=str, default="vis", help="Directory for visualizations")
    parser.add_argument('--learning_curve_dir', type=str, default="learning_curves", help="Directory to save learning curves")
    parser.add_argument('--outputs_dir', type=str, default="results", help="Directory for inference results")
    parser.add_argument('--model_name', type=str, default="rexnet_150", help="Model architecture from timm")
    parser.add_argument('--save_dir', type=str, default="saved_models", help="Directory to save the model checkpoints")
    parser.add_argument('--image_size', type=int, default=224, help="Input image size for the model")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for dataloaders")
    parser.add_argument('--patience', type=int, default=3, help="Early stopping patience")

    return parser.parse_args()

def main():
    args = parse_args()
    # ds_nomlari = ["car_brands", "dog_breeds", "lentils", "pet_disease", "rice_leaf_disease", "apple_disease"]
    ds_nomlari = ["geo_scene"]

    for ds_nomi in ds_nomlari:

        print(f"{ds_nomi} dataset bilan train jarayoni boshlanmoqda...")

        args.dataset_name = ds_nomi

        device = args.device
        ds_path = os.path.join(args.dataset_root, args.dataset_name)

        if not os.path.isdir(ds_path): DatasetDownloader(save_dir=ds_path).download(ds_nomi=args.dataset_name)
        else: print(f"{args.dataset_name} dataseti allaqachon {args.dataset_root} yo'lagiga yuklab olingan.")

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        tfs = get_tfs(im_size=args.image_size, mean=mean, std=std)    

        tr_dl, val_dl, ts_dl, classes, cls_counts = CustomDataset.get_dls(
            data_turgan_yolak=args.dataset_root,
            ds_nomi=args.dataset_name,
            tfs=tfs,
            bs=args.batch_size
        )
        os.makedirs(args.cls_root, exist_ok=True)
        with open(f"{args.cls_root}/{args.dataset_name}_cls_names.pkl", "wb") as f: pickle.dump(classes, f)

        print(f"Train data {len(tr_dl)} ta, validation data {len(val_dl)} ta, test data {len(ts_dl)} ta mini-batchlardan iborat.")    
        print(f"Datasetdagi klasslar -> {classes}")    

        vis = Visualization(
            vis_datas=[tr_dl, val_dl, ts_dl],
            n_ims=20,
            rows=4,
            cmap="rgb",
            vis_dir=args.vis_dir,
            ds_nomi=args.dataset_name,
            cls_names=list(classes.keys()),
            cls_counts=cls_counts
        )
        vis.analysis(); vis.pie_chart(); vis.visualization()

        trainer = TrainValidation(
            model_name=args.model_name,
            device=device,
            save_prefix=args.dataset_name,
            classes=classes,
            patience=args.patience,
            tr_dl=tr_dl,
            val_dl=val_dl,
            dev_mode=False
        )
        trainer.run()

        PlotLearningCurves(
            tr_losses=trainer.tr_losses,
            val_losses=trainer.val_losses,
            tr_accs=trainer.tr_accs,
            val_accs=trainer.val_accs,
            tr_f1s=trainer.tr_f1s,
            val_f1s=trainer.val_f1s,
            save_dir=args.learning_curve_dir,
            ds_nomi=args.dataset_name
        ).visualize()

        model = timm.create_model(
            model_name=args.model_name,
            pretrained=True,
            num_classes=len(classes)
        ).to(device)
        model.load_state_dict(torch.load(f"{args.save_dir}/{args.dataset_name}_best_model.pth"))

        inference_visualizer = ModelInferenceVisualizer(
            model=model,
            device=device,
            outputs_dir=args.outputs_dir,
            ds_nomi=args.dataset_name,
            mean=mean,
            std=std,
            class_names=list(classes.keys()),
            im_size=args.image_size
        )
        inference_visualizer.infer_and_visualize(ts_dl, num_images=20, rows=4)

if __name__ == "__main__": main()