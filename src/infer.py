import cv2, random, torch, torchmetrics, os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from src.train import TrainValidation
from src.transform import get_tfs


class Denormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class ModelInferenceVisualizer:
    def __init__(self, model, device, mean, std, outputs_dir, ds_nomi, class_names=None, im_size=224):
        
        self.denormalize = Denormalize(mean, std)
        self.model = model
        self.device = device
        self.class_names = class_names
        self.outputs_dir = outputs_dir        
        self.ds_nomi = ds_nomi
        self.im_size = im_size
        self.model.eval()  
        self.f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=len(class_names)).to(self.device)

    def tensor_to_image(self, tensor):
        
        tensor = self.denormalize(tensor)  
        tensor = tensor.permute(1, 2, 0)  
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    def plot_value_array(self, logits, gt, class_names):
        
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1)
        
        plt.grid(visible=True)
        plt.xticks(range(len(class_names)), class_names, rotation='vertical')
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        bars = plt.bar(range(len(class_names)), [p.item() for p in probs[0]], color="#777777")
        plt.ylim([0, 1])
        if isinstance(gt, str):
            bars[pred_class].set_color('green') if pred_class.item() == class_names[gt] else bars[pred_class].set_color('red')            
        else:
            bars[pred_class].set_color('green') if pred_class.item() == gt else bars[pred_class].set_color('red')
        
        # Save figure to a buffer
        import io
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()  # Close the figure after saving it to free memory
        buf.seek(0)

        return buf 

    def generate_cam_visualization(self, image_tensor):
        
        cam = GradCAM(model=self.model, target_layers=[self.model.features[-1].conv], use_cuda=self.device == "cuda")
        grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0))[0, :]
        return grayscale_cam

    def demo(self, im_path):
        
        with Image.open(im_path) as im: im = im.convert("RGB")
        im_size = im.size[0], im.size[1] 
        if isinstance(im_path, str): gt = os.path.splitext(os.path.basename(im_path))[0].split("___")[-1]
        else: gt = "Uploaded Image"         
        im_tn = get_tfs()(im).unsqueeze(dim = 0).to(self.device)
        with torch.no_grad(): logits = self.model(im_tn)
        pred   = torch.argmax(logits, dim=1)

        grayscale_cam = cv2.resize(self.generate_cam_visualization(im_tn.squeeze(dim=0)), im_size)
        gradcam = show_cam_on_image(np.array(im).astype(np.uint8) / 255, grayscale_cam, image_weight=0.4, use_rgb=True)
        if logits.dim() == 1:  # If 1D, add a batch dimension
            logits = logits.unsqueeze(0)
                
        di = {}
        di["pred"] = pred.item()
        di["gt"] = gt
        di["original_im"] = im
        di["gradcam"] = gradcam
        di["probs"]   = self.plot_value_array(logits=logits, gt=gt, class_names=self.class_names)
        di["confidence"] = (torch.max(torch.nn.functional.softmax(logits, dim=1), dim = 1)[0].item() * 100)
        
        return di

    def infer_and_visualize(self, test_dl, num_images=5, rows=2, demo=False):
        
        preds, images, lbls, logitss = [], [], [], []
        accuracy, count = 0, 1
        with torch.no_grad():

            for idx, batch in tqdm(enumerate(test_dl), desc="Inference"):
                im, gt = TrainValidation.to_device(batch, device=self.device)                
                logits = self.model(im)
                pred_class = torch.argmax(logits, dim=1)
                
                accuracy += (pred_class == gt).sum().item()
                self.f1_metric.update(logits, gt)  
        
                images.append(im[0])
                logitss.append(logits[0])
                preds.append(pred_class[0].item())
                lbls.append(gt[0].item())
        
        # Compute metrics AFTER the loop
        print(f"Accuracy of the model on the test data -> {(accuracy / len(test_dl.dataset)):.3f}")
        print(f"F1 score of the model on the test data -> {(self.f1_metric.compute().item()):.3f}") 

        plt.figure(figsize=(20, 10))
        indices = [random.randint(0, len(images) - 1) for _ in range(num_images)]
        for idx, index in enumerate(indices):
            # Convert and denormalize image
            im = self.tensor_to_image(images[index].squeeze())
            pred_idx = preds[index]
            gt_idx = lbls[index]

            # Display image
            plt.subplot(rows, 2 * num_images // rows, count)
            count += 1
            plt.imshow(im, cmap="gray")
            plt.axis("off")

            # GradCAM visualization
            grayscale_cam = self.generate_cam_visualization(images[index])
            visualization = show_cam_on_image(im / 255, grayscale_cam, image_weight=0.4, use_rgb=True)
            plt.imshow(cv2.resize(visualization, (self.im_size, self.im_size), interpolation=cv2.INTER_LINEAR), alpha=0.7, cmap='jet')
            plt.axis("off")

            # Prediction probability array
            logits = logitss[index]
            if logits.dim() == 1:  # If 1D, add a batch dimension
                logits = logits.unsqueeze(0)
            plt.subplot(rows, 2 * num_images // rows, count)
            count += 1
            bars = self.plot_value_array(logits=logits, gt=gt_idx, class_names=self.class_names)

            # Title with GT and Prediction
            if self.class_names:
                gt_name = self.class_names[gt_idx]
                pred_name = self.class_names[pred_idx]
                color = "green" if gt_name == pred_name else "red"
                plt.title(f"GT -> {gt_name} ; PRED -> {pred_name}", color=color)
        
        os.makedirs(self.outputs_dir, exist_ok = True)
        plt.tight_layout()        
        plt.savefig(f"{self.outputs_dir}/{self.ds_nomi}_model_performance_analysis.png")

        # Plot confusion matrix
        plt.figure(figsize=(20, 10))
        cm = confusion_matrix(lbls, preds)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"{self.outputs_dir}/{self.ds_nomi}_confusion_matrix.png")