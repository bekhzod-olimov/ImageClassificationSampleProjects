import os, numpy as np
from matplotlib import pyplot as plt

class PlotLearningCurves:
    
    def __init__(self, tr_losses, val_losses, tr_accs, val_accs, tr_f1s, val_f1s, save_dir, ds_nomi):
        
        os.makedirs(save_dir, exist_ok=True); self.save_dir = save_dir; self.ds_nomi = ds_nomi
        self.tr_losses, self.val_losses, self.tr_accs, self.val_accs, self.tr_f1s, self.val_f1s = tr_losses, val_losses, tr_accs, val_accs, tr_f1s, val_f1s
        
    def plot(self, array_1, array_2, label_1, label_2, color_1, color_2):
        
        plt.plot(array_1, label = label_1, c = color_1)
        plt.plot(array_2, label = label_2, c = color_2)
        
    def create_figure(self):  plt.figure(figsize = (10, 5))
    
    def decorate(self, ylabel, xlabel = "Epochs"): 
        
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.xticks(ticks = np.arange(len(self.tr_accs)), labels = [i for i in range(1, len(self.tr_accs) + 1)]); plt.legend()        
        
    def visualize(self):
        
        # Figure 1: Loss Curves with more colorful colors
        self.create_figure()
        self.plot(array_1 = self.tr_losses, array_2 = self.val_losses, label_1 = "Train Loss", label_2 = "Validation Loss", color_1 = "#FF6347", color_2 = "#3CB371")  # Tomato and MediumSeaGreen
        self.decorate(ylabel = "Loss Values")
        plt.savefig(f"{self.save_dir}/{self.ds_nomi}_loss_values.png")
        
        # Figure 2: Accuracy Curves with more colorful colors
        self.create_figure()
        self.plot(array_1 = self.tr_accs, array_2 = self.val_accs, label_1 = "Train Accuracy", label_2 = "Validation Accuracy", color_1 = "#FF4500", color_2 = "#32CD32")  # OrangeRed and LimeGreen
        self.decorate(ylabel = "Accuracy Scores")
        plt.savefig(f"{self.save_dir}/{self.ds_nomi}_acc_scores.png")
        
        # Figure 3: F1 Score Curves with more colorful colors
        self.create_figure()
        self.plot(array_1 = self.tr_f1s, array_2 = self.val_f1s, label_1 = "Train F1 Score", label_2 = "Validation F1 Score", color_1 = "#8A2BE2", color_2 = "#DC143C")  # BlueViolet and Crimson
        self.decorate(ylabel = "F1 Scores")
        plt.savefig(f"{self.save_dir}/{self.ds_nomi}_f1_scores.png")