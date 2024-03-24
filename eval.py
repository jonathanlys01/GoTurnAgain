import eval_utils as utils
import skimage.io as io
import cv2
import PIL
import os
import torch
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model import FasterGTA, GoNet 

from tracker import Tracker


def main(path="sequences-train",
         model_path="model.pth",
         model_type="FasterGTA",
            show=False
         ):
    
    os.makedirs("results", exist_ok=True)
    
    if model_type == "FasterGTA":
        model = FasterGTA()
        model.classifier.load_state_dict(torch.load(model_path))
    elif model_type == "GoNet":
        model = GoNet()
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"))["state_dict"])
    else:
        raise ValueError("Invalid model type")
    
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    tracker = Tracker(model, k_context=3)
    
    annotations = utils.load_sequences(path)
    
    for object_name in annotations:
        print(object_name)
    
        img1 = io.imread(annotations[object_name][0]["path"])
        
        box1 = annotations[object_name][0]["box"]
        
        tracker.init_tracker(img1, box1)
        
        imgs = [img1.copy()]
        centroid_errors = [0]
        ious = [1]
        
        box_c = box1
        
        for i in tqdm(range(1, len(annotations[object_name]))):
            img_c = io.imread(annotations[object_name][i]["path"])
        
            
            box_gt = annotations[object_name][i]["box"]
            
            box_c, search_region = tracker.step(img_c)
            box_c = box_c.astype(int)        
            
            img_c = cv2.rectangle(img_c, (box_c[0], box_c[1]), (box_c[2], box_c[3]), (255, 0, 0), 2)
            img_c = cv2.rectangle(img_c, (box_gt[0], box_gt[1]), (box_gt[2], box_gt[3]), (0, 255, 0), 2)
            
            imgs.append(img_c)

            ious.append(utils.iou_unit(box_c, box_gt))
            centroid_errors.append(utils.centroid_error(box_c, box_gt))
            
            if show:
                plt.imshow(search_region)
                plt.axis("off")
                plt.show()
            
        
        ious = np.array(ious)   
        centroid_errors = np.array(centroid_errors)
        
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 1, 1)
        plt.plot(ious)
        plt.ylim(0, 1)
        plt.title("IoU")
        plt.xlabel("Frame")
        plt.ylabel("IoU")

        plt.subplot(2, 1, 2)
        plt.plot(centroid_errors)
        plt.title("Centroid error")
        plt.xlabel("Frame")
        plt.ylabel("Centroid error")
        
        plt.tight_layout()  
        plt.savefig(f"results/{model_type}_{object_name}.png")
        plt.close()
   
        imgs = [PIL.Image.fromarray(img) for img in imgs]

        utils.make_gif(imgs, f"results/{model_type}_{object_name}.gif")     


if __name__ == "__main__":
    
    # Example usage:
    # python3 eval.py --path /users/local/sequences-train --model_path /users/local/saved_checkpoints/exp3/model_itr_300000_loss_56.81.pth.tar  --model_type FasterGTA
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--path", type=str, default="sequences-train")
        parser.add_argument("--model_path", type=str, default="pytorch_goturn.pth")
        parser.add_argument("--model_type", type=str, default="GoNet")
        
        args = parser.parse_args()
        
        main(args.path, args.model_path, args.model_type)
    
