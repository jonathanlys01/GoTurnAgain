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
        model.classifier.load_state_dict(torch.load(model_path)["state_dict"])
    elif model_type == "GoNet":
        model = GoNet()
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"))["state_dict"])
    else:
        raise ValueError("Invalid model type")
    
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    tracker_vanilla = Tracker(model, optical_flow=None)
    target_size = (32, 32) if model_type == "GoNet" else (64, 64)
    tracker_of = Tracker(model, optical_flow="tvl1", target_size=target_size)
    
    annotations = utils.load_sequences(path)
    
    for object_name in annotations:
        print(object_name)
    
        img1 = io.imread(annotations[object_name][0]["path"])
        
        box1 = annotations[object_name][0]["box"]
        
        tracker_vanilla.init_tracker(img1, box1)
        tracker_of.init_tracker(img1, box1)
        
        imgs = [img1.copy()]
        
        
        centroid_errors_v = [0]
        ious_v = [1]
        
        centroid_errors_of = [0]
        ious_of = [1]
        
        box_c_v = box1
        box_c_of = box1
        
        for i in tqdm(range(1, len(annotations[object_name]))):
            img_c = io.imread(annotations[object_name][i]["path"])
        
            
            box_gt = annotations[object_name][i]["box"]
            
            box_c_v, search_region = tracker_vanilla.step(img_c, show=show)
            box_c_v = box_c_v.astype(int)
            
            box_c_of, _ = tracker_of.step(img_c, show=show)
            box_c_of = box_c_of.astype(int)
            
            
            img_c = cv2.rectangle(img_c, (box_c_v[0], box_c_v[1]), (box_c_v[2], box_c_v[3]), (255, 0, 0), 2) # vanilla in blue
            img_c = cv2.rectangle(img_c, (box_c_of[0], box_c_of[1]), (box_c_of[2], box_c_of[3]), (0, 0, 255), 2) # optical flow in red
            img_c = cv2.rectangle(img_c, (box_gt[0], box_gt[1]), (box_gt[2], box_gt[3]), (0, 255, 0), 2) # ground truth in green
            
            imgs.append(img_c)

            ious_v.append(utils.iou_unit(box_c_v, box_gt))
            ious_of.append(utils.iou_unit(box_c_of, box_gt))
            
            centroid_errors_v.append(utils.centroid_error(box_c_v, box_gt))
            centroid_errors_of.append(utils.centroid_error(box_c_of, box_gt))
            
            if show:
                plt.imshow(search_region)
                plt.axis("off")
                plt.show()
            
        
        ious_v = np.array(ious_v)
        ious_of = np.array(ious_of)

        print(f"IoU v: {np.mean(ious_v):.2f} of: {np.mean(ious_of):.2f}")
        
        centroid_errors_v = np.array(centroid_errors_v)
        centroid_errors_of = np.array(centroid_errors_of)  
        
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 1, 1)
        plt.plot(ious_v, label="Vanilla")
        plt.plot(ious_of, label="Optical flow")
        plt.legend()
        plt.title(f"IoU v: {np.mean(ious_v):.2f} of: {np.mean(ious_of):.2f}")
        plt.ylim(0, 1)
        plt.xlabel("Frame")
        plt.ylabel("IoU")

        plt.subplot(2, 1, 2)
        plt.plot(centroid_errors_v, label="Vanilla")
        plt.plot(centroid_errors_of, label="Optical flow")
        plt.legend()
        plt.title(f"C error v: {np.mean(centroid_errors_v):.2f} of: {np.mean(centroid_errors_of):.2f}")
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
        parser.add_argument("--show", "-s", action="store_true")
        
        args = parser.parse_args()
        
        main(args.path, args.model_path, args.model_type, args.show)
    
