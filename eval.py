import os 
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import PIL.Image
import cv2

def main():
    
    path = "sequences-train"

    unique = set()

    for filename in os.listdir(path):
        if filename.startswith("."):
            continue
        name = filename.split("-")[0]
        unique.add(name)
        
    print(unique)

    names = {un: {"start":float("inf"),
                "end":float("-inf"),
                "files":[]} for un in unique}

    for filename in os.listdir(path):
        if filename.startswith("."):
            continue
        name, num = filename.split("-")[:2] # last is the extension
        names[name]["files"].append(filename)
        num = int(num.split(".")[0])
        if num < names[name]["start"]:
            names[name]["start"] = num
        if num > names[name]["end"]:
            names[name]["end"] = num

    for name in names:
        object_name = name
    
        img1 = io.imread(f"sequences-train/{object_name}-001.bmp")
        h,w,c = img1.shape
        
        list_images = [PIL.Image.fromarray(img1)]   
        for i in range(names[object_name]["start"], names[object_name]["end"]+1):
            img = io.imread(f"sequences-train/{object_name}"+"-%03d.bmp" % i)
            mask = io.imread(f"sequences-train/{object_name}"+"-%03d.png" % i)
            indexes = np.where(mask>0)
            box = np.array([indexes[1].min()/w, indexes[0].min()/h, indexes[1].max()/w, indexes[0].max()/h])
            
            
            cv2.rectangle(img, (int(box[0]*w), int(box[1]*h)), (int(box[2]*w), int(box[3]*h)), (0,255,0), 2)
            img = PIL.Image.fromarray(img)
            list_images.append(img)

        os.makedirs("gifs", exist_ok=True)
        make_gif(list_images, f"gifs/{object_name}.gif", duration=100)

def make_gif(list_images, path, duration=100):
    assert len(list_images) > 1, "Need at least 2 images to make a gif"
    
    list_images[0].save(path, save_all=True, append_images=list_images[1:], optimize=False, duration=duration, loop=0)

def iou_unit(box1, box2):
    """
    box1 and box2 are [x1, y1, x2, y2], all values are in [0,1]    
    """
    
    intersection = [max(box1[0], box2[0]), 
                    max(box1[1], box2[1]), 
                    min(box1[2], box2[2]), 
                    min(box1[3], box2[3])]
    
    if intersection[2] < intersection[0] or intersection[3] < intersection[1]:
        return 0
    
    area_intersection = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])
    
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return area_intersection / (area_box1 + area_box2 - area_intersection)
    # Aunion = Abox1 + Abox2 - Aintersection
    

if __name__ == "__main__":
    
    main()
    
    
    
    
    
    
    
