import os 
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import PIL.Image
import cv2

def load_sequences(path="sequences-train"):
    unique = set()

    for filename in os.listdir(path):
        if filename.startswith("."): # ignore .DS_Store
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
            
    annotations = {name: [] for name in names}
    
    for name in names:
        for i in range(names[name]["start"], names[name]["end"]+1):
            mask = io.imread(f"{path}/{name}"+"-%03d.png" % i)
            indexes = np.where(mask>0)
            box = np.array([indexes[1].min(), indexes[0].min(), indexes[1].max(), indexes[0].max()])
            box = box.astype(int)
            path_to_file = f"{path}/{name}"+"-%03d.bmp" % i
            annotations[name].append(
                {
                    "box": box,
                    "path": path_to_file
                }
            )
            
    return annotations


def main():
    
    annotations = load_sequences()

    for object_name in annotations:
        print(object_name)
        img1 = io.imread(annotations[object_name][0]["path"])
        h,w,c = img1.shape
        box = annotations[object_name][0]["box"]
        cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
        
        list_images = [img1]
        for i in range(1, len(annotations[object_name])):
            img = io.imread(annotations[object_name][i]["path"])
            box = annotations[object_name][i]["box"]
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
            list_images.append(img)
        
        list_images = [PIL.Image.fromarray(img) for img in list_images]
        os.makedirs("gifs", exist_ok=True)
        make_gif(list_images, f"gifs/{object_name}.gif", duration=100)

def make_gif(list_images, path, duration=100):
    assert len(list_images) > 1, "Need at least 2 images to make a gif"
    
    list_images[0].save(path, save_all=True, append_images=list_images[1:], optimize=True, duration=duration, loop=0)

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
    
def centroid_error(box1, box2):
    """
    box1 and box2 are [x1, y1, x2, y2], all values are in [0,1]
    """
    
    c1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    c2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
    

if __name__ == "__main__":
    
    main()
    
    
    
    
    
    
    
