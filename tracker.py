from model import GoNet, FasterGTA
import torch
from helper import NormalizeToTensor, Rescale, BoundingBox, crop_sample
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from eval_utils import delta_by_optical_flow

class Tracker():
    def __init__(self, model, k_context=2, optical_flow=None):
        assert optical_flow in [None, "tvl1", "ilk"], "Invalid optical flow method"
        self.optical_flow = optical_flow

        self.net = model
        self.net.eval()
        
        self.is_init = False
        
        self.scale = Rescale((224, 224))
        self.transform_tensor = transforms.Compose([NormalizeToTensor()])
        self.opts = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.is_init = False
        
        self.scale_context = k_context
        
    def init_tracker(self, img, box):
        """
        Box is in [xmin, ymin, xmax, ymax] format
        """
        self.prev_img = img
        self.prev_box = box
        
        self.is_init = True
        
    def step(self, image, show=False):
        assert self.is_init, "Tracker not initialized"
        
        
        prev_sample, opts_prev = crop_sample({'image': self.prev_img,
                                             'bb': self.prev_box})
        curr_sample, opts_curr = crop_sample({'image': image,
                                             'bb': self.prev_box})
        
        if self.optical_flow is not None:
            # compute optical flow of cropped images
            temp_prev = prev_sample['image']
            temp_curr = curr_sample['image']
            if show:
                plt.subplot(1, 2, 1)
                plt.imshow(temp_prev)
                plt.subplot(1, 2, 2)
                plt.imshow(temp_curr)
                plt.show()
            u,v = delta_by_optical_flow(temp_prev, temp_curr, mode=self.optical_flow)
        else:
            u, v = 0, 0
            
        new_box = self.prev_box + np.array([u, v, u, v])
        
        curr_sample, opts_curr = crop_sample({'image': image,
                                             'bb': new_box})
        
        if show:
            plt.subplot(1, 2, 1)
            plt.imshow(prev_sample['image'])
            plt.subplot(1, 2, 2)
            plt.imshow(curr_sample['image'])
            plt.show()
        
        search_region = opts_curr['search_region']
        
        self.opts = opts_curr
        self.curr_img = image
        curr_img = self.scale(curr_sample, opts_curr)['image']
        prev_img = self.scale(prev_sample, opts_prev)['image']
        sample = {'previmg': prev_img, 'currimg': curr_img}
        sample = self.transform_tensor(sample)

        # do forward pass to get box
        box = np.array(self._get_rect(sample))

        # update previous box and image
        self.prev_img = np.copy(image)
        self.prev_box = box
        
        return box, search_region
    
    def _get_rect(self, sample):
        """
        Performs forward pass through the GOTURN network to regress
        bounding box coordinates in the original image dimensions.
        """
        x1, x2 = sample['previmg'], sample['currimg']
        x1 = x1.unsqueeze(0).to(self.device)
        x2 = x2.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            y = self.net(x1, x2)
        bb = y.data.cpu().numpy().transpose((1, 0))
        bb = bb[:, 0]
        bbox = BoundingBox(bb[0], bb[1], bb[2], bb[3], kContext= self.scale_context)

        # inplace conversion
        bbox.unscale(self.opts['search_region'])
        bbox.uncenter(self.curr_img,
                      self.opts['search_location'],
                      self.opts['edge_spacing_x'],
                      self.opts['edge_spacing_y'])
        return bbox.get_bb_list()
        
        
        
        


    
