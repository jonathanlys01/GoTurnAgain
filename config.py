from box import Box
from pprint import pprint

config = {
    "log":False,
    "wandb" : False,
    "paths" : {
        "imagenetloc" : "/nasbrain/datasets/imagenetloc/",
        "imagenet": "/nasbrain/datasets/imagenet/images/val",
    },
    "batch_size" : 32,
    "num_workers" : 4,
}

cfg = Box(config)

if __name__ == "__main__":
    pprint(cfg)
