from box import Box
from pprint import pprint

config = {
    "log":False,
    "wandb" : False,
    "paths" : {
        "imagenetloc" : "/nasbrain/datasets/imagenetloc/",
    },
    "batch_size" : 32,
}

cfg = Box(config)

if __name__ == "__main__":
    pprint(cfg)
