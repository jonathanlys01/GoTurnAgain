from box import Box
from pprint import pprint

config = {
    "log":False,
    "wandb" : True,
    "paths" : {
        "imagenetloc" : "/users/local/imagenet",
        "imagenet" : "/users/local/imagenet",
        "alov" : "/users/local/alov300pp",
    },
    "batch_size" : 32,
    "num_workers" : 4,
    "bb_params": {
        "lambda_shift_frac": 5,
        "lambda_scale_frac": 15,
        "min_scale": -0.4,
        "max_scale": 0.4,
    },
    "input_size" : 224
}

cfg = Box(config)

if __name__ == "__main__":
    pprint(cfg)