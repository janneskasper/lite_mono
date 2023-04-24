from __future__ import absolute_import, division, print_function

from options import LiteMonoOptions
from trainer import Trainer

import torch, gc

options = LiteMonoOptions()
opts = options.parse()


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    trainer = Trainer(opts)
    trainer.train()
