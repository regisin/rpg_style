import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastai.vision import *
from fastai.vision.gan import *

folder = 'barbarian'
path = Path('data')
dest = path/folder

def get_data(bs, size):
    return (GANItemList.from_folder(path, noise_sz=100)
               .no_split()
               .label_from_func(noop)
               .transform(tfms=[[crop_pad(size=size, row_pct=(0,1), col_pct=(0,1))], []], size=size, tfm_y=True)
               .databunch(bs=bs, num_workers=0)
               .normalize(stats = [torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])], do_x=False, do_y=True))

bs=8
size=256
data = get_data(bs, size)

generator = basic_generator(in_size=size, n_channels=3, n_extra_layers=1)
critic    = basic_critic   (in_size=size, n_channels=3, n_extra_layers=1)

learn = GANLearner.wgan(data, generator, critic, switch_eval=False, 
                        opt_func = partial(optim.Adam, betas = (0.,0.99)), wd=0.)

learn.fit(30, 2e-4)

learn.gan_trainer.switch(gen_mode=True)
learn.show_results(ds_type=DatasetType.Train, rows=4, figsize=(8,8))