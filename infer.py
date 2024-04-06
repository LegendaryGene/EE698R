import torch
from models import VAE_GAN
from dataloader import dataloader
from utils import show_and_save
from torch.autograd import Variable
from torchvision.utils import make_grid
import numpy as np
from torch.nn.functional import normalize
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bs = 16
run_num = 1
data_loader=dataloader(bs, "../DeepFashion/test_images")
print("Loaded Data")
gen = VAE_GAN().to(device)
gen.load_state_dict(torch.load(f"./saves/run{run_num}/gen_60.pth"))
print("Loaded Model")

Dictionary = {
  "MEN-Tees_Tanks": 0,
  "MEN-Shirts_Polos": 1,
  "MEN-Denim": 2,
  "MEN-Pants": 3,
  "MEN-Jackets_Vests": 4,
  "MEN-Shorts": 5,
  "MEN-Suiting": 6,
  "MEN-Sweaters": 7,
  "MEN-Sweatshirts_Hoodies": 8,
  "WOMEN-Tees_Tanks": 9,
  "WOMEN-Blouses_Shirts": 10,
  "WOMEN-Cardigans": 11,
  "WOMEN-Denim": 12,
  "WOMEN-Dresses": 13,
  "WOMEN-Graphic_Tees": 14,
  "WOMEN-Jackets_Coats": 15,
  "WOMEN-Leggings": 16,
  "WOMEN-Pants": 17,
  "WOMEN-Rompers_Jumpsuits": 18,
  "WOMEN-Shorts": 19,
  "WOMEN-Skirts": 20,
  "WOMEN-Sweaters": 21,
  "WOMEN-Sweatshirts_Hoodies": 22}

Yes = np.load(f"./saves/run{run_num}/Yes_60.npy")
No = np.load(f"./saves/run{run_num}/No_60.npy")
Toward = No - Yes

for i in range(1):
	real_batch = next(iter(data_loader))
	show_and_save(f"Results/{i}_original", make_grid((real_batch[0]*0.5+0.5).cpu(),max(1,bs//4)))
	x_fixed = Variable(real_batch[0]).to(device)
	b,_,a = gen(x_fixed)
	c = gen.decoder(b)
	temp = torch.from_numpy(Toward)
	temp = temp.to(device).to(torch.float32)
	#d = gen.decoder(temp)
	
	for key in Dictionary:
		temp1 = temp[Dictionary[key]]
		#temp3 = torch.randn(256).to(device)
		temp2 = torch.zeros(0).to(device)
		temp2 = torch.cat((temp2, temp1.repeat(bs))).reshape(bs,256)
		e = gen(x_fixed, temp2)[2]

		#print(e)
		#print()
		
		temp1.detach()
		temp2.detach()
		#temp3.detach()
		e.detach()
		show_and_save(f"Results/{i}_{key}", make_grid((e*0.5+0.5).cpu(),max(1,bs//4)))
	
	a.detach()
	b.detach()
	c.detach()
	#d.detach()
	temp.detach()
	show_and_save(f'Results/{i}_recreated', make_grid((c*0.5+0.5).cpu(),max(1,bs//4)))
	show_and_save(f'Results/{i}_sampled', make_grid((a*0.5+0.5).cpu(),max(1,bs//4)))
	#show_and_save(f"Results/{i}_attrib", make_grid((d*0.5+0.5).cpu(),bs//4))
