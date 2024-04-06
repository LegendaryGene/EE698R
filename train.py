import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)
from dataloader import dataloader
from models import VAE_GAN,Discriminator
from utils import show_and_save, RollingMeasure

batch_size = 16
epochs=61
lr=3e-4
alpha=0.1
gamma=15
run_num=2
load_from=1

data_loader=dataloader(batch_size)
gen=VAE_GAN().to(device)
discrim=Discriminator().to(device)
writer = SummaryWriter(comment="_DeepFashion")

#gen.load_state_dict(torch.load(f"./saves/run{load_from}/gen_60.pth"))
#discrim.load_state_dict(torch.load("./saves/run{load_from}/discrim_60.pth"))
#print("Loaded Successfully!")

real_batch = next(iter(data_loader))
criterion=nn.BCELoss().to(device)
optim_E=torch.optim.RMSprop(gen.encoder.parameters(), lr=lr)
optim_D=torch.optim.RMSprop(gen.decoder.parameters(), lr=lr)
optim_Dis=torch.optim.RMSprop(discrim.parameters(), lr=lr*alpha)
z_fixed=Variable(torch.randn((batch_size,256))).to(device)
x_fixed=Variable(real_batch[0]).to(device)
original=real_batch[0]

temp1 = np.zeros([23, 256])
temp2 = np.zeros([23, 256])
Dictionary_no = {
  "MEN-Tees_Tanks": [0, 0],
  "MEN-Shirts_Polos": [1, 0],
  "MEN-Denim": [2, 0],
  "MEN-Pants": [3, 0],
  "MEN-Jackets_Vests": [4, 0],
  "MEN-Shorts": [5, 0],
  "MEN-Suiting": [6, 0],
  "MEN-Sweaters": [7, 0],
  "MEN-Sweatshirts_Hoodies": [8, 0],
  "WOMEN-Tees_Tanks": [9, 0],
  "WOMEN-Blouses_Shirts": [10, 0],
  "WOMEN-Cardigans": [11, 0],
  "WOMEN-Denim": [12, 0],
  "WOMEN-Dresses": [13, 0],
  "WOMEN-Graphic_Tees": [14, 0],
  "WOMEN-Jackets_Coats": [15, 0],
  "WOMEN-Leggings": [16, 0],
  "WOMEN-Pants": [17, 9927],
  "WOMEN-Rompers_Jumpsuits": [18, 0],
  "WOMEN-Shorts": [19, 0],
  "WOMEN-Skirts": [20, 0],
  "WOMEN-Sweaters": [21, 0],
  "WOMEN-Sweatshirts_Hoodies": [22, 0]}

Dictionary_yes = {
  "MEN-Tees_Tanks": [0, 0],
  "MEN-Shirts_Polos": [1, 0],
  "MEN-Denim": [2, 0],
  "MEN-Pants": [3, 0],
  "MEN-Jackets_Vests": [4, 0],
  "MEN-Shorts": [5, 0],
  "MEN-Suiting": [6, 0],
  "MEN-Sweaters": [7, 0],
  "MEN-Sweatshirts_Hoodies": [8, 0],
  "WOMEN-Tees_Tanks": [9, 0],
  "WOMEN-Blouses_Shirts": [10, 0],
  "WOMEN-Cardigans": [11, 0],
  "WOMEN-Denim": [12, 0],
  "WOMEN-Dresses": [13, 0],
  "WOMEN-Graphic_Tees": [14, 0],
  "WOMEN-Jackets_Coats": [15, 0],
  "WOMEN-Leggings": [16, 0],
  "WOMEN-Pants": [17, 0],
  "WOMEN-Rompers_Jumpsuits": [18, 0],
  "WOMEN-Shorts": [19, 0],
  "WOMEN-Skirts": [20, 0],
  "WOMEN-Sweaters": [21, 0],
  "WOMEN-Sweatshirts_Hoodies": [22, 0]}

for epoch in range(epochs):
  
  prior_loss_list,gan_loss_list,recon_loss_list=RollingMeasure(),RollingMeasure(),RollingMeasure()
  dis_real_list,dis_fake_list,dis_prior_list=RollingMeasure(),RollingMeasure(),RollingMeasure()

  for i, (data,label_clothing) in enumerate(data_loader, 0):
    bs=data.size()[0]
    
    ones_label=Variable(torch.ones(bs,1)).to(device)
    zeros_label=Variable(torch.zeros(bs,1)).to(device)
    zeros_label1=Variable(torch.zeros(bs,1)).to(device)
    datav = Variable(data).to(device)
    mean, logvar, rec_enc = gen(datav)
    z_p = Variable(torch.randn(bs,256)).to(device)
    x_p_tilda = gen.decoder(z_p)
    
    for j in range(bs):
        meann = mean[j].clone()
        meannn = meann.detach().cpu().numpy()
        clothing = label_clothing[j]
        temp1[Dictionary_yes[clothing][0]] = temp1[Dictionary_yes[clothing][0]] + meannn
        Dictionary_yes[clothing][1] = Dictionary_yes[clothing][1] + 1
        for key in Dictionary_no:
            if(key != clothing):
                temp2[Dictionary_no[key][0]] = temp2[Dictionary_no[key][0]] + meannn
                Dictionary_no[key][1] = Dictionary_no[key][1] + 1

    
    output = discrim(datav)[0]
    errD_real = criterion(output, ones_label)
    dis_real_list(errD_real.item())
    output = discrim(rec_enc)[0]
    errD_rec_enc = criterion(output, zeros_label)
    dis_fake_list(errD_rec_enc.item())
    output = discrim(x_p_tilda)[0]
    errD_rec_noise = criterion(output, zeros_label1)
    dis_prior_list(errD_rec_noise.item())
    gan_loss = errD_real + errD_rec_enc + errD_rec_noise
    gan_loss_list(gan_loss.item())
    optim_Dis.zero_grad()
    gan_loss.backward(retain_graph=True)
    optim_Dis.step()


    output = discrim(datav)[0]
    errD_real = criterion(output, ones_label)
    output = discrim(rec_enc)[0]
    errD_rec_enc = criterion(output, zeros_label)
    output = discrim(x_p_tilda)[0]
    errD_rec_noise = criterion(output, zeros_label1)
    gan_loss = errD_real + errD_rec_enc + errD_rec_noise
    

    x_l_tilda = discrim(rec_enc)[1]
    x_l = discrim(datav)[1]
    rec_loss = ((x_l_tilda - x_l) ** 2).mean()
    err_dec = gamma * rec_loss - gan_loss 
    recon_loss_list(rec_loss.item())
    optim_D.zero_grad()
    err_dec.backward(retain_graph=True)
    optim_D.step()
    
    mean, logvar, rec_enc = gen(datav)
    x_l_tilda = discrim(rec_enc)[1]
    x_l = discrim(datav)[1]
    rec_loss = ((x_l_tilda - x_l) ** 2).mean()
    prior_loss = 1 + logvar - mean.pow(2) - logvar.exp()
    prior_loss = (-0.5 * torch.sum(prior_loss))/torch.numel(mean.data)
    prior_loss_list(prior_loss.item())
    err_enc = prior_loss + 5*rec_loss

    optim_E.zero_grad()
    err_enc.backward(retain_graph=True)
    optim_E.step()

    if i % 10 == 0:
            print('[%d/%d][%d/%d]\tLoss_gan: %.4f\tLoss_prior: %.4f\tRec_loss: %.4f\tdis_real_loss: %0.4f\tdis_fake_loss: %.4f\tdis_prior_loss: %.4f'
                  % (epoch,epochs, i, len(data_loader),
                     gan_loss.item(), prior_loss.item(),rec_loss.item(),errD_real.item(),errD_rec_enc.item(),errD_rec_noise.item()))

  for key in Dictionary_no:
      temp1[Dictionary_yes[key][0]] = temp1[Dictionary_yes[key][0]]/Dictionary_yes[key][1]
      temp2[Dictionary_no[key][0]] = temp2[Dictionary_no[key][0]]/Dictionary_no[key][1]
      Dictionary_yes[key][1] = 0
      Dictionary_no[key][1] = 0

  if(epoch%15 == 0):
    torch.save(gen.state_dict(), f"./saves/run{run_num}/gen_{epoch}.pth")
    torch.save(discrim.state_dict(), f"./saves/run{run_num}/discrim_{epoch}.pth")
    np.save(f"./saves/run{run_num}/Yes_{epoch}.npy", temp1)
    np.save(f"./saves/run{run_num}/No_{epoch}.npy", temp2)
    print("Saved!")

  a,_,b=gen(x_fixed)
  a=gen.decoder(a)
  a.detach()
  b=b.detach()
  c=gen.decoder(z_fixed)
  c=c.detach()

  writer.add_scalar("Gen Prior Loss", prior_loss_list.measure, epoch)
  writer.add_scalar("Gen Reconstructiom Loss", recon_loss_list.measure, epoch)
  writer.add_scalar("GAN Loss", gan_loss_list.measure, epoch)
  writer.add_scalar("Disc Prior Loss", dis_prior_list.measure, epoch)
  writer.add_scalar("Disc Fake Loss", dis_fake_list.measure, epoch)
  writer.add_scalar("Disc Real Loss", dis_real_list.measure, epoch)
  writer.add_image("Original", make_grid((original*0.5+0.5).cpu(),8), epoch)
  writer.add_image("MLE", make_grid((a*0.5+0.5).cpu(),8), epoch)
  writer.add_image("MLE + Var*Randn", make_grid((b*0.5+0.5).cpu(),8), epoch)
  writer.add_image("Noise Reconstructed", make_grid((c*0.5+0.5).cpu(),8), epoch)
