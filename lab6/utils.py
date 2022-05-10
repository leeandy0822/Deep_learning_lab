import math
from operator import pos
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
import imageio
import os



def kl_criterion(mu, logvar, args):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  
  
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= args.batch_size  
  return KLD
    
def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c]) 
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def mse_metric(x1, x2):
    # x1 = x1.data.cpu().numpy()
    # x2 = x2.data.cpu().numpy()
    x1 = torch.tensor(x1)
    x2 = torch.tensor(x2)
    err = torch.sum(((x1 - x2) ** 2))
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_x(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i].detach().cpu().numpy()
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def finn_psnr(x, y, data_range=1.):
    mse = ((x - y)**2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
def pred(x, cond, modules, args):
    
    x = x.float()
    # print(x.shape)

    gen_seq = []
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()

    with torch.no_grad():
        h_seq = [ modules['encoder'](x[:,i]) for i in range(args.n_past + args.n_future)] # x : [10,12,3,64,64] h_seq : [12,10,128]

        # te_em = nn.Embedding(4, 7)
        # tense_embedding = te_em(cond.view(-1, 1))
        #print(h_seq[0][0].size())
        for i in range(1, args.n_past + args.n_future):
            h_target = h_seq[i][0]
            #print(h_seq[i][0].size())  # [10,128]

            if args.last_frame_skip or i < args.n_past:	
                h = h_seq[i-1][0] 
                skip = h_seq[i-1][1]
            else:
                h = h_seq[i-1][0]
                
            if i > 1:
                previous_img = x_pred
                pr_latent = modules['encoder'](previous_img)
                h_no_teacher = pr_latent[0]
                
            else:
                h_no_teacher = h    
            c = cond[:, i].float()

            z_t, mu, logvar = modules['posterior'](h_target)
            
            if i > 1:
                h_pred = modules['frame_predictor'](torch.cat([h, z_t,c], 1))
            else:
                h_pred = modules['frame_predictor'](torch.cat([h_no_teacher, z_t,c], 1))

            x_pred = modules['decoder']([h_pred, skip])
            
            if i > 1 :
                gen_seq.append(x_pred.data.cpu().numpy())
        
    return gen_seq


def plot_pred(validate_seq, validate_cond, modules, epoch, args, device, mode, priority: bool):
    

    prior = '_with_prior' if priority else '_without_prior'
    pred_mode = 'val/' if mode =='validate' else 'test/'
    filename_gif_gt = './gen_gif/'+pred_mode +str(epoch)+ prior+ '_gt'
    filename_gif_pred = './gen_gif/'+pred_mode +str(epoch)+ prior+ '_pred'

    y_pred = pred(validate_seq, validate_cond, modules, args)

    validate_x = validate_seq.cpu().numpy()
    number_of_batch = 0
    to_plot_gt = torch.tensor(validate_x[:,number_of_batch])

    # total 10 frames   
    # [10, [12, 3, 64, 64]]
    y_pred = np.array(y_pred)
    to_plot_pred = torch.tensor(y_pred[:,number_of_batch])


    if not os.path.isdir('./gen_gif/'+pred_mode):
        os.makedirs('./gen_gif/'+pred_mode)
        
    save_gif_and_jpg(filename=filename_gif_pred, inputs = to_plot_pred)
    save_gif_and_jpg(filename=filename_gif_gt, inputs= to_plot_gt)

def save_gif_and_jpg(filename, inputs, duration = 0.5):
    # inputs shape [12,3,64,64]
    images = []
    for tensor in inputs:
        #img = tensor/2 + 0.5
        img = tensor*255
        npImg = img.cpu().numpy().astype(np.uint8)
        npImg = np.transpose(npImg, (1,2,0))
        images.append(npImg)
        
    filename_gif = filename+'.gif'
    filename_jpg = filename+'.jpg'
    imageio.mimsave(filename_gif, images, duration = duration)
    images_seq = np.concatenate(images, 1)
    imageio.imwrite(filename_jpg, images_seq)