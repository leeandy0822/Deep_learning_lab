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
        
def plot_rec(validate_x, validate_cond, modules, epoch, args):
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    
    gen_seq = []
    gen_seq.append(validate_x[0])
    x_in = validate_x[0]
    h_seq = [modules['encoder'](validate_x[i]) for i in range(args.n_past+args.n_future)]
    
    for i in range(1, args.n_past+args.n_future):
        
        h_target = h_seq[i][0].detach()
        if args.last_frame_skip or i < args.n_past:	
            h, skip = h_seq[i-1]
        else:
            h, _ = h_seq[i-1]
            
        h = h.detach()
        z_t, mu, logvar = modules['posterior'](h_target)
        if i < args.n_past:
            modules['frame_predictor'](torch.cat([h, z_t, validate_cond], 1)) 
            gen_seq.append(validate_x[i])
        else:
            h_pred = modules['frame_predictor'](torch.cat([h, z_t, validate_cond], 1)).detach()
            x_pred = modules['decoder']([h_pred, skip]).detach()
            gen_seq.append(x_pred)
   
    to_plot = []
    nrow = min(args.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(args.n_past+args.n_future):
            row.append(gen_seq[t][i]) 
        to_plot.append(row)
    fname = '%s/gen/rec_%d.png' % (args.log_dir, epoch) 
    save_tensors_image(fname, to_plot)
    
def pred(x, cond, modules, epoch, args):
    
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


def plot_pred(x, validate_cond, modules, epoch, args ):
    
    y_pred = pred(x, validate_cond, modules, epoch,  args)
    
    to_plot_gt = x[0].squeeze(0)
    to_plot_pred = y_pred[:,0].squeeze(1)
    
    filename_git_gt = './test_git/'+str(epoch) + "_gt.git"
    filename_gif_pred = './test_git/'+str(epoch) + "_pred.git"
            

    save_gif(filename=filename_git_gt, inputs = to_plot_gt)
    
    save_gif(filename=filename_gif_pred, inputs = to_plot_pred)

def save_gif(filename, inputs, duration = 0.5):
    images = []
    
    for tensor in inputs:
        img = tensor/2 + 0.5
        img = img*255
        npImg = img.cpu().numpy().astype(np.uint8)
        npImg = np.transpose(npImg, (1, 2, 0))
        images.append(npImg)
    imageio.mimsave(filename, images, duration=duration)
    
def save_image(filename, tensor):
    img = make_image(tensor)
    img.save(filename)

def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    # pdb.set_trace()
    img = tensor.numpy().reshape(5069, 779, 3)
    return Image.fromarray(np.uint8(img)).convert('RGB')
  

def normalize_data(dtype, sequence):
    
    sequence.transpose_(0, 1)
    sequence.transpose_(3, 4).transpose_(2, 3)

    return sequence_input(sequence, dtype)


def sequence_input(seq, dtype):
    return [Variable(x.type(dtype)) for x in seq]

def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))
    
def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                   (i+1) * y_dim + i * padding].copy_(image)
        return result


def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(filename, images)

