import os
import sys
import errno
import cv2
import numpy as np
import argparse
import torch
from torch import nn
from torch.optim import LBFGS
from torchvision import transforms
from models.vgg import Vgg16, Vgg19

# values from: https://pytorch.org/vision/stable/models.html
IMAGENET_MEAN_255 = [255*a for a in [0.485, 0.456, 0.406]]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

def bgr2rgb(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

def prepare_imgs(content_img_path, style_img_path):
    """ Return scaled RGB images as numpy array of type np.uint8 """
    
    if not os.path.exists(content_img_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), content_img_path)
    if not os.path.exists(style_img_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), style_img_path)
    
    content_im = cv2.imread(content_img_path)
    style_im = cv2.imread(style_img_path)
    
    # check sizes in order to avoid huge computation times:
    h,w,c = content_im.shape
    ratio = 1.
    if h > 512:
        ratio = 512./h
    if (w > 512) and (w>h):
        ratio = 512./w
    content_im = cv2.resize(content_im, dsize=None, fx=ratio, fy=ratio,
                            interpolation=cv2.INTER_CUBIC)        
    # reshape style_im to match the content_im shape 
    # (method followed in Gatys et al. paper):
    style_im = cv2.resize(style_im, content_im.shape[1::-1], cv2.INTER_CUBIC)
    
    # show initial images:
    cv2.imshow('style', style_im)
    cv2.imshow('content', content_im)
    # pass from BGR (OpenCV) to RGB:
    content_im = cv2.cvtColor(content_im, cv2.COLOR_BGR2RGB)
    style_im   = cv2.cvtColor(style_im, cv2.COLOR_BGR2RGB)    
    return content_im, style_im


def gram_matrix(x, normalize=True):
    c, h, w = x.shape
    # Get F^l (Gatys et. al notation) for every l:
    Fs = x.view(c,h*w)
    # Gram matrix:
    gram = Fs @ Fs.T
    if normalize:
        gram /= c*h*w # in order to 
    return gram

def build_loss(cfg, content_gt, style_gt, features, opt_im, criterion):
    
    # CONTENT:
    if cfg['model'].lower()=='vgg19': 
        content_loss = criterion(content_gt, features[2].squeeze(0))
    else:
        content_loss = criterion(content_gt, features[0].squeeze(0))
    
    # STYLE:
    # obtain gram matrices for the predicted features:
    current_style = [gram_matrix(ft_maps.squeeze(0)) for ft_maps in features]
    # style loss:
    style_loss = 0.0
    for k, (gm, gm_gt) in enumerate(zip(current_style, style_gt)):
        if k != 4:
            style_loss += criterion(gm,gm_gt)
    style_loss /= len(current_style)
    
    # TOTAL VRARIATION LOSS:
    tv_loss = torch.sum(torch.abs(opt_im[:, :, :-1] - opt_im[:, :, 1:])) + \
              torch.sum(torch.abs(opt_im[:, :-1, :] - opt_im[:, 1:, :]))
        
    # Final loss:
    total_loss = cfg['content_weight']*content_loss + cfg[
        'style_weight']*style_loss + cfg['tv_weight']*tv_loss    
    return total_loss

def unNormalize(tensor, mean=IMAGENET_MEAN_255, std=IMAGENET_STD):
    """Convert normalized tensor to its unnormalized representation"""
    x = tensor.clone()
    for channel, mean_c, std_c in zip(x, mean, std):
        channel.mul_(std_c).add_(mean_c) # in-place operations
    return x      
    
def tensor2img(x):
    """ Get unnormalized image and convert to numpy (in order to use OpenCV)
    """
    # get unnormalize image and convert to numpy (in order to use OpenCV):
    x_un = unNormalize(x)
    #x_un    = x.mul(torch.tensor(IMAGENET_STD)).add(torch.tensor(IMAGENET_MEAN_255))
    x_numpy = x_un.cpu().numpy()
    # get RGB representation:
    im = bgr2rgb(x_numpy.transpose(1,2,0))
    im = np.clip(im,0,255)
    im = im.astype(np.uint8)
    return im


def show_image(x, cfg, save=True):
    """ Show image related to tensor x
    """
    bgr_im = tensor2img(x)  
    
    if cfg["running_app"]:
        cfg["res_im_ph"].image(bgr_im, channels="BGR")
        
    else:
        cv2.imshow('n-s-t', bgr_im)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            if save:
                cv2.imwrite(cfg['output_img_path'], bgr_im)
                # generate triplet image: [style, content, n-s-t]:
                triplet = np.concatenate((bgr2rgb(cfg['style_img']), 
                            bgr2rgb(cfg['content_img']), bgr_im), axis=1)
                triplet_name = cfg['output_img_path'][:cfg['output_img_path'
                                                ].index('.')] + '_triplet.jpg'
                cv2.imwrite(triplet_name, triplet)
            sys.exit('terminating...')
            

def neural_style_transfer(cfg, device):
    """
    Neural style transfer between the content image
    and the style image following the method 
    proposed in the work of Gatys et al.:
    https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
    """
    
    # as explained in: https://pytorch.org/vision/stable/models.html
    # the vgg used was trained with images in a [0,1] pixel scale and 
    # were normlized (zero-mean and unit-std). This is done hereafter.
    # However, following  https://github.com/gordicaleksa/pytorch-neural-style-transfer
    # we operate on a [0,255] scale (I think this makes more stable gradients):
    transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Lambda(lambda x: x.mul(255.)),
             transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD)
             ])
    # apply them:
    content_img = transform(cfg["content_img"]).to(device)
    style_img = transform(cfg["style_img"]).to(device)
        
    # create initial image as gaussian noise. As a sidenote, the last 
    # operation is the in-place method requires_grad_() in order to keep 
    # optimizing_img as a leaf variable:
    optimizing_img = (90*torch.randn(content_img.shape).to(device)).requires_grad_()
    
    # get model and ground-truth data:
    if cfg['model'].lower()=='vgg19':        
        model = Vgg19().to(device)
        content_gt = model(content_img[None])[2].squeeze(0)
    elif cfg['model'].lower()=='vgg16':
        model = Vgg16().to(device)
        content_gt = model(content_img[None])[0].squeeze(0)
    else:
        raise ValueError(f'{cfg["model"]} is not implemented. Give it a try to vgg19 or vgg16.')
    style_gt = [gram_matrix(ft_maps.squeeze(0)) for ft_maps in model(style_img[None])]
    
    # optimization (Gatys et al. recommend using LBFGS method):
    niter = 10
    optimizer = LBFGS([optimizing_img], max_iter=niter, 
                      line_search_fn='strong_wolfe', history_size=10)
    criterion = nn.MSELoss(reduction='mean')
    
    for i in range(cfg["niter"]):#niter//10
        # LBFGS requires a closure function that computes the loss and clears
        # the gradient, see:
        # https://pytorch.org/docs/1.9.1/optim.html#:~:text=and%20return%20it.-,Example,-%3A
        def closure():
            optimizer.zero_grad()
            features = model(optimizing_img[None])
            loss = build_loss(cfg, content_gt, style_gt, features, optimizing_img, criterion)
            loss.backward()
            with torch.no_grad():
                show_image(optimizing_img, cfg, cfg['save_flag'])
            return loss
        optimizer.step(closure)
        if cfg["running_app"]:
            cfg["st_bar"].progress((i+1)/cfg["niter"])
    
    with torch.no_grad():
        im = tensor2img(optimizing_img)
    
    if cfg['save_flag']:
        cv2.imwrite(cfg['output_img_path'], im)
        # generate triplet image: [style, content, n-s-t]:
        triplet = np.concatenate((bgr2rgb(cfg['style_img']), 
                    bgr2rgb(cfg['content_img']), im), axis=1)
        triplet_name = cfg['output_img_path'][:cfg['output_img_path'
                                        ].index('.')] + '_triplet.jpg'
        cv2.imwrite(triplet_name, triplet)
    
    return im
    

if __name__ == '__main__':
    
    # parent directory of this file:
    parent_dir = os.path.dirname(__file__)
    # directory where the data will be stored:
    data_dir = os.path.join(parent_dir, 'data')
    
    # directories for content, style and output images:
    content_dir = os.path.join(data_dir, 'content-images')
    style_dir   = os.path.join(data_dir, 'style-images')
    output_dir  = os.path.join(data_dir, 'output-images') 
    os.makedirs(output_dir, exist_ok=True)
    
    # pass configuration parameters with argparse:
    parser = argparse.ArgumentParser()    
    parser.add_argument("--content_img_name", type=str, 
                        help=""" name and extension of the CONTENT image 
                        located at the "data/content-images" folder. 
                        For example: lion.jpg
                        """, default='lion.jpg')
    parser.add_argument("--style_img_name", type=str, 
                        help=""" name and extension of the STYLE image 
                        located at the "data/style-images" folder. 
                        For example: wave.jpg
                        """, default='wave.jpg')
                        
    parser.add_argument("--content_weight", type=float, 
                        help="""weight (importance) of the CONTENT image in the 
                        resulting stylized image""", default=1e-3)
    parser.add_argument("--style_weight", type=float, 
                        help="""weight (importance) of the STYLE image in the 
                        resulting stylized image""", default=1e-1)
    parser.add_argument("--tv_weight", type=float, 
                        help="""The higher value of this weight, 
                        the higher degree of smoothness in the stylized 
                        image""", default=0.)
    
    parser.add_argument("--model", type=str, choices=["vgg16,vgg19"], 
                        help="""Select which VGG model (vgg16 or vgg19) to use
                        to define the perceptual losses (recommendation:
                        choose vgg19 as it offers better results)               
                        """, default="vgg19")
    parser.add_argument("--save_stylized_image", help="""write this flag if 
                        you want to save the resulting stylized image""",
                        action="store_true")
    parser.add_argument("--niter", type=int, help=""" Number of iterations
                        to perform during the optimization process""",
                        default=30)
    args = parser.parse_args()
    
    # content_img_name = 'lion.jpg'
    content_img_path = os.path.join(content_dir, args.content_img_name)
    # style_img_name = 'wave.jpg'
    style_img_path = os.path.join(style_dir, args.style_img_name)
    
    # use cuda or cpu:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load RGB images as np.uint8 arrays and scale them if needed:
    c_img, s_img = prepare_imgs(content_img_path, style_img_path)
    
    # path of the output image:
    out_name = 'c' + args.content_img_name[:args.content_img_name.index('.')] + '_s' + \
        args.style_img_name[:args.style_img_name.index('.')] + '.jpg'
    out_img_path = os.path.join(output_dir, out_name)
    
    cfg = {
        'output_img_path' : out_img_path,
        'style_img' : s_img,
        'content_img' : c_img,
        'content_weight' : args.content_weight,
        'style_weight' : args.style_weight,
        'tv_weight' : args.tv_weight,
        'optimizer' : 'lbfgs',
        'model' : args.model,
        'init_metod' : 'random',
        'running_app' : False,
        'res_im_ph' : None,
        'save_flag' : args.save_stylized_image,
        'st_bar' : None,
        'niter' : args.niter
        }
    
    out = neural_style_transfer(cfg, device)
    
    # Mantain the resulting image until a key of the keyboard is pressed:
    cv2.destroyAllWindows()
    cv2.imshow('press any key to finsih the execution', out)
    cv2.waitKey(0)
