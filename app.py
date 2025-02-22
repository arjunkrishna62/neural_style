import os
import streamlit as st

import torch
import numpy as np
import cv2
from PIL import Image

from main import neural_style_transfer

@st.cache_data
def prepare_imgs(content_im, style_im, RGB=False):
    """ Return scaled RGB images as numpy array of type np.uint8 """    
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
    
    # pass from BGR (OpenCV) to RGB:
    if not RGB:
        content_im = cv2.cvtColor(content_im, cv2.COLOR_BGR2RGB)
        style_im   = cv2.cvtColor(style_im, cv2.COLOR_BGR2RGB)
    
    return content_im, style_im

    
def print_info_NST():
    """ Print basic information about Neural Style Transfer within the app.
    """    
    st.markdown("""
                ## What is NST?
                **NST** (*Neural Style Transfer*) is a Deep Learning
                technique to generate an image based on the content and
                style of two different images.  
                Let's have a look to an
                example (left column, top and bottom, are the *content*
                and *style*, respectively.):""")

    # Show exemplar images:
    root_content = os.path.join('data', 'content-images', 'lion.jpg')
    root_style = os.path.join('data', 'style-images', 'wave.jpg')
    
    content_im = cv2.imread(root_content)
    style_im = cv2.imread(root_style)    
    im_cs, im_ss = prepare_imgs(content_im, style_im)    
    im_rs = cv2.imread(os.path.join('data', 'output-images', 'clion_swave_sample.jpg'))
    
    col1, col2 = st.columns([1,2.04])
    col1.header("Base")
    col1.image(im_cs, use_container_width=True)
    col1.image(im_ss, use_container_width=True)
    col2.header("Result")
    col2.image(im_rs, use_container_width=True, channels="BGR")
    
    # Information about the parameters:
    st.markdown("""
            ## Parameters at the left sidebar
            ### Weights of the Loss function (lambdas)
            """)
    st.latex(r"""
            \mathcal{L}(\lambda_{\text{content}}, 
            \lambda_{\text{style}}, \lambda_{\text{variation}}) =
            \lambda_{\text{content}}\mathcal{L}_{\text{content}} +
            \lambda_{\text{style}}\mathcal{L}_{\text{style}} +
            \lambda_{\text{variation}}\mathcal{L}_{\text{variation}}
            """)
    st.markdown("""
            - **Content**: A higher values increases the influence of the *Content* image,
            - **Style**: A higher value increases the influence of the *Style* image,
            - **Variation**: A higher value make the resulting image to look more smoothed.
            """)
    st.markdown("""
            ### Number of iterations
            Its value defines the duration of the optimization process.
            A higher number will make the optimization process longer.
            Thereby if the image looks unoptimized, try to increase its number
            (or tune the weights of the loss function).
            """)
    st.markdown("""
            ### Save result
            If this option is checked, then once the optimization finishes,
            the image will be saved in the computer 
            (in the same folder where the app.py file of this project is located)
            """)

if __name__ == "__main__":
    
    # app title and sidebar:
    st.title('Apply Neural Style Transfer')

    # Select what to do:
    st.sidebar.title('Configuration')
    st.sidebar.subheader('Select what page to show')
    options = ['About NST', 'Run NST on pair of images']
    app_mode = st.sidebar.selectbox('Select what to do/ show:',
                                    options
                                    )
    
    # Set parameters to tune at the sidebar:
    st.sidebar.title('Parameters')
    #Weights of the loss function
    st.sidebar.subheader('Weights')
    step=1e-1
    cweight = st.sidebar.number_input("Content", value=1e-3, step=step, format="%.5f")
    sweight = st.sidebar.number_input("Style", value=1e-1, step=step, format="%.5f")
    vweight = st.sidebar.number_input("Variation", value=0.0, step=step, format="%.5f")
    # number of iterations
    st.sidebar.subheader('Number of iterations')
    niter = st.sidebar.number_input('Iterations', min_value=1, max_value=1000, value=20, step=1)
    # save or not the image:
    st.sidebar.subheader('Save or not the stylized image')
    save_flag = st.sidebar.checkbox('Save result')
    
    # Show the page of the selected page:
    if app_mode == options[0]:
        print_info_NST()
        
    elif app_mode == options[1]:        
        st.markdown("### Upload the pair of images to use")        
        col1, col2 = st.columns(2)
        im_types = ["png", "jpg", "jpeg"]
        
        # Create file uploaders in a two column layout, as well as
        # placeholder to later show the images uploaded:
        with col1:
            file_c = st.file_uploader("Choose CONTENT Image", type=im_types)
            imc_ph = st.empty()            
        with col2: 
            file_s = st.file_uploader("Choose STYLE Image", type=im_types)
            ims_ph = st.empty()
        
        # if both images have been uploaded then preprocess and show them:
        if all([file_s, file_c]):
            # preprocess:
            im_c = np.array(Image.open(file_c))
            im_s = np.array(Image.open(file_s))
            im_c, im_s = prepare_imgs(im_c, im_s, RGB=True)
            
            # Show images:
            imc_ph.image(im_c, use_column_width=True)
            ims_ph.image(im_s, use_column_width=True) 
        
        st.markdown("""
                    ### When ready, START the image generation!
                    """)
        
        # button for starting the stylized image:
        start_flag = st.button("START", help="Start the optimization process")
        bt_ph = st.empty() # Possible message above the button
    
        if start_flag:
            if not all([file_s, file_c]):
                bt_ph.markdown("You need to **upload the images** first! :)")
            elif start_flag:
                bt_ph.markdown("Optimizing...")
                
        if start_flag and all([file_s, file_c]):
            # Create progress bar:
            progress = st.progress(0.)
            # Create place-holder for the stylized image:
            res_im_ph = st.empty()
            # config the NST function:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # parent directory of this file:
            parent_dir = os.path.dirname(__file__)
            out_img_path = os.path.join(parent_dir, "app_stylized_image.jpg")
            cfg = {
                'output_img_path' : out_img_path,
                'style_img' : im_s,
                'content_img' : im_c,
                'content_weight' : cweight,
                'style_weight' : sweight,
                'tv_weight' : vweight,
                'optimizer' : 'lbfgs',
                'model' : 'vgg19',
                'init_metod' : 'random',
                'running_app' : True,
                'res_im_ph' : res_im_ph,
                'save_flag' : save_flag,
                'st_bar' : progress,
                'niter' : niter
                }
            
            result_im = neural_style_transfer(cfg, device)
            # res_im_ph.image(result_im, channels="BGR")
            bt_ph.markdown("This is the resulting **stylized image**!")