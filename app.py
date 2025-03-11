import os
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import sys
from dotenv import load_dotenv
from datetime import datetime
from streamlit_drawable_canvas import st_canvas
import io
from torchvision import transforms

from typing import Optional, Dict, List


from main import neural_style_transfer
from text_to_image import TextToImageGenerator
from src.pixel2turbo import Pix2Pix_Turbo
from using_cnn import StyleTransferCNN

load_dotenv()

# Import the Pixel2Turbo model
p = "src/"
sys.path.append(p)
from src.pixel2turbo import Pix2Pix_Turbo




@st.cache_data
def prepare_imgs(content_im, style_im, RGB=False):
    """ Return scaled RGB images as numpy array of type np.uint8 """ 
    if content_im is None or style_im is None:
        st.error("One or both input images are invalid")
        return None, None   
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
    style_im = cv2.resize(style_im, content_im.shape[1::-1], cv2.INTER_CUBIC)
    
    # pass from BGR (OpenCV) to RGB:
    if not RGB:
        content_im = cv2.cvtColor(content_im, cv2.COLOR_BGR2RGB)
        style_im   = cv2.cvtColor(style_im, cv2.COLOR_BGR2RGB)
    
    return content_im, style_im

def initialize_text2img_generator():
    """Initialize the Text-to-Image generator with API keys"""
    try:
        # Try to get API key from environment variables first
        api_key_stability = os.getenv('STABILITY_API_KEY')
        api_key_openai = os.getenv('OPENAI_API_KEY')
        
        # If keys are not in environment variables, get them from session state
        if 'stability_api_key' in st.session_state:
            api_key_stability = st.session_state.stability_api_key
        
        if 'openai_api_key' in st.session_state:
            api_key_openai = st.session_state.openai_api_key
        
        # Combine API keys into a SINGLE dictionary
        combined_api_keys = {'stability': api_key_stability, 'openai': api_key_openai}
        
        return TextToImageGenerator(combined_api_keys)  # Pass the combined dictionary as ONE argument
    except Exception as e:
        st.error(f"Error initializing generator: {str(e)}")
        return None
            
        #st.write("Initializing generator with API key")
        
        # Update to only use Stability API
        combined_api_keys = {'stability': api_key_stability, 'openai': api_key_openai}
        generator = TextToImageGenerator(api_keys=combined_api_keys)
        return generator
    except Exception as e:
        st.error(f"Error initializing generator: {str(e)}")
        return None

# Initialize session state variables at the beginning of your app
if 'txt2img_model' not in st.session_state:
    st.session_state.txt2img_model = 'stable-diffusion-v1-5'  # Default value
if 'txt2img_size' not in st.session_state:
    st.session_state.txt2img_size = '512x512'  # Default value
if 'num_images' not in st.session_state:
    st.session_state.num_images = 1
if 'guidance_scale' not in st.session_state:
    st.session_state.guidance_scale = 7.5
if 'steps' not in st.session_state:
    st.session_state.steps = 30
if 'seed' not in st.session_state:
    st.session_state.seed = None
if 'save_txt2img_flag' not in st.session_state:
    st.session_state.save_txt2img_flag = False
if 'generator_initialized' not in st.session_state:
    st.session_state.generator_initialized = False

# Only initialize the generator once and store it in a global variable
# You might want to put this at the beginning of your app
@st.cache_resource
def get_generator():
    return initialize_text2img_generator()

# Get or create the generator
generator = get_generator()


@st.cache_resource
def load_pixel2turbo_model(model_type):
   
    if model_type == "Edge to Image":
        model = Pix2Pix_Turbo(pretrained_name="edge_to_image")
    elif model_type == "Sketch to Image (Stochastic)":
        model = Pix2Pix_Turbo(pretrained_name="sketch_to_image_stochastic")
    else:
        model = Pix2Pix_Turbo()  # Default initialization
        
    model.set_eval()  # Set model to evaluation mode
    return model

def process_pixel2turbo(model, input_image, prompt, randomness):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to appropriate device
    model = model.to(device)
    
    # Convert PIL image to tensor
    input_tensor = torch.from_numpy(input_image).float().to(device) / 127.5 - 1.0
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
    
    # Process deterministically or with randomness
    deterministic = randomness <= 0.0
    
    if deterministic:
        output = model(input_tensor, prompt=prompt, deterministic=True)
    else:
        # Generate noise map for stochastic generation
        shape = input_tensor.shape
        noise = torch.randn((shape[0], 4, shape[2]//8, shape[3]//8), device=device)
        output = model(input_tensor, prompt=prompt, deterministic=False, r=1.0-randomness, noise_map=noise)
    
    # Convert output tensor to numpy array
    output_image = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    output_image = ((output_image + 1.0) * 127.5).astype(np.uint8)
    
    return output_image

def display_generated_images(images, container):
    """Display generated images in the provided container."""
    # Create columns based on number of images
    if len(images) > 2:
        # For 3 or 4 images, use a 2x2 grid
        rows = [container.columns(2) for _ in range(2)]
        cols = [col for row in rows for col in row]
    else:
        # For 1 or 2 images, use a single row
        cols = container.columns(len(images))
    
    # Display each image with a download button
    for i, (col, image) in enumerate(zip(cols, images)):
        with col:
            st.image(image, caption=f"Generated Image {i+1}", use_container_width=True)
            
            # Convert to PIL Image if it's a numpy array
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image  # Already a PIL Image
            
            # Convert to bytes for download
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            # Add download button
            st.download_button(
                label="Download",
                data=byte_im,
                file_name=f"generated_image_{i+1}.png",
                mime="image/png",
                use_container_width=True
            )
            
            # Save image if option is selected
            if 'save_txt2img_flag' in st.session_state and st.session_state.save_txt2img_flag:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"txt2img_generated_{timestamp}_{i+1}.png"
                pil_image.save(filename)
                st.success(f"Image saved as {filename}")


def print_info_NST():
       
    st.markdown("""
                ## What is NST?
                **NST** (*Neural Style Transfer*) is a Deep Learning
                technique to generate an image based on the content and
                style of two different images.  
                Let's have a look to an
                example (left column, top and bottom, are the *content*
                and *style*, respectively.):""")
    
    # Show exemplar images:
    root_content = os.path.join('data', 'content-images', 'picasso.jpg')
    root_style = os.path.join('data', 'style-images', '2reIEHS.jpg')
    
    # root_content = os.path.join(current_dir, "/Users/arjunkrisha/Desktop/neural_style/nst-project/picasso.jpg")
    # root_style = os.path.join(current_dir, "/Users/arjunkrisha/Desktop/neural_style/nst-project/2reIEHS.jpg")
    
    
    content_im = cv2.imread(root_content)
    style_im = cv2.imread(root_style)    
    im_cs, im_ss = prepare_imgs(content_im, style_im)    
    im_rs = cv2.imread(os.path.join('data', 'output-images', 'cbridge_svangogh2.jpg'))
    
    col1, col2 = st.columns([1,2.04])
    col1.header("Base")
    col1.image(im_cs, use_container_width=True)
    col1.image(im_ss, use_container_width=True)
    col2.header("Result")
    col2.image(im_rs, use_container_width=True, channels="BGR")
    
    
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

def print_info_pixel2turbo():
    
    st.markdown("""
                ## What is Pixel2Turbo?
                **Pixel2Turbo** is a fast image-to-image translation model based on Stable Diffusion Turbo. 
                It allows you to transform sketches, edges, or other image inputs into detailed images 
                guided by text prompts.
                
                The model supports two primary modes:
                - **Edge to Image**: Transforms edge drawings into photorealistic images
                - **Sketch to Image (Stochastic)**: Converts sketches to images with randomized variations
                
                Unlike Neural Style Transfer, Pixel2Turbo uses a pre-trained diffusion model with 
                text guidance to generate images in a single forward pass, making it very fast.
                """)
                
    # Show exemplar images (placeholders - you would need to replace these with actual examples)
    st.markdown("""
                ### Example:
                Below is an example of sketch-to-image transformation. The left shows the input sketch,
                and the right shows the generated image.
                """)
                
    # You'd need example images for Pixel2Turbo - this is just placeholder text
    st.info("Upload an image and try Pixel2Turbo to see the results!")

def print_info_txt2img():
    """Print basic information about Text-to-Image generation within the app."""
    st.markdown("""
                ## What is Text-to-Image Generation?
                **Text-to-Image** generation is a deep learning technique that creates images from textual descriptions.
                The model interprets your text prompt and generates visual content that matches your description.
                
                Our application supports two primary AI models:
                - **Stable Diffusion**: An open-source model known for its creative flexibility and detail
                - **DALL-E**: OpenAI's powerful text-to-image model with strong capabilities for realistic imagery
                
                Unlike Neural Style Transfer or Pixel2Turbo which require input images, text-to-image generation
                creates completely new images based solely on your text description.
                """)
                
    # Example prompts and images
    st.markdown("""
                ### Example Prompts:
                - "A serene mountain lake at sunset with reflections of pine trees"
                - "A futuristic cityscape with flying cars and neon lights"
                - "A photorealistic portrait of a fantasy creature with detailed features"
                - "An impressionist painting of a flower garden in spring"
                """)
                
    st.info("Enter a detailed prompt to generate an image!")
    

def neural_style_transfer_wrapper(content_img, style_img, model_type, device, **kwargs):
   
    if model_type == "VGG-19":
        return neural_style_transfer(kwargs['cfg'], device)
    elif model_type == "VGG-16":
        kwargs['cfg']['model'] = 'vgg16'
        return neural_style_transfer(kwargs['cfg'], device)
    elif model_type == "CNN":
        return neural_style_transfer_cnn(content_img, style_img, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def neural_style_transfer_cnn(content_tensor, style_tensor, device):
   
    try:
        # Initialize model
        model = StyleTransferCNN().to(device)
        model.eval()  # Set to evaluation mode
        
        # Move tensors to device
        content_tensor = content_tensor.to(device)
        style_tensor = style_tensor.to(device)
        
        # Generate stylized image
        with torch.no_grad():
            generated = model(content_tensor, style_tensor)
        
        # Convert output tensor to numpy array
        output = generated.squeeze(0).cpu()
        
        # Denormalize the output
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        output = output * std + mean
        
        # Convert to numpy array and ensure proper range
        output = output.numpy()
        output = np.transpose(output, (1, 2, 0))
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        
        return output
        
    except Exception as e:
        st.error(f"Error in CNN style transfer: {str(e)}")
        return None

def get_size_options(model_option):
    
    if model_option in ['stable-diffusion-xl-1024-v0-9', 'stable-diffusion-xl-1024-v1-0']:
        return [
            "1024x1024",  
            "1152x896",
            "1152x896",
            "1344x768",
            "1536x640",
            "640x1536",
            "768x1344",
            "1152x896",
            "896x1152"
        ]
    else:
        return ["1152x896", "1152x896", "1024x1024","1152x896"]

if __name__ == "__main__":
    
    
    st.title('Optmized NST')
    
    st.sidebar.title('Configuration')
    with st.sidebar:
        with st.expander("⚙️ Settings", expanded=True):
            options = ['About NST', 'Try NST', 'About Pix2img', 'Try Pixel2img', 
                      'About Text-to-Image', 'Generate from prompt']
            app_mode = st.selectbox('Mode:', options)
            st.info(f"Selected: {app_mode}")
   
    
    if app_mode == "Try NST":
        # Model selection in sidebar
        st.sidebar.title('NST Parameters')
        model_type = st.sidebar.selectbox(
            "Select Model Architecture",
            ["VGG-19", "VGG-16", "CNN"],
            help="Choose the neural network for style transfer"
        )
        
        # Dynamic parameter adjustment based on model
        if model_type in ["VGG-19", "VGG-16"]:
            st.sidebar.subheader("Weights",
                             help="Higher values preserve content structure better")
            step=1e-1
            cweight = st.sidebar.number_input("Content", value=1e-3, step=step, format="%.5f")
            sweight = st.sidebar.number_input("Style", value=1e-1, step=step, format="%.5f")
            vweight = st.sidebar.number_input("Variation", value=0.0, step=step, format="%.5f")
           
            st.sidebar.subheader('Number of iterations')
            niter = st.sidebar.number_input('Iterations', min_value=1, max_value=1000, value=20, step=1)
           
            st.sidebar.subheader('Save or not the stylized image')
            save_flag = st.sidebar.checkbox('Save result', key='save_vgg')
            
            # # Configuration for VGG models
            # cfg = {
            #     'output_img_path': os.path.join(os.path.dirname(__file__), "app_stylized_image.jpg"),
            #     'style_img': None,  # Will be set when images are uploaded
            #     'content_img': None,  # Will be set when images are uploaded
            #     'content_weight': cweight,
            #     'style_weight': sweight,
            #     'tv_weight': vweight,
            #     'optimizer': 'lbfgs',
            #     'model': model_type.lower(),  # 'vgg19' or 'vgg16'
            #     'init_method': 'random',
            #     'running_app': True,
            #     'save_flag': save_flag,
            #     'niter': niter
            # }
        
        elif model_type == "CNN":
            st.sidebar.subheader("Model Parameters")
             
            cweight = st.sidebar.slider("Content Weight", 0.0, 1e1, 1e5)  # Add this
            sweight = st.sidebar.slider("Style Weight", 0.0, 1e1, 1e5)     # Add this
            vweight = st.sidebar.slider("Variation Weight", 0.0, 1e1, 1e4)  # Add this if needed
            save_flag = st.sidebar.checkbox('Save result')

            st.sidebar.subheader('Number of iterations')
            niter = st.sidebar.number_input('Iterations', min_value=1, max_value=1000, value=20, step=1)
           
            st.sidebar.subheader('Save or not the stylized image')
            save_flag = st.sidebar.checkbox('Save result', key = 'save_cnn')
          
            
            
        # st.markdown("### Upload the pair of images to use")        
        # col1, col2 = st.columns(2)
        # im_types = ["png", "jpg", "jpeg"]
        
        # # Create file uploaders in a two column layout
        # with col1:
        #     file_c = st.file_uploader("Choose CONTENT Image", 
        #                             type=im_types,
        #                             key="nst_content_uploader")
        #     imc_ph = st.empty()            
        # with col2: 
        #     file_s = st.file_uploader("Choose STYLE Image", 
        #                             type=im_types,
        #                             key="nst_style_uploader")
        #     ims_ph = st.empty()
        
        # # If both images have been uploaded then preprocess and show them
        # if all([file_s, file_c]):
        #     # Preprocess
        #     im_c = np.array(Image.open(file_c))
        #     im_s = np.array(Image.open(file_s))
        #     im_c, im_s = prepare_imgs(im_c, im_s, RGB=True)
            
        #     # Show images
        #     imc_ph.image(im_c, use_column_width=True)
        #     ims_ph.image(im_s, use_column_width=True)

            # # Update VGG configuration with images if using VGG
            # if model_type in ["VGG-19", "VGG-16"]:
            #     cfg['content_img'] = im_c
            #     cfg['style_img'] = im_s
            # elif model_type == "CNN":
            #     # Prepare tensors for CNN model
            #     transform = transforms.Compose([
            #         transforms.ToTensor(),
            #         transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            #                           std=[0.229, 0.224, 0.225])
            #     ])
            #     content_tensor = transform(Image.fromarray(im_c)).unsqueeze(0)
            #     style_tensor = transform(Image.fromarray(im_s)).unsqueeze(0)
            
            # st.markdown("### When ready, START the image generation!")
            
            # # Button for starting the stylized image
            # start_flag = st.button("START", help="Start the style transfer process")
            # bt_ph = st.empty()
        
            # if start_flag:
            #     if not all([file_s, file_c]):
            #         bt_ph.markdown("You need to **upload the images** first! :)")
            #     else:
            #         bt_ph.markdown(f"Processing using {model_type}...")
                    
            #         # Create progress bar
            #         progress = st.progress(0.)
            #         res_im_ph = st.empty()
                    
            #         try:
            #             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        
            #             if model_type in ["VGG-19", "VGG-16"]:
            #                 # Add progress bar and result placeholder to config
            #                 cfg['res_im_ph'] = res_im_ph
            #                 cfg['st_bar'] = progress
            #                 result_im = neural_style_transfer(cfg, device)
            #             else:  # CNN model
            #                 result_im = neural_style_transfer_cnn(content_tensor, style_tensor, device)
                        
            #             if result_im is not None:
            #                 res_im_ph.image(result_im)
            #                 bt_ph.markdown("Style transfer complete!")
                            
            #                 # Save if requested
            #                 if save_flag:
            #                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #                     filename = f"nst_{model_type.lower()}_{timestamp}.jpg"
            #                     if isinstance(result_im, np.ndarray):
            #                         Image.fromarray(result_im).save(filename)
            #                     else:
            #                         result_im.save(filename)
            #                     st.success(f"Image saved as {filename}")
                    
            #         except Exception as e:
            #             st.error(f"An error occurred: {str(e)}")
            #             st.exception(e)
            
    elif app_mode in ['About Pixel2Turbo', 'Try Pixel2Turbo']:
        st.sidebar.title('Pixel2Turbo Parameters')
        st.sidebar.subheader('Model Type')
        model_type = st.sidebar.selectbox(
            'Select model type:',
            ['Edge to Image', 'Sketch to Image (Stochastic)']
        )
        
        # Text prompt for guiding the generation
        st.sidebar.subheader('Text Prompt')
        prompt = st.sidebar.text_input("Enter your text prompt:", "A beautiful landscape painting")
        
        # Randomness slider (only for the stochastic model)
        if model_type == 'Sketch to Image (Stochastic)':
            st.sidebar.subheader('Randomness')
            randomness = st.sidebar.slider(
                'Adjust randomness level:',
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Higher values increase variation in generated images"
            )
        else:
            randomness = 0.0
        
        # Save option
        st.sidebar.subheader('Save generated image')
        save_p2t_flag = st.sidebar.checkbox('Save result')

     
    # Text-to-Image parameters
    elif app_mode in ['Generate from prompt']:
        st.sidebar.title('Text-to-Image Parameters')
        
        # Initialize generator
        generator = initialize_text2img_generator()
        
        # # Model selection
        # st.sidebar.subheader('Model')
        # txt2img_model = st.sidebar.selectbox(
        #     'Select AI Model:',
        #     [
        #         'stable-diffusion-v1-5',
        #         'stable-diffusion-xl-1024-v1-0',
        #         'stable-diffusion-xl-1024-v0-9'
        #     ],
        #     key='txt2img_model'
        # )
        
        # Get appropriate size options based on model
        size_options = get_size_options(st.session_state.txt2img_model)
            
        # Size selection with default that matches model requirements
        # txt2img_size = st.sidebar.selectbox(
        #     'Select image size:',
        #     size_options,
        #     index=0,
        #     key='txt2img_size_select'
        # )
        # st.sidebar.subheader('Generation Options')
        # num_images = st.sidebar.slider(
        #     'Number of images:',
        #     min_value=1,
        #     max_value=4,
        #     value=1,
        #     key='txt2img_num_images'
        # )
        
        # # Advanced options in expander
        # with st.sidebar.expander("Advanced Settings", expanded=False):
        #     guidance_scale = st.slider(
        #         "Guidance Scale:",
        #         min_value=1.0,
        #         max_value=20.0,
        #         value=7.5,
        #         step=0.5,
        #         help="Higher values increase adherence to the prompt. Lower values allow more creativity.",
        #         key='txt2img_guidance'
        #     )
            
        #     steps = st.slider(
        #         "Generation Steps:",
        #         min_value=10,
        #         max_value=150,
        #         value=30,
        #         step=1,
        #         help="More steps generally result in higher quality but take longer.",
        #         key='txt2img_steps'
        #     )
            
        #     use_random_seed = st.checkbox("Use Random Seed", True, key='txt2img_use_random')
        #     if not use_random_seed:
        #         seed = st.number_input("Seed:", 0, 9999999, 42, key='txt2img_seed')
        #     else:
        #         seed = None
        
        # Save option
        # st.sidebar.subheader('Save Options')
        # st.session_state.save_txt2img_flag = st.sidebar.checkbox('Save generated images', key='txt2img_save')

    # Show the page of the selected page:
    if app_mode == options[0]:  # About NST
        print_info_NST()
        
    elif app_mode == options[1]:  # Run NST
        st.markdown("### Upload the pair of images to use")        
        col1, col2 = st.columns(2)
        im_types = ["png", "jpg", "jpeg"]
        
        # Create file uploaders in a two column layout, as well as
        # placeholder to later show the images uploaded:
        with col1:
            file_c = st.file_uploader("Choose CONTENT Image", 
                                     type=im_types,
                                     key="nst_content_uploader_1")
            imc_ph = st.empty()            
        with col2: 
            file_s = st.file_uploader("Choose STYLE Image", 
                                     type=im_types,
                                     key="nst_style_uploader_1")
            ims_ph = st.empty()
        
        # if both images have been uploaded then preprocess and show them:
        if all([file_s, file_c]):
            # preprocess:
            im_c = np.array(Image.open(file_c))
            im_s = np.array(Image.open(file_s))
            im_c, im_s = prepare_imgs(im_c, im_s, RGB=True)
            
            # Show images:
            imc_ph.image(im_c, use_container_width=True)
            ims_ph.image(im_s, use_container_width=True) 
        
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
            
    elif app_mode == options[2]:  # About Pixel2Turbo
        print_info_pixel2turbo()
        
    elif app_mode == options[3]:  # Run Pixel2Turbo
        st.markdown("### Generate Images with Pixel2Turbo")
        
        # Add model type selection in sidebar
        st.sidebar.title('Pixel2Turbo Parameters')
        model_type = st.sidebar.selectbox(
            'Select model type:',
            ['Edge to Image', 'Sketch to Image (Stochastic)'],
            key='pixel2turbo_model'
        )
        
        # Text prompt input
        prompt = st.text_input(
            "Enter your text prompt:",
            "A beautiful landscape painting",
            help="Describe the image you want to generate",
            key='pixel2turbo_prompt'
        )
        
        # Randomness slider (only for stochastic model)
        randomness = 0.0
        if model_type == 'Sketch to Image (Stochastic)':
            randomness = st.sidebar.slider(
                'Adjust randomness level:',
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Higher values increase variation in generated images",
                key='pixel2turbo_randomness'
            )
        
        # Save option
        save_p2t_flag = st.sidebar.checkbox('Save result', key='pixel2turbo_save')
        
        # File uploader for the input image
        im_types = ["png", "jpg", "jpeg"]
        file_input = st.file_uploader(
            "Choose Input Image", 
            type=im_types,
            key="pixel2turbo_uploader"
        )
        
        # Display the input image if uploaded
        if file_input:
            input_image = np.array(Image.open(file_input))
            st.image(input_image, caption="Input Image", use_container_width=True)
            
            # Resize image if needed (SD models typically work with 512x512)
            h, w = input_image.shape[:2]
            ratio = min(512/h, 512/w)
            if ratio < 1:
                new_size = (int(w*ratio), int(h*ratio))
                input_image = cv2.resize(input_image, new_size, interpolation=cv2.INTER_AREA)
                st.info(f"Image resized to {new_size[0]}x{new_size[1]} for processing")
            
            # Button to start generation
            start_p2t_flag = st.button(
                "Generate Image", 
                help="Start the Pixel2Turbo generation process",
                key='pixel2turbo_generate'
            )
            bt_p2t_ph = st.empty()  # Message placeholder
            
            if start_p2t_flag:
                bt_p2t_ph.markdown("Generating image with Pixel2Turbo...")
                
                try:
                    # Load the model
                    model = load_pixel2turbo_model(model_type)
                    
                    # Process the image with the prompt
                    output_image = process_pixel2turbo(model, input_image, prompt, randomness)
                    
                    # Show the result
                    st.image(output_image, caption="Generated Image", use_container_width=True)
                    
                    # Save the result if requested
                    if save_p2t_flag:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"pixel2turbo_output_{timestamp}.jpg"
                        Image.fromarray(output_image).save(filename)
                        st.success(f"Image saved as {filename}")
                    
                    bt_p2t_ph.markdown("Image generation complete!")
                    
                except Exception as e:
                    bt_p2t_ph.error(f"Error during image generation: {str(e)}")
                    st.exception(e)
        else:
            st.info("Please upload an image to begin")
    elif app_mode == options[4]:  # About Text-to-Image
        print_info_txt2img()
        
    elif app_mode == options[5]:  # Generate from prompt
        st.markdown("### Generate Images from Text Prompts")
        
        # Initialize the generator
        if 'txt2img_generator' not in st.session_state:
            st.session_state.txt2img_generator = initialize_text2img_generator()
        
        generator = st.session_state.txt2img_generator
        
        # Sidebar controls
        st.sidebar.title('Text-to-Image Parameters')
        
        # Model selection
        selected_model = st.sidebar.selectbox(
            "AI Model",
            ["Stable Diffusion", "DALL-E"],
            index=0,
            key='txt2img_model'
        )
        
        # Dynamic API key handling
        if selected_model == "Stable Diffusion":
            if not generator.api_keys.get('stability'):
                st.sidebar.text_input(
                    "Stability API Key",
                    type="password",
                    key="stability_key_input",
                    help="Get key from https://platform.stability.ai/"
                )
                if st.session_state.stability_key_input:
                    st.session_state.stability_api_key = st.session_state.stability_key_input
                    st.session_state.txt2img_generator = initialize_text2img_generator()
                    st.rerun()
        else:  # DALL-E
            if not generator.api_keys.get('openai'):
                st.sidebar.text_input(
                    "OpenAI API Key",
                    type="password",
                    key="openai_key_input",
                    help="Get key from https://platform.openai.com/"
                )
                if st.session_state.openai_key_input:
                    st.session_state.openai_api_key = st.session_state.openai_key_input
                    st.session_state.txt2img_generator = initialize_text2img_generator()
                    st.rerun()

        # Size selection based on the updated model capabilities
        if selected_model == "DALL-E":
            # DALL-E 3 sizes
            size_options = ["1024x1024", "1024x1792", "1792x1024"]
        else:
            size_options = get_size_options(selected_model)
        
        size = st.sidebar.selectbox("Image Size", size_options, key='txt2img_size')
        
        # Generation parameters
        num_images = st.sidebar.slider("Number of Images", 1, 4, 1, key='txt2img_num')
        
        if selected_model == "Stable Diffusion":
            guidance = st.sidebar.slider("Guidance Scale", 1.0, 20.0, 7.5, key='txt2img_guidance')
            steps = st.sidebar.slider("Steps", 10, 150, 50, key='txt2img_steps')
        else:
            guidance = None
            steps = None
        
        seed = st.sidebar.number_input("Seed", value=42, key='txt2img_seed') if st.sidebar.checkbox("Custom Seed", False) else None
        
        # Main content
        prompt = st.text_area("Prompt", height=100, placeholder="Describe the image you want to generate...")
        negative_prompt = st.text_area("Negative Prompt", height=68, placeholder="What to exclude from the image...")
        
        if st.button("Generate"):
            if not prompt:
                st.warning("Please enter a prompt")
            else:
                try:
                    with st.spinner(f"Generating images with {selected_model}..."):
                        images = generator.generate_images(
                            prompt=prompt,
                            negative_prompt=negative_prompt if negative_prompt else None,
                            model=selected_model,
                            size=size,
                            num_images=num_images,
                            guidance_scale=guidance,
                            steps=steps,
                            seed=seed
                        )
                        display_generated_images(images, st)
                    
                    # Fix for saving images
                    if st.session_state.get('save_txt2img_flag', False):
                        save_generated_images(images, selected_model)
                        
                except Exception as e:
                    st.error(f"Generation failed: {str(e)}")


    # if 'save_txt2img_flag' in st.session_state and st.session_state.save_txt2img_flag:
    #                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #                 filename = f"txt2img_generated_{timestamp}_{i+1}.png"
    #                 pil_image.save(filename)
    #                 st.success(f"Image saved as {filename}")

