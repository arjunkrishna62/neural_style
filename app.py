import os
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import sys
from dotenv import load_dotenv
from datetime import datetime
import io
from torchvision import transforms


from typing import Optional, Dict, List

from laaca_protection.protect_style import StyleProtector




from main import neural_style_transfer
from text_to_image import TextToImageGenerator

from using_cnn import StyleTransferCNN

load_dotenv()

# protector = StyleProtector()
# # protected_image = protector.protect_style_image()

# # Example usage (after training)
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Load and preprocess an image
# image = Image.open("data/style-images/2reIEHS.jpg")
# input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# # Protect the image
# protected_tensor = protector.protect_image(input_tensor)

# # Convert back to image (denormalize and convert to PIL)
# denormalize = transforms.Normalize(
#     mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
#     std=[1/0.229, 1/0.224, 1/0.225]
# )
# protected_tensor = denormalize(protected_tensor.squeeze())
# protected_image = transforms.ToPILImage()(protected_tensor.clamp(0, 1))
# protected_image.save("protected_image.jpg")

@st.cache_resource
def get_style_protector():
    return StyleProtector(device='cuda' if torch.cuda.is_available() else 'cpu')


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

def print_info_style_protection():
    """Print information about LAACA Style Protection"""
    st.markdown("""
                ## What is LAACA Style Protection?
                
                **LAACA** (Latent Adversarial Autoencoder for Copyright Assurance) is a technique designed to protect the style of an image from being copied by neural style transfer algorithms. It works by:
                
                1. Adding subtle perturbations to the style image
                2. These perturbations are invisible to humans but disrupt the style extraction process
                3. When a protected style image is used in style transfer, the resulting stylized image will show visible artifacts
                
                This technology helps artists and creators protect their unique artistic styles from unauthorized copying or mimicry using AI tools.
                """)
    
    st.markdown("""
                ### How to use Style Protection
                
                You can protect your style images in two ways:
                
                1. **In the "Try NST" mode**: Enable the Style Protection option in the sidebar when performing style transfer
                
                2. **In the "Style Protection Demo" mode**: Upload a style image and apply protection to see how it works
                
                The protection strength can be adjusted to balance between protection effectiveness and image quality.
                """)


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
            options = ['About NST', 'Try NST','About Style Protection','Style Protection Demo', 'About Text-to-Image', 'Generate from prompt', 'Style Protection Demo']
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
        
        # Add Style Protection controls to sidebar
        with st.sidebar.expander("Style Protection (LAACA)", expanded=False):
            use_protection = st.checkbox(
                "Enable Style Protection", 
                value=False,
                help="Apply LAACA protection to the style image"
            )
            protection_strength = st.slider(
                "Protection Strength",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Higher values provide stronger protection but may affect image quality"
            )
            st.info("Style protection prevents unwanted style copying by adding invisible perturbations to the style image.")
        
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
            
            # Apply style protection if enabled
            if use_protection:
                try:
                    st.info("Applying LAACA style protection...")
                    # Convert numpy array to PIL image for protection
                    style_pil = Image.fromarray(im_s)
                    # Apply protection using the imported function
                    protected_style = protect_style_image(style_pil, strength=protection_strength)
                    # Convert back to numpy array
                    im_s = np.array(protected_style)
                    st.success("Style protection applied successfully!")
                except Exception as e:
                    st.error(f"Error applying style protection: {e}")
            
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
    
    elif app_mode == "About Style Protection":
        print_info_style_protection()
        
    elif app_mode == "Style Protection Demo":
        st.markdown("## Style Protection Demo")
        
        st.markdown("""
        This demonstration shows how LAACA protection works to prevent style copying.
        Upload a style image and apply protection to see the differences.
        """)
        
        # Create file uploader for style image
        im_types = ["png", "jpg", "jpeg"]
        file_style = st.file_uploader("Upload Style Image", type=im_types, key="style_protection_uploader")
        
        # Control parameters
        col1, col2 = st.columns(2)
        with col1:
            protection_strength = st.slider(
                "Protection Strength",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Higher values provide stronger protection but may affect visual quality"
            )
        
        if file_style:
            # Load and display original image
            original_image = Image.open(file_style)
            
            # Create two columns for side-by-side comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Style Image")
                st.image(original_image, use_container_width=True)
            
            # Apply protection
            with st.spinner("Applying protection..."):
                try:
                    protected_image = protect_style_image(original_image, strength=protection_strength)
                    
                    with col2:
                        st.subheader("Protected Style Image")
                        st.image(protected_image, use_container_width=True)
                    
                    # Add explanation text
                    st.markdown("""
                    ### How the Protection Works
                    
                    The protected image contains imperceptible perturbations that will disrupt style transfer algorithms when 
                    someone attempts to copy your style. To humans, the images look nearly identical, but neural networks will
                    struggle to extract the style features correctly.
                    """)
                    
                    # Provide download option for protected image
                    buf = io.BytesIO()
                    protected_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="Download Protected Image",
                        data=byte_im,
                        file_name="protected_style_image.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"Error applying protection: {str(e)}")
        else:
            st.info("Please upload a style image to continue")
    
    elif app_mode == options[0]:  # About NST
        print_info_NST()
    
    elif app_mode == "About Text-to-Image":  # About Text-to-Image
        print_info_txt2img()
        
    elif app_mode == "Generate from prompt":  # Generate from prompt
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