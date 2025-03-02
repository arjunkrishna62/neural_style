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


from main import neural_style_transfer
from text_to_image import TextToImageGenerator
from src.pixel2turbo import Pix2Pix_Turbo

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

@st.cache_resource
def initialize_text2img_generator():
    """Initialize the Text-to-Image generator with API keys from environment variables"""
    api_keys = {
        'stability': os.getenv('STABILITY_API_KEY'),
        'openai': os.getenv('OPENAI_API_KEY')
    }
    return TextToImageGenerator(api_keys)


@st.cache_resource
def load_pixel2turbo_model(model_type):
    """Load the Pixel2Turbo model based on selection"""
    if model_type == "Edge to Image":
        model = Pix2Pix_Turbo(pretrained_name="edge_to_image")
    elif model_type == "Sketch to Image (Stochastic)":
        model = Pix2Pix_Turbo(pretrained_name="sketch_to_image_stochastic")
    else:
        model = Pix2Pix_Turbo()  # Default initialization
        
    model.set_eval()  # Set model to evaluation mode
    return model

def process_pixel2turbo(model, input_image, prompt, randomness=0.0):
    """Process an image with the Pixel2Turbo model"""
    # Convert PIL image to tensor
    input_tensor = torch.from_numpy(input_image).float().cuda() / 127.5 - 1.0
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
    
    # Process deterministically or with randomness
    deterministic = randomness <= 0.0
    
    if deterministic:
        output = model(input_tensor, prompt=prompt, deterministic=True)
    else:
        # Generate noise map for stochastic generation
        shape = input_tensor.shape
        noise = torch.randn((shape[0], 4, shape[2]//8, shape[3]//8), device="cuda")
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
            st.image(image, caption=f"Generated Image {i+1}", use_column_width=True)
            
            # Convert numpy array to bytes for download
            pil_image = Image.fromarray(image)
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
            if 'save_txt2img_flag' in locals() and save_txt2img_flag:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"txt2img_generated_{timestamp}_{i+1}.png"
                pil_image.save(filename)
                st.success(f"Image saved as {filename}")


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
    root_content = os.path.join('data', 'content-images', 'picasso.jpg')
    root_style = os.path.join('data', 'style-images', '2reIEHS.jpg')
    
    # root_content = os.path.join(current_dir, "/Users/arjunkrisha/Desktop/neural_style/nst-project/picasso.jpg")
    # root_style = os.path.join(current_dir, "/Users/arjunkrisha/Desktop/neural_style/nst-project/2reIEHS.jpg")
    
    
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

def print_info_pixel2turbo():
    """Print basic information about Pixel2Turbo within the app."""
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
    

if __name__ == "__main__":
    
    
    st.title('Optimized Neural Styel Transfer')
    
    st.sidebar.title('Configuration')
    with st.sidebar:
        with st.expander("⚙️ Settings", expanded=True):
            options = ['About NST', 'Try NST', 'About Pixel2Turbo', 'Run Pixel2Turbo', 
                      'About Text-to-Image', 'Generate from Text']
            app_mode = st.selectbox('Mode:', options)
            st.info(f"Selected: {app_mode}")
    
    # Text-to-Image parameters
    if app_mode in ['Generate from Text']:
        st.sidebar.title('Text-to-Image Parameters')
        
        # Initialize generator
        generator = initialize_text2img_generator()
        
        # Model selection
        st.sidebar.subheader('Model')
        model_option = st.sidebar.selectbox(
            'Select AI Model:',
            generator.get_available_models()
        )
        
        # Image size selection
        st.sidebar.subheader('Image Size')
        size_option = st.sidebar.selectbox(
            'Select image size:',
            generator.get_available_sizes(model_option)
        )
        
        # Number of images
        st.sidebar.subheader('Generation Options')
        num_images = st.sidebar.slider(
            'Number of images:',
            min_value=1,
            max_value=4,
            value=1
        )
        
        # Advanced options in expander
        with st.sidebar.expander("Advanced Settings", expanded=False):
            guidance_scale = st.slider(
                "Guidance Scale:",
                min_value=1.0,
                max_value=20.0,
                value=7.5,
                step=0.5,
                help="Higher values increase adherence to the prompt. Lower values allow more creativity."
            )
            
            steps = st.slider(
                "Generation Steps:",
                min_value=20,
                max_value=150,
                value=50,
                help="More steps generally result in higher quality but take longer."
            )
            
            use_random_seed = st.checkbox("Use Random Seed", True)
            if not use_random_seed:
                seed = st.number_input("Seed:", 0, 9999999, 42)
            else:
                seed = None
        
        # Save option
        st.sidebar.subheader('Save Options')
        save_txt2img_flag = st.sidebar.checkbox('Save generated images')
    
    elif app_mode in ['Try NST']:
        st.sidebar.title('Parameters')
        
        st.sidebar.subheader("Weights",
                         help="Higher values preserve content structure better")
        step=1e-1
        cweight = st.sidebar.number_input("Content", value=1e-3, step=step, format="%.5f")
        sweight = st.sidebar.number_input("Style", value=1e-1, step=step, format="%.5f")
        vweight = st.sidebar.number_input("Variation", value=0.0, step=step, format="%.5f")
       
        st.sidebar.subheader('Number of iterations')
        niter = st.sidebar.number_input('Iterations', min_value=1, max_value=1000, value=20, step=1)
       
        st.sidebar.subheader('Save or not the stylized image')
        save_flag = st.sidebar.checkbox('Save result')
            
    elif app_mode in ['About Pixel2Turbo', 'Try Pixel2Turbo']:
        st.sidebar.title('Pixel2Turbo Parameters')
        st.sidebar.subheader('Model Type')
        model_type = st.sidebar.selectbox(
            'Select model type:',
            ['Edge to Image', 'Sketch to Image (Stochastic)']
        )
        
        # Text prompt for guiding the generation
        st.sidebar.subheader('Text Prompt')
        prompt = st.sidebar.text_area(
            'Enter a text prompt to guide the generation:',
            'A detailed, high-quality photorealistic image.'
        )
        
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
        st.markdown("")
        
        # File uploader for the input image
        im_types = ["png", "jpg", "jpeg"]
        file_input = st.file_uploader("Choose Input Image", type=im_types)
        
        # Display the input image if uploaded
        if file_input:
            input_image = np.array(Image.open(file_input))
            st.image(input_image, caption="Input Image", use_column_width=True)
            
            # Resize image if needed (SD models typically work with 512x512)
            h, w = input_image.shape[:2]
            ratio = min(512/h, 512/w)
            if ratio < 1:
                new_size = (int(w*ratio), int(h*ratio))
                input_image = cv2.resize(input_image, new_size, interpolation=cv2.INTER_AREA)
                st.info(f"Image resized to {new_size[0]}x{new_size[1]} for processing")
            
            # Button to start generation
            start_p2t_flag = st.button("Generate Image", help="Start the Pixel2Turbo generation process")
            bt_p2t_ph = st.empty()  # Message placeholder
            
            if start_p2t_flag:
                bt_p2t_ph.markdown("Generating image with Pixel2Turbo...")
                
                try:
                    # Load the selected model
                    with st.spinner(f"Loading {model_type} model..."):
                        model = load_pixel2turbo_model(model_type)
                    
                    # Process the image
                    with st.spinner("Processing image..."):
                        output_image = process_pixel2turbo(model, input_image, prompt, randomness)
                    
                    # Show the result
                    st.image(output_image, caption="Generated Image", use_column_width=True)
                    
                    # Save the result if requested
                    if save_p2t_flag:
                        parent_dir = os.path.dirname(__file__)
                        out_img_path = os.path.join(parent_dir, "pixel2turbo_output.jpg")
                        cv2.imwrite(out_img_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
                        st.success(f"Image saved to {out_img_path}")
                    
                    bt_p2t_ph.markdown("Image generation complete!")
                    
                except Exception as e:
                    bt_p2t_ph.error(f"Error during image generation: {str(e)}")
                    st.exception(e)
            
        else:
            st.info("Please upload an image to begin")
    elif app_mode == options[4]:  # About Text-to-Image
        print_info_txt2img()
        
    elif app_mode == options[5]:  # Generate from Text
        st.markdown("### Generate Images from Text Prompts")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Text prompt input
            st.subheader("Enter Your Prompt")
            
            prompt = st.text_area(
                "Describe the image you want to generate:",
                height=100,
                placeholder="A serene landscape with mountains reflecting in a crystal clear lake at sunset, dramatic sky with vibrant colors..."
            )
            
            # Negative prompt input
            negative_prompt = st.text_area(
                "Elements to avoid (negative prompt):",
                height=50,
                placeholder="blurry, low quality, distorted, watermark, text, deformed..."
            )
            
            # Quick example prompts
            st.subheader("Example Prompts")
            example_prompts = [
                "A futuristic city with flying cars and neon lights",
                "A photorealistic portrait of a fantasy creature",
                "A cozy cabin in a snowy forest with northern lights"
            ]
            
            # Create buttons for examples
            cols = st.columns(len(example_prompts))
            for i, (col, ex_prompt) in enumerate(zip(cols, example_prompts)):
                with col:
                    if st.button(f"Example {i+1}", key=f"txt2img_example_{i}"):
                        prompt = ex_prompt
                        st.session_state.txt2img_prompt = ex_prompt
                        st.experimental_rerun()
            
            # Generate button
            generate_pressed = st.button("Generate Image", type="primary", use_container_width=True)
        
        with col2:
            st.subheader("Generated Images")
            image_placeholder = st.empty()
            
            # Process generation when button is pressed
            if generate_pressed and prompt:
                with st.spinner("Generating your images..."):
                    try:
                        # Initialize the generator if not already done
                        if 'txt2img_generator' not in st.session_state:
                            st.session_state.txt2img_generator = initialize_text2img_generator()
                        
                        generator = st.session_state.txt2img_generator
                        
                        # Generate images
                        images = generator.generate_images(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            model=model_option,
                            size=size_option,
                            num_images=num_images,
                            guidance_scale=guidance_scale,
                            steps=steps,
                            seed=seed
                        )
                        
                        # Store images in session state
                        st.session_state.generated_images = images
                        
                        # Display images
                        if images:
                            display_generated_images(images, image_placeholder)
                        else:
                            st.error("Failed to generate images. Please try again.")
                    
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            
            # Display previously generated images if they exist
            elif 'generated_images' in st.session_state:
                display_generated_images(st.session_state.generated_images, image_placeholder)

