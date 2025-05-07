import os
import time
import numpy as np
import torch
import cv2
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Import your existing NST functions
from main import neural_style_transfer
from using_cnn import StyleTransferCNN

class ModelComparer:
    """Compare different neural style transfer models and evaluate their performance."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = ["VGG-19", "VGG-16", "CNN"]
        self.metrics = ["Style Transfer Quality", "Content Preservation", "Processing Time", "Overall Score"]
    
    def prepare_tensors(self, content_img, style_img):
        """Prepare image tensors for the CNN model."""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert numpy arrays to PIL images
        if isinstance(content_img, np.ndarray):
            content_pil = Image.fromarray(content_img)
            style_pil = Image.fromarray(style_img)
        else:
            content_pil = content_img
            style_pil = style_img
            
        # Apply transforms
        content_tensor = transform(content_pil).unsqueeze(0).to(self.device)
        style_tensor = transform(style_pil).unsqueeze(0).to(self.device)
        
        return content_tensor, style_tensor
    
    def compare_models(self, content_img, style_img, iterations=20, progress_callback=None):
        """
        Compare different models using the same content and style image.
        
        Args:
            content_img: Content image as numpy array (RGB)
            style_img: Style image as numpy array (RGB)
            iterations: Number of iterations for optimization
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary with results and metrics
        """
        results = {}
        metrics = {}
        
        # Set up common parameters
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        out_img_path = os.path.join(parent_dir, "comparison_stylized_image.jpg")
        
        # Empty progress callback if none provided
        if progress_callback is None:
            progress_callback = lambda x, y: None
            
        # Process with each model
        for i, model_type in enumerate(self.models):
            progress_callback(i / len(self.models), f"Processing with {model_type}...")
            
            start_time = time.time()
            
            if model_type in ["VGG-19", "VGG-16"]:
                # Configure parameters for VGG models
                cfg = {
                    'output_img_path': out_img_path,
                    'style_img': style_img,
                    'content_img': content_img,
                    'content_weight': 1e-3,
                    'style_weight': 1e-1,
                    'tv_weight': 0.0,
                    'optimizer': 'lbfgs',
                    'model': model_type.lower().replace('-', ''),
                    'init_metod': 'random',
                    'running_app': False,
                    'res_im_ph': None,
                    'save_flag': False,
                    'st_bar': None,
                    'niter': iterations
                }
                
                # Run neural style transfer
                result_img = neural_style_transfer(cfg, self.device)
                
            elif model_type == "CNN":
                # Process using CNN model
                content_tensor, style_tensor = self.prepare_tensors(content_img, style_img)
                
                # Initialize and run CNN model
                model = StyleTransferCNN().to(self.device)
                model.eval()
                
                with torch.no_grad():
                    output = model(content_tensor, style_tensor)
                
                # Convert output tensor to numpy array
                output = output.squeeze(0).cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                output = output * std + mean
                output = output.numpy()
                output = np.transpose(output, (1, 2, 0))
                result_img = np.clip(output * 255, 0, 255).astype(np.uint8)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Calculate metrics
            style_score = self._calculate_style_score(result_img, style_img)
            content_score = self._calculate_content_score(result_img, content_img)
            
            # Store results and metrics
            results[model_type] = result_img
            metrics[model_type] = {
                "Style Transfer Quality": style_score,
                "Content Preservation": content_score,
                "Processing Time": processing_time,
                "Overall Score": (style_score + content_score) / 2  # Simple average
            }
            
            progress_callback((i + 1) / len(self.models), f"Finished processing with {model_type}")
        
        return {
            "results": results,
            "metrics": metrics
        }
    
    def _calculate_style_score(self, result_img, style_img):
        """
        Calculate a score for how well the style was transferred.
        Higher is better (0-10 scale).
        
        This is a simplified approximation using SSIM in color space.
        A more accurate approach would use network feature comparisons.
        """
        # Resize style image to match result image if needed
        if style_img.shape != result_img.shape:
            style_img = cv2.resize(style_img, (result_img.shape[1], result_img.shape[0]), 
                                   interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale for texture comparison
        result_gray = cv2.cvtColor(result_img, cv2.COLOR_RGB2GRAY)
        style_gray = cv2.cvtColor(style_img, cv2.COLOR_RGB2GRAY)
        
        # Calculate texture similarity (using SSIM)
        texture_similarity, _ = ssim(result_gray, style_gray, full=True)
        
        # Heuristic scoring function - converts to 0-10 scale
        # Lower SSIM is actually better for style transfer (we want texture similarity but not exact matching)
        style_score = 5.0 + 5.0 * (1.0 - texture_similarity)
        
        return min(max(style_score, 0), 10)  # Clamp to 0-10 range
    
    def _calculate_content_score(self, result_img, content_img):
        """
        Calculate a score for how well the content was preserved.
        Higher is better (0-10 scale).
        """
        # Resize content image to match result image if needed
        if content_img.shape != result_img.shape:
            content_img = cv2.resize(content_img, (result_img.shape[1], result_img.shape[0]), 
                                    interpolation=cv2.INTER_CUBIC)
        
        # Calculate SSIM for structure preservation
        ssim_value, _ = ssim(cv2.cvtColor(result_img, cv2.COLOR_RGB2GRAY), 
                            cv2.cvtColor(content_img, cv2.COLOR_RGB2GRAY), 
                            full=True)
        
        # Calculate PSNR for overall difference
        psnr_value = psnr(content_img, result_img)
        
        # Normalize PSNR to 0-5 scale (typical PSNR values range from 20-40)
        psnr_score = min(max(psnr_value - 15, 0) / 5, 5)
        
        # Combine metrics (SSIM is already 0-1, scale to 0-5)
        content_score = ssim_value * 5 + psnr_score
        
        return min(content_score, 10)  # Clamp to 0-10 range

def display_comparison_view(content_img, style_img, comparison_results):
    """Display comparison results in Streamlit."""
    st.subheader("Style Transfer Model Comparison")
    
    # Display original images
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Content Image**")
        st.image(content_img, use_container_width=True)
    with col2:
        st.markdown("**Style Image**")
        st.image(style_img, use_container_width=True)
    
    # Display results
    st.markdown("### Generated Images")
    cols = st.columns(len(comparison_results["results"]))
    
    for i, (model_name, result_img) in enumerate(comparison_results["results"].items()):
        with cols[i]:
            st.markdown(f"**{model_name}**")
            st.image(result_img, use_container_width=True)
            
            # Display key metrics under each image
            st.markdown(f"**Style Score:** {comparison_results['metrics'][model_name]['Style Transfer Quality']:.2f}/10")
            st.markdown(f"**Content Score:** {comparison_results['metrics'][model_name]['Content Preservation']:.2f}/10")
            st.markdown(f"**Time:** {comparison_results['metrics'][model_name]['Processing Time']:.2f} sec")
    
    # Display radar chart of metrics
    st.markdown("### Performance Metrics Comparison")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df = {}
    
    # Extract metrics for each model
    for model_name, metrics in comparison_results["metrics"].items():
        metrics_df[model_name] = [
            metrics["Style Transfer Quality"],
            metrics["Content Preservation"],
            10 - min(metrics["Processing Time"] / 10, 10),  # Invert time (lower is better)
            metrics["Overall Score"]
        ]
    
    # Get metric names
    metric_names = list(comparison_results["metrics"][list(comparison_results["metrics"].keys())[0]].keys())
    
    # Create bar chart
    x = np.arange(len(metric_names))
    width = 0.2
    multiplier = 0
    
    for model_name, scores in metrics_df.items():
        offset = width * multiplier
        ax.bar(x + offset, scores, width, label=model_name)
        multiplier += 1
    
    # Add labels and legend
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 10)
    ax.set_ylabel('Score (0-10)')
    ax.set_title('Model Performance Comparison')
    
    # Display the chart
    st.pyplot(fig)
    
    # Display summary
    st.markdown("### Summary")
    best_model = max(comparison_results["metrics"].items(), 
                     key=lambda x: x[1]["Overall Score"])[0]
    
    st.info(f"**Best overall model:** {best_model}")
    
    fastest_model = min(comparison_results["metrics"].items(), 
                        key=lambda x: x[1]["Processing Time"])[0]
    st.info(f"**Fastest model:** {fastest_model} ({comparison_results['metrics'][fastest_model]['Processing Time']:.2f} seconds)")
    
    best_style = max(comparison_results["metrics"].items(), 
                     key=lambda x: x[1]["Style Transfer Quality"])[0]
    st.info(f"**Best style transfer:** {best_style}")
    
    best_content = max(comparison_results["metrics"].items(), 
                       key=lambda x: x[1]["Content Preservation"])[0]
    st.info(f"**Best content preservation:** {best_content}")
    
    # Add explanations
    st.markdown("""
    ### About the Metrics
    
    - **Style Transfer Quality**: Measures how well the style characteristics were transferred (0-10)
    - **Content Preservation**: Measures how well the content structure was preserved (0-10)  
    - **Processing Time**: Execution speed (converted to 0-10 scale, higher is faster)
    - **Overall Score**: Combined performance metric
    
    > Note: These scores are approximations based on image similarity metrics and may not perfectly 
    > capture perceptual quality differences.
    """)