import os
import io
import requests
from PIL import Image
from typing import List, Optional, Tuple, Dict, Any
import base64

class TextToImageGenerator:
    """
    A modular text-to-image generation class that can be integrated into existing Streamlit applications.
    Supports multiple AI models for image generation based on text prompts.
    """
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        
            api_keys: 
                     {'stability': 'your-stability-api-key',
                      'openai': 'your-openai-api-key'}
        """
        self.api_keys = api_keys or {}
        
        
        if 'stability' not in self.api_keys and os.getenv('STABILITY_API_KEY'):
            self.api_keys['stability'] = os.getenv('STABILITY_API_KEY')
            
        if 'openai' not in self.api_keys and os.getenv('OPENAI_API_KEY'):
            self.api_keys['openai'] = os.getenv('OPENAI_API_KEY')
    
    def generate_images(
        self,
        prompt: str,
        model: str = "Stable Diffusion",
        size: str = "512x512",
        num_images: int = 1,
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        steps: int = 50,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generate images based on text prompt using the specified model.
        
        Args:
            prompt: Text description of the desired image
            model: Model to use for generation ('Stable Diffusion', 'DALL-E', 'Midjourney API')
            size: Image size as 'widthxheight' string
            num_images: Number of images to generate
            negative_prompt: Elements to avoid in the generated image
            guidance_scale: Higher values increase prompt adherence
            steps: Number of diffusion steps (more = higher quality but slower)
            seed: Random seed for reproducibility
            
        Returns:
            List of PIL Image objects
        """
        if model == "Stable Diffusion":
            return self._generate_with_stability_ai(
                prompt, negative_prompt, size, num_images, guidance_scale, steps, seed
            )
        elif model == "DALL-E":
            return self._generate_with_dalle(prompt, size, num_images, negative_prompt)
        elif model == "Midjourney API":
            return self._generate_with_midjourney(prompt, size, num_images)
        else:
            # Default to placeholder images for unsupported models
            return [self._create_placeholder_image(f"{model}: {prompt[:30]}...", size) 
                   for _ in range(num_images)]
    
    def _generate_with_stability_ai(
        self, 
        prompt: str, 
        negative_prompt: str, 
        size: str, 
        num_images: int,
        guidance_scale: float,
        steps: int,
        seed: Optional[int]
    ) -> List[Image.Image]:
        """
        Generate images using Stability AI's API (Stable Diffusion).
        """
        api_key = self.api_keys.get('stability')
        if not api_key:
            print("Warning: Stability AI API key not found. Using placeholder images.")
            return [self._create_placeholder_image(f"Stable Diffusion: {prompt[:30]}...", size) 
                   for _ in range(num_images)]
        
        try:
            # Parse size string to get width and height
            width, height = map(int, size.split('x'))
            
            url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "text_prompts": [
                    {"text": prompt, "weight": 1.0}
                ],
                "cfg_scale": guidance_scale,
                "height": height,
                "width": width,
                "samples": num_images,
                "steps": steps,
            }
            
            # Add negative prompt if provided
            if negative_prompt:
                payload["text_prompts"].append({"text": negative_prompt, "weight": -1.0})
            
            # Add seed if provided
            if seed is not None:
                payload["seed"] = seed
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                images = []
                data = response.json()
                for image_data in data["artifacts"]:
                    image_bytes = base64.b64decode(image_data["base64"])
                    image = Image.open(io.BytesIO(image_bytes))
                    images.append(image)
                return images
            else:
                print(f"Error from Stability API: {response.status_code}, {response.text}")
                return [self._create_placeholder_image(f"API Error: {response.status_code}", size) 
                       for _ in range(num_images)]
                
        except Exception as e:
            print(f"Exception in Stability AI generation: {str(e)}")
            return [self._create_placeholder_image(f"Error: {str(e)[:30]}...", size) 
                   for _ in range(num_images)]
    
    def _generate_with_dalle(
        self, 
        prompt: str, 
        size: str, 
        num_images: int,
        negative_prompt: str = ""
    ) -> List[Image.Image]:
        """
        Generate images using OpenAI's DALL-E API.
        """
        api_key = self.api_keys.get('openai')
        if not api_key:
            print("Warning: OpenAI API key not found. Using placeholder images.")
            return [self._create_placeholder_image(f"DALL-E: {prompt[:30]}...", size) 
                   for _ in range(num_images)]
        
        try:
            # For actual implementation, you would use OpenAI's API
            # This is an example of how it would be implemented:
            """
            import openai
            openai.api_key = api_key
            
            if negative_prompt:
                # For DALL-E, you might integrate negative prompt into the main prompt
                full_prompt = f"{prompt}. Please avoid: {negative_prompt}"
            else:
                full_prompt = prompt
            
            response = openai.Image.create(
                prompt=full_prompt,
                n=num_images,
                size=size  # OpenAI expects sizes like "1024x1024"
            )
            
            images = []
            for data in response['data']:
                image_url = data['url']
                image_response = requests.get(image_url)
                image = Image.open(io.BytesIO(image_response.content))
                images.append(image)
            
            return images
            """
            
            # For now, return placeholder images
            return [self._create_placeholder_image(f"DALL-E: {prompt[:30]}...", size) 
                   for _ in range(num_images)]
                   
        except Exception as e:
            print(f"Exception in DALL-E generation: {str(e)}")
            return [self._create_placeholder_image(f"Error: {str(e)[:30]}...", size) 
                   for _ in range(num_images)]
    
    def _generate_with_midjourney(
        self, 
        prompt: str, 
        size: str, 
        num_images: int
    ) -> List[Image.Image]:
        """
        Generate images using Midjourney (Note: this is a placeholder as Midjourney doesn't have an official API).
        """
        # For demo purposes only - Midjourney doesn't have an official API
        return [self._create_placeholder_image(f"Midjourney: {prompt[:30]}...", size) 
               for _ in range(num_images)]
    
    def _create_placeholder_image(self, text: str, size: str = "512x512") -> Image.Image:
        """
        Create a placeholder image with text for demonstration purposes.
        """
        try:
            width, height = map(int, size.split('x'))
        except ValueError:
            width, height = 512, 512
            
        # Create a light gray image
        image = Image.new("RGB", (width, height), color=(240, 240, 240))
        
        # Return the placeholder image
        return image
    
    def get_available_models(self) -> List[str]:
        """
        Returns a list of available text-to-image models.
        """
        return ["Stable Diffusion", "DALL-E", "Midjourney API"]
    
    def get_available_sizes(self, model: str = None) -> List[str]:
        """
        Returns available image sizes, optionally filtered by model.
        """
        if model == "DALL-E":
            return ["256x256", "512x512", "1024x1024"]
        else:
            return ["512x512", "768x768", "1024x1024", "1024x1536"]