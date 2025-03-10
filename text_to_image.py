import os
import io
import requests
from PIL import Image
from typing import List, Optional, Tuple, Dict, Any
import base64

class TextToImageGenerator:
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self.api_keys = api_keys or {}
        
        # Load from environment if not provided in constructor
        if 'stability' not in self.api_keys and os.getenv('STABILITY_API_KEY'):
            self.api_keys['stability'] = os.getenv('STABILITY_API_KEY')
            
        if 'openai' not in self.api_keys and os.getenv('OPENAI_API_KEY'):
            self.api_keys['openai'] = os.getenv('OPENAI_API_KEY')
        
        # Debug output for initialization
        print(f"TextToImageGenerator initialized with API keys for: {list(self.api_keys.keys())}")
    
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
        """
        print(f"Generating images with model: {model}, prompt: {prompt[:50]}...")
        
        if model == "Stable Diffusion":
            return self._generate_with_stability_ai(
                prompt, negative_prompt, size, num_images, guidance_scale, steps, seed
            )
        elif model == "DALL-E":
            return self._generate_with_openai(prompt, size, num_images, negative_prompt)
        else:
            # Default to placeholder images for unsupported models
            print(f"Unsupported model: {model}")
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
            
            if negative_prompt:
                payload["text_prompts"].append({"text": negative_prompt, "weight": -1.0})
            
            if seed is not None:
                payload["seed"] = seed
            
            print(f"Sending request to Stability API with payload: {payload}")
            
            response = requests.post(url, headers=headers, json=payload)
            
            print(f"Stability API response status code: {response.status_code}")
            
            if response.status_code == 200:
                images = []
                data = response.json()
                for image_data in data["artifacts"]:
                    image_bytes = base64.b64decode(image_data["base64"])
                    image = Image.open(io.BytesIO(image_bytes))
                    images.append(image)
                print(f"Successfully generated {len(images)} images")
                return images
            else:
                print(f"Error from Stability API: {response.status_code}, {response.text}")
                return [self._create_placeholder_image(f"API Error: {response.status_code}", size) 
                       for _ in range(num_images)]
                
        except Exception as e:
            print(f"Exception in Stability AI generation: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return [self._create_placeholder_image(f"Error: {str(e)[:30]}...", size) 
                   for _ in range(num_images)]
    
    def _generate_with_openai(
        self, 
        prompt: str, 
        size: str, 
        num_images: int,
        negative_prompt: str = ""
    ) -> List[Image.Image]:
        """Generate images using OpenAI's DALL-E API."""
        from openai import OpenAI
        
        api_key = self.api_keys.get('openai')
        if not api_key:
            print("Warning: OpenAI API key not found. Using placeholder images.")
            return [self._create_placeholder_image(f"DALL-E: {prompt[:30]}...", size) 
                   for _ in range(num_images)]
        
        try:
            client = OpenAI(api_key=api_key)
            
            # Convert size to DALL-E format (e.g., "1024x1024")
            size_map = {
                "256x256": "256x256",
                "512x512": "512x512", 
                "1024x1024": "1024x1024",
                "768x768": "1024x1024",  # Fallback to closest supported size
                "1024x1536": "1024x1024"  # Fallback to closest supported size
            }
            dalle_size = size_map.get(size, "1024x1024")
            
            # For DALL-E, incorporate negative prompt if provided
            if negative_prompt:
                full_prompt = f"{prompt}. Please avoid: {negative_prompt}"
            else:
                full_prompt = prompt
                
            print(f"Sending request to OpenAI DALL-E with prompt: {full_prompt[:100]}...")
            
            response = client.images.generate(
                model="dall-e-3",
                prompt=full_prompt,
                size=dalle_size,
                quality="standard",
                n=num_images
            )
            
            images = []
            for data in response.data:
                image_url = data.url
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    image = Image.open(io.BytesIO(image_response.content))
                    images.append(image)
            
            print(f"Successfully generated {len(images)} images with DALL-E")
            return images
            
        except Exception as e:
            print(f"Exception in DALL-E generation: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return [self._create_placeholder_image(f"Error: {str(e)[:30]}...", size) 
                   for _ in range(num_images)]
    
    def _create_placeholder_image(self, text: str, size: str = "512x512") -> Image.Image:
        """Create a placeholder image with error text."""
        try:
            width, height = map(int, size.split('x'))
        except ValueError:
            width, height = 512, 512
            
        # Create a blank image
        image = Image.new("RGB", (width, height), color=(240, 240, 240))
        
        # Add text if PIL has ImageDraw
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)
            
            # Use default font
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
                
            # Add text to indicate error
            text_lines = [text[i:i+30] for i in range(0, len(text), 30)]
            y_position = height // 2 - 10 * len(text_lines)
            
            for line in text_lines:
                text_width = draw.textlength(line, font=font) if hasattr(draw, 'textlength') else 6 * len(line)
                draw.text(
                    ((width - text_width) // 2, y_position), 
                    line, 
                    fill=(0, 0, 0), 
                    font=font
                )
                y_position += 30
        except Exception as e:
            print(f"Error creating text on placeholder: {e}")
            
        return image
    
    def get_available_models(self) -> List[str]:
        """Return list of available models based on API keys."""
        models = []
        if 'stability' in self.api_keys:
            models.append("Stable Diffusion")
        if 'openai' in self.api_keys:
            models.append("DALL-E")
        return models
    
    def get_available_sizes(self, model: str = None) -> List[str]:
        """Return available sizes for the specified model."""
        if model == "DALL-E":
            return ["256x256", "512x512", "1024x1024"]
        else:
            return ["512x512", "768x768", "1024x1024", "1024x1536"]

