import os
import requests
from typing import Optional, Union, List
import sys
import copy
from tqdm import tqdm
import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL, 
    UNet2DConditionModel,
    DDIMScheduler
)
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
from safetensors.torch import load_file

p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd

# Constants
WEIGHT_INIT_VALUE = 1e-5
MODEL_URLS = {
    "edge_to_image": "https://www.cs.cmu.edu/~img2img-turbo/models/edge_to_image_loras.pkl",
    "sketch_to_image_stochastic": "https://www.cs.cmu.edu/~img2img-turbo/models/sketch_to_image_stochastic_lora.pkl"
}

def download_checkpoint(url: str, output_path: str) -> bool:
    """
    Download checkpoint from given URL to specified path.
    
    Args:
        url: Source URL for the checkpoint
        output_path: Destination path to save the file
        
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with tqdm(total=total_size, unit='iB', unit_scale=True) as progress_bar:
            with open(output_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
                    
        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Downloaded size does not match expected size")
            
        print(f"Downloaded successfully to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error downloading checkpoint: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

def download_url(url: str, output_path: str) -> bool:
    """
    Download a file from a URL to a local path with progress bar.
    
    Args:
        url: Source URL
        output_path: Destination path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with tqdm(total=total_size, unit='iB', unit_scale=True) as progress_bar:
            with open(output_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
                    
        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Downloaded size does not match expected size")
            
        print(f"Downloaded successfully to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

class TwinConv(nn.Module):
    """
    Dual convolution module that interpolates between pretrained and current weights.
    """
    def __init__(self, convin_pretrained: nn.Module, convin_curr: nn.Module):
        super(TwinConv, self).__init__()
        self.conv_in_pretrained = copy.deepcopy(convin_pretrained)
        self.conv_in_curr = copy.deepcopy(convin_curr)
        self.interpolation_weight: Optional[float] = None

    def forward(self, x: Tensor, num_inference_steps: 20) -> Tensor:
        self.scheduler.set_timesteps(num_inference_steps)
        self.timesteps = self.scheduler.timesteps
    
        if self.interpolation_weight is None:
            raise ValueError("Interpolation weight (r) must be set before forward pass")
            
        with torch.no_grad():
            x1 = self.conv_in_pretrained(x)
        x2 = self.conv_in_curr(x)
        return x1 * (1 - self.interpolation_weight) + x2 * self.interpolation_weight


class Pix2Pix_Turbo(nn.Module):
    def __init__(
        self,
        pretrained_name: Optional[str] = None,
        pretrained_path: Optional[str] = None,
        ckpt_folder: str = "checkpoints",
        cache_dir: str = "models",
        lora_rank_unet: int = 8,
        lora_rank_vae: int = 4
    ):
        """
        Initialize Pix2Pix Turbo model.
        
        Args:
            pretrained_name: Name of pretrained model to load
            pretrained_path: Path to custom pretrained weights
            ckpt_folder: Folder to store downloaded checkpoints
            cache_dir: Folder to store downloaded models
            lora_rank_unet: Rank for UNet LoRA adaptation
            lora_rank_vae: Rank for VAE LoRA adaptation
        """
        super().__init__()
        
        # Device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model components
        model_id = "stabilityai/sd-turbo"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, 
            subfolder="text_encoder"
        ).to(self.device)
        
        # Assign VAE as class attribute
        self.vae = AutoencoderKL.from_pretrained(
            model_id, 
            subfolder="vae"
        ).to(self.device)
        
        # Assign UNet as class attribute
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, 
            subfolder="unet"
        ).to(self.device)
        
        # Initialize other components
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # Load pretrained weights if specified
        if pretrained_name is not None:
            self._load_pretrained(pretrained_name, ckpt_folder)
        elif pretrained_path is not None:
            self._load_checkpoint(pretrained_path)
            
        # Initialize random components
        self._initialize_random(lora_rank_unet, lora_rank_vae)

        # Ensure all models are on the correct device
        self._move_to_device()

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint weights from file."""
        try:
            print("Loading checkpoint...")
            sd = torch.load(checkpoint_path, map_location="cpu")
            
            # Initialize LoRA configurations
            print("Initializing LoRA configurations...")
            unet_lora_config = LoraConfig(
                r=sd["rank_unet"],
                init_lora_weights="gaussian",
                target_modules=sd["unet_lora_target_modules"]
            )
            vae_lora_config = LoraConfig(
                r=sd["rank_vae"],
                init_lora_weights="gaussian",
                target_modules=sd["vae_lora_target_modules"]
            )
            
            # Initialize skip connections in decoder
            print("Initializing skip connections...")
            self.vae.decoder.skip_conv_1 = nn.Conv2d(
                in_channels=self.vae.config.latent_channels,
                out_channels=self.vae.config.latent_channels,
                kernel_size=1
            ).to(self.device)
            self.vae.decoder.skip_conv_2 = nn.Conv2d(
                in_channels=self.vae.config.latent_channels,
                out_channels=self.vae.config.latent_channels,
                kernel_size=1
            ).to(self.device)
            self.vae.decoder.skip_conv_3 = nn.Conv2d(
                in_channels=self.vae.config.latent_channels,
                out_channels=self.vae.config.latent_channels,
                kernel_size=1
            ).to(self.device)
            self.vae.decoder.skip_conv_4 = nn.Conv2d(
                in_channels=self.vae.config.latent_channels,
                out_channels=self.vae.config.latent_channels,
                kernel_size=1
            ).to(self.device)
            
            # Add LoRA adapters
            print("Adding LoRA adapters...")
            self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            self.unet.add_adapter(unet_lora_config)
            
            # Load state dicts
            print("Loading VAE state dict...")
            _sd_vae = self.vae.state_dict()
            for k in sd["state_dict_vae"]:
                _sd_vae[k] = sd["state_dict_vae"][k]
            self.vae.load_state_dict(_sd_vae, strict=False)
            
            print("Loading UNet state dict...")
            _sd_unet = self.unet.state_dict()
            for k in sd["state_dict_unet"]:
                _sd_unet[k] = sd["state_dict_unet"][k]
            self.unet.load_state_dict(_sd_unet, strict=False)
            
            # Store configurations
            self.lora_rank_unet = sd["rank_unet"]
            self.lora_rank_vae = sd["rank_vae"]
            self.target_modules_vae = sd["vae_lora_target_modules"]
            self.target_modules_unet = sd["unet_lora_target_modules"]
            
            print("Checkpoint loaded successfully!")
            
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            raise

    def _safe_adapter_management(self):
        """Safely remove existing adapters from VAE and UNet."""
        print("Cleaning up existing adapters...")
        
        # For VAE
        if hasattr(self.vae, "peft_config"):
            try:
                # Check if the adapter exists
                if "vae_skip" in self.vae.active_adapters:
                    print("Removing existing VAE adapter: vae_skip")
                    self.vae.delete_adapters("vae_skip")
            except Exception as e:
                print(f"Warning: Error cleaning VAE adapters: {e}")
        
        # For UNet
        if hasattr(self.unet, "peft_config"):
            try:
                # Check if the adapter exists
                if "unet_skip" in self.unet.active_adapters:
                    print("Removing existing UNet adapter: unet_skip")
                    self.unet.delete_adapters("unet_skip")
            except Exception as e:
                print(f"Warning: Error cleaning UNet adapters: {e}")

    def _initialize_random(self, lora_rank_unet: int, lora_rank_vae: int) -> None:
        """Initialize model with random weights."""
        print("Initializing model with random weights")

        if hasattr(self.vae, "peft_config") and "vae_skip" in self.vae.peft_config:
            print(f"Note: Adapter 'vae_skip' already exists, but we'll create a new one with a unique name")
        # Clean up existing adapters
        self._safe_adapter_management()
        
        # Initialize skip connections in decoder
        self.vae.decoder.skip_conv_1 = nn.Conv2d(
            in_channels=self.vae.config.latent_channels,
            out_channels=self.vae.config.latent_channels,
            kernel_size=1
        ).to(self.device)
        self.vae.decoder.skip_conv_2 = nn.Conv2d(
            in_channels=self.vae.config.latent_channels,
            out_channels=self.vae.config.latent_channels,
            kernel_size=1
        ).to(self.device)
        self.vae.decoder.skip_conv_3 = nn.Conv2d(
            in_channels=self.vae.config.latent_channels,
            out_channels=self.vae.config.latent_channels,
            kernel_size=1
        ).to(self.device)
        self.vae.decoder.skip_conv_4 = nn.Conv2d(
            in_channels=self.vae.config.latent_channels,
            out_channels=self.vae.config.latent_channels,
            kernel_size=1
        ).to(self.device)

        # Initialize skip connection weights
        for skip_conv in [
            self.vae.decoder.skip_conv_1,
            self.vae.decoder.skip_conv_2,
            self.vae.decoder.skip_conv_3,
            self.vae.decoder.skip_conv_4
        ]:
            nn.init.constant_(skip_conv.weight, WEIGHT_INIT_VALUE)

        # Initialize LoRA configurations
        target_modules_vae = [
            "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
            "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
            "to_k", "to_q", "to_v", "to_out.0",
        ]
        vae_lora_config = LoraConfig(
            r=lora_rank_vae, 
            init_lora_weights="gaussian",
            target_modules=target_modules_vae
        )
        import uuid
        unique_adapter_name = f"vae_skip_{uuid.uuid4().hex[:8]}"
        self.vae.add_adapter(vae_lora_config, adapter_name=unique_adapter_name)
    
        target_modules_unet = [
            "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", 
            "conv_shortcut", "conv_out", "proj_in", "proj_out", 
            "ff.net.2", "ff.net.0.proj"
        ]
        unet_lora_config = LoraConfig(
            r=lora_rank_unet, 
            init_lora_weights="gaussian",
            target_modules=target_modules_unet
        )
        self.unet.add_adapter(unet_lora_config, adapter_name="unet_skip")

        # Store configurations
        self.lora_rank_unet = lora_rank_unet
        self.lora_rank_vae = lora_rank_vae
        self.target_modules_vae = target_modules_vae
        self.target_modules_unet = target_modules_unet

    def _move_to_device(self) -> None:
        """Move all model components to the specified device."""
        self.unet = self.unet.to(self.device)
        self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999], device=self.device).long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def forward(
        self,
        control_image: Tensor,
        prompt: Optional[str] = None,
        prompt_tokens: Optional[Tensor] = None,
        deterministic: bool = True,
        interpolation_weight: float = 1.0,
        noise_map: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass of the model.
        
        Args:
            control_image: Input control image
            prompt: Text prompt (mutually exclusive with prompt_tokens)
            prompt_tokens: Pre-tokenized prompt (mutually exclusive with prompt)
            deterministic: Whether to use deterministic generation
            interpolation_weight: Interpolation weight between pretrained and current weights
            noise_map: Optional noise map for non-deterministic generation
            
        Returns:
            Tensor: Generated image
        """
        # either the prompt or the prompt_tokens should be provided
        assert (prompt is None) != (prompt_tokens is None), "Either prompt or prompt_tokens should be provided"

        if prompt is not None:
            # encode the text prompt and move tokens to the detected device
            caption_tokens = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.device)
            caption_enc = self.text_encoder(caption_tokens)[0]
        else:
            caption_enc = self.text_encoder(prompt_tokens)[0]

        if deterministic:
            encoded_control = self.vae.encode(control_image).latent_dist.sample() * self.vae.config.scaling_factor
            model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc).sample
            x_denoised = self.scheduler.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
            x_denoised = x_denoised.to(model_pred.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        else:
            # scale the lora weights based on the r value
            self.unet.set_adapters(["default"], weights=[interpolation_weight])
            set_weights_and_activate_adapters(self.vae, ["vae_skip"], [interpolation_weight])
            encoded_control = self.vae.encode(control_image).latent_dist.sample() * self.vae.config.scaling_factor
            # combine the input and noise
            unet_input = encoded_control * interpolation_weight + noise_map * (1 - interpolation_weight)
            self.unet.conv_in.r = interpolation_weight
            unet_output = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc).sample
            self.unet.conv_in.r = None
            x_denoised = self.scheduler.step(unet_output, self.timesteps, unet_input, return_dict=True).prev_sample
            x_denoised = x_denoised.to(unet_output.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = interpolation_weight
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return output_image

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_target_modules"] = self.target_modules_unet
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        torch.save(sd, outf)

    def _load_pretrained(self, pretrained_name, ckpt_folder):
        """Load pretrained weights for the model."""
        try:
            # Map model names to their URLs
            model_urls = {
                "edge_to_image": "https://www.cs.cmu.edu/~img2img-turbo/models/edge_to_image_loras.pkl",
                "sketch_to_image_stochastic": "https://www.cs.cmu.edu/~img2img-turbo/models/sketch_to_image_stochastic_lora.pkl"
            }
            
            if pretrained_name not in model_urls:
                raise ValueError(f"Unknown pretrained model name: {pretrained_name}")
            
            # Create checkpoint folder if it doesn't exist
            os.makedirs(ckpt_folder, exist_ok=True)
            
            # Construct local path for the checkpoint
            local_path = os.path.join(ckpt_folder, f"{pretrained_name}.pkl")
            
            # Load the checkpoint first to get the shapes
            if os.path.exists(local_path):
                print(f"Loading checkpoint from {local_path}")
                checkpoint = torch.load(local_path, map_location="cpu")
            else:
                print(f"Downloading {pretrained_name} checkpoint...")
                success = download_url(model_urls[pretrained_name], local_path)
                if not success:
                    raise RuntimeError(f"Failed to download checkpoint")
                checkpoint = torch.load(local_path, map_location="cpu")
            
            # Clean up existing adapters
            self._safe_adapter_management()
            
            # Initialize LoRA configurations
            print("Initializing LoRA configurations...")
            unet_lora_config = LoraConfig(
                r=checkpoint["rank_unet"],
                init_lora_weights="gaussian",
                target_modules=checkpoint["unet_lora_target_modules"]
            )
            vae_lora_config = LoraConfig(
                r=checkpoint["rank_vae"],
                init_lora_weights="gaussian",
                target_modules=checkpoint["vae_lora_target_modules"]
            )
            
            # Initialize skip connections with shapes from checkpoint
            self._initialize_skip_connections(checkpoint)
            
            # Add LoRA adapters
            print("Adding LoRA adapters...")
            self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
            self.unet.add_adapter(unet_lora_config)
            
            # Load state dicts
            print("Loading model weights...")
            self.vae.load_state_dict(checkpoint["state_dict_vae"], strict=False)
            self.unet.load_state_dict(checkpoint["state_dict_unet"], strict=False)
            
            # Store configurations
            self.lora_rank_unet = checkpoint["rank_unet"]
            self.lora_rank_vae = checkpoint["rank_vae"]
            self.target_modules_vae = checkpoint["vae_lora_target_modules"]
            self.target_modules_unet = checkpoint["unet_lora_target_modules"]
            
            print(f"Successfully loaded {pretrained_name} checkpoint")
            
        except Exception as e:
            print(f"Error loading pretrained model: {str(e)}")
            raise

    def _initialize_skip_connections(self, checkpoint=None):
        """Initialize skip connections in the VAE decoder with correct shapes."""
        print("Initializing skip connections...")
        
        # Get shapes from checkpoint if available, otherwise use defaults
        if checkpoint and "state_dict_vae" in checkpoint:
            skip_shapes = {}
            for i in range(1, 5):
                key = f"decoder.skip_conv_{i}.base_layer.weight"
                if key in checkpoint["state_dict_vae"]:
                    shape = checkpoint["state_dict_vae"][key].shape
                    skip_shapes[i] = (shape[0], shape[1])  # (out_channels, in_channels)
        else:
            # Default shapes if no checkpoint available
            skip_shapes = {
                1: (512, 512),
                2: (256, 256),
                3: (256, 128),
                4: (256, 128)
            }
        
        # Initialize skip connections with correct shapes
        for i, (out_ch, in_ch) in skip_shapes.items():
            skip_conv = nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=1
            ).to(self.device)
            setattr(self.vae.decoder, f"skip_conv_{i}", skip_conv)
            print(f"Initialized skip_conv_{i} with shape: ({in_ch}, {out_ch})")
