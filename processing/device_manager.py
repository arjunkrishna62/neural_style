import torch
from torch.cuda.amp import GradScaler
import logging
import gc
import psutil
import os

class DeviceManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        self.setup_device()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing device manager on: {self.device}")
        
    def setup_device(self):
        if self.device.type == 'cuda':
            self.scaler = GradScaler(enabled=True)
            self.enable_cuda_optimizations()
            self.log_gpu_info()
        else:
            self.scaler = None
            self.log_cpu_info()

    def enable_cuda_optimizations(self):
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.cuda.empty_cache()
            
    def log_gpu_info(self):
        if self.device.type == 'cuda':
            props = torch.cuda.get_device_properties(0)
            self.logger.info(f"GPU: {props.name}")
            self.logger.info(f"Memory: {props.total_memory / 1e9:.2f} GB")
            
    def log_cpu_info(self):
        vm = psutil.virtual_memory()
        self.logger.info(f"CPU Memory: {vm.total / 1e9:.2f} GB")
        self.logger.info(f"CPU Cores: {os.cpu_count()}")
        
    def prepare_input(self, input_tensor):
        """Prepare input tensor for model"""
        if isinstance(input_tensor, torch.Tensor):
            return input_tensor.to(self.device)
        return input_tensor
    
    def optimize_memory(self):
        """Optimize memory usage"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()  # Always collect garbage

    def train_step(self, model, input_data, optimizer, loss_fn):
        input_data = self.prepare_input(input_data)
        optimizer.zero_grad()

        if self.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                output = model(input_data)
                loss = loss_fn(output)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
        else:
            output = model(input_data)
            loss = loss_fn(output)
            loss.backward()
            optimizer.step()

        return output, loss.item()

    @torch.no_grad()

    def inference_step(self, model, input_data):
        input_data = self.prepare_input(input_data)
        model.eval()

        if self.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                output = model(input_data)
        else:
            output = model(input_data)

        return output.cpu() if self.device.type == 'cuda' else output
    
    def __str__(self):
        """Return device information"""
        info = [f"Device: {self.device}"]
        if self.device.type == 'cuda':
            info.extend([
                f"GPU: {torch.cuda.get_device_name(0)}",
                f"Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB",
                f"Memory Cached: {torch.cuda.memory_reserved()/1e9:.2f}GB"
            ])
        else:
            vm = psutil.virtual_memory()
            info.extend([
                f"CPU Memory Used: {vm.used/1e9:.2f}GB",
                f"CPU Memory Total: {vm.total/1e9:.2f}GB"
            ])
        return "\n".join(info)