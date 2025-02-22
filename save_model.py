from processing.fast_nst import FastNST
import torch

# Initialize the model
model = FastNST()

# Save the model's state dictionary
save_path = "models/fast_nst.pth"
torch.save(model.state_dict(), save_path)

print(f"Model state dictionary saved to {save_path}")