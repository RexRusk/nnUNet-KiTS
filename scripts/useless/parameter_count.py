
import torch

def count_parameters_in_model(model_state_dict):
    """
    Counts the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model_state_dict.values() if torch.is_tensor(p))

# Load the .pth file
model_path = ''  # Replace with the actual path to your .pth file
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # Use 'cpu' or 'cuda' based on your setup

# Ensure the checkpoint contains a state_dict
if 'state_dict' in checkpoint:
    model_state_dict = checkpoint['state_dict']
else:
    model_state_dict = checkpoint  # In case the entire state_dict is saved directly without a 'state_dict' key

# Filter out non-tensor elements to avoid AttributeError
model_state_dict = {k: v for k, v in model_state_dict.items() if torch.is_tensor(v)}

# Count the parameters
total_params = count_parameters_in_model(model_state_dict)
print(f"The model has {total_params} trainable parameters.")