import torch
# import os
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

loaded_data = torch.load("/home/rusk/projects/nnUNet-KiTS/DATASET/nnUNet_trained_models/Dataset996_KiTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth")

# Access specific keys from the dictionary
model_weights = loaded_data["model_state_dict"]
optimizer_state = loaded_data["optimizer_state_dict"]
epoch_number = loaded_data["epoch"]

print(f"Model weights: {model_weights}")
print(f"Optimizer state: {optimizer_state}")
print(f"Epoch number: {epoch_number}")

torch.serialization._use_new_zipfile_serialization = True