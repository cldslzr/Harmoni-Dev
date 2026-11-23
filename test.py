import torch

print("CUDA available ?", torch.cuda.is_available())
print("Nb of GPUs :", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Current GPU :", torch.cuda.get_device_name(0))

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(DEVICE)