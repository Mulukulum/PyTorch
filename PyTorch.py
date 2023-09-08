import torch
tens = torch.rand(3,4)

print(f"Shape of tensor: {tens.shape}")
print(f"Datatype of tensor: {tens.dtype}")
print(f"Device tensor is stored on: {tens.device}")

print(tens.view(-1,4))
if torch.cuda.is_available():
    device=torch.device('cuda')
    tnes=tens.to(device)
    print(tens)
    print(f"Device tensor is stored on: {tens.device}")
    tens=tens+tens
    print(tens)