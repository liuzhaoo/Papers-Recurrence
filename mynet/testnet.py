import LAU_net
import torch

model = LAU_net.gennerate_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data = torch.rand((1, 3, 23, 299, 299),device=device)
model.to(device)
out = model(data)
print(out.size())
