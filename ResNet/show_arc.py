
from torchsummary import summary
from cifa_model import resnet56
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet56()

model.to(device)


# summary(model, (3, 32, 32))

print(model)

