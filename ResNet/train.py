import torch
import model
from dataset import  cifa10
from torch.utils.data import DataLoader
path = './'
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
train_dataloader = cifa10(path,train = True)


img,label = train_dataloader[99]

print(class_names[label])


