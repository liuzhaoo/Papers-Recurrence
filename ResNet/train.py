import torch
import timeit
from datetime import datetime
import socket
import glob
import os
from dataset import  cifa10
# import tqdm
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
from model import resnet34,resnet50,resnet101
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
abspath = './'
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)
nEpochs = 500  # Number of epochs for training
resume_epoch = 0
num_classes = 10
snapshot = 50 # Store a model every snapshot epochs
lr = 1e-3
useTest = True
#


if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(abspath, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(abspath, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(abspath, 'run', 'run_' + str(run_id))
saveName = 'resnet-cifa10'
modelName = 'resnet34'

def train(save_dir=save_dir,lr =lr,num_epochs=nEpochs,
          save_epoch=snapshot,useTest=useTest):
    if modelName == 'resnet34':
        model = resnet34()
    elif modelName == 'resnet100':
        model = resnet101()

    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.1)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,gamma=0.1)
    transform = transforms.Compose([
        transforms.ToTensor(),

    ])


    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                       map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    model.to(device)
    criterion.to(device)
    # model = nn.DataParallel(model, device_ids=[0, 2])

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)
    print('Training model on {} dataset...'.format('cifa10'))

    train_dataloader = DataLoader(cifa10(abspath, train=True,transform = transform),batch_size=40, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(cifa10(abspath,train=False,transform=transform))

    train_size = len(train_dataloader.dataset)
    test_size = len(test_dataloader.dataset)

    for epoch in range(resume_epoch,num_epochs):
        start_time = timeit.default_timer()
        runningloss=0.0
        runnincorrect = 0.0

        model.train()

        for img, label in tqdm(train_dataloader):

            imgs = Variable(img,requires_grad=True).to(device)
            labels = Variable(label).to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]

            loss.backward()
            optimizer.step()
            # scheduler.step()


            runningloss += loss.item()*imgs.size(0)
            runnincorrect += torch.sum(preds == labels.data)

        epoch_loss = runningloss/train_size
        epoch_acc = runnincorrect.double()/train_size

        writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
        writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)

        print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format('train', epoch + 1, nEpochs, epoch_loss, epoch_acc))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))


        if useTest:
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()

if __name__ == '__main__':
    train()
