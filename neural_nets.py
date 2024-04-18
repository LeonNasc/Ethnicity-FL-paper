import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torcheval
import torcheval.metrics
from torchmetrics.functional.classification import multiclass_cohen_kappa

from torchvision.models import resnet50, resnet18 

from typing import List, Tuple
from collections import OrderedDict

def _make_layers(cfg):
    layers = []
    in_channels = 3
    for layer_cfg in cfg:
        if layer_cfg == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=layer_cfg,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=True))
            layers.append(nn.BatchNorm2d(num_features=layer_cfg))
            layers.append(nn.ReLU(inplace=True))
            in_channels = layer_cfg
    return nn.Sequential(*layers)

class _VGG(nn.Module):
    """
    VGG module for 3x32x32 input, 10 classes
    """

    def __init__(self, name, classes=10, shape=(32,32)):
        super(_VGG, self).__init__()
        cfg = _cfg[name]
        self.layers = _make_layers(cfg)
        self.classes = classes
        #print(f'fc1 expects: {2*shape[0]*shape[1]}')
        num_pools = len([x for x in cfg if x == "M"])
        shape = int(shape[0]*shape[1] * (512*(1/4)**num_pools))
        self.fc1 = nn.Linear(shape, 4096)  
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # Adding dropout for regularization

    def forward(self, x):
        
        y = self.layers(x)
        y = y.view(y.size(0), -1)  # Flatten the input features
        y = self.fc1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.fc3(y)
        return y
    

class Net(nn.Module):
    def __init__(self, classes=10, shape=(32,32)) -> None:
        super(Net, self).__init__()
        self.base = 128
        self.input_shape = shape
        self.conv1 = nn.Conv2d(3, self.base, 5, padding=2)  # Add padding to preserve spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.base, self.base*2, 5, padding=2)  # Add padding to preserve spatial dimensions
        self.conv3 = nn.Conv2d(self.base*2, 256, 5, padding=2)  # Add padding to preserve spatial dimensions

        # Calculate the size of the fully connected layer dynamically
        self.fc_input_size = self.calculate_fc_input_size()

        self.fc1 = nn.Linear(self.fc_input_size, 120)
        self.fc2 = nn.Linear(120, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convolutional_layers_output(x)
        # Dynamic input size handling
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def calculate_fc_input_size(self) -> int:
        # Create a sample input and pass it through the convolutional layers
        # to get the output shape for the fully connected layer
        sample_input = torch.randn(1, 3, self.input_shape[0], self.input_shape[1])  # Assuming input size is 32x32
        sample_output = self.convolutional_layers_output(sample_input)
        return sample_output.view(sample_output.size(0), -1).size(1)

    def convolutional_layers_output(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x



def train(net, trainloader, epochs: int, verbose=False, DEVICE="cpu"):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total

        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader, DEVICE="cpu", classes=2):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss, f1, roc, kappa  = 0, 0, 0.0, 0.0, 0.0, 0.0
    net.eval()
    with torch.no_grad():
        batches = 1
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)

            loss += criterion(outputs, labels).item()
            roc += torcheval.metrics.functional.multiclass_auroc(outputs, labels, num_classes= classes)

            _, predicted = torch.max(outputs.data, 1)

            f1 += torcheval.metrics.functional.multiclass_f1_score(predicted, labels, num_classes= classes)
            precision = torcheval.metrics.functional.multiclass_precision(predicted, labels, num_classes= classes)
            recall = torcheval.metrics.functional.multiclass_recall(predicted, labels, num_classes= classes)
            kappa += multiclass_cohen_kappa(predicted, labels, num_classes=classes)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batches += 1

    loss /= batches
    kappa /= batches
    f1 /= batches
    roc /= batches
    accuracy = correct / total

    return loss, accuracy, f1, roc, kappa, precision, recall

def centralized_training(trainloader, valloader, testloader, DEVICE="cpu", net=None, epochs=5, classes=10):

        if net is None:
            print("choosing random net")
            net = Net(classes).to(DEVICE)
        else:
            net = net.to(DEVICE)

        for i in range(10):
            current_time = datetime.datetime.now()
            print(f"Begin:Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            train(net, trainloader, epochs, DEVICE=DEVICE)
            metrics = test(net, valloader, classes=classes, DEVICE=DEVICE)

            current_time = datetime.datetime.now()
            print(f"End: Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            # Get the current date and time
            print(f"Validation round #{i} loss {metrics[0]}, accuracy {metrics[1]}, f1 {metrics[2]}")

        metrics = test(net, testloader, DEVICE=DEVICE, classes=classes)
        print(f"Final test set performance:\n\tloss {metrics[0]}\n\taccuracy {metrics[1]}")

        return metrics


# VGG Setups 
_cfg = {    
    'VGG7': [64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'VGG9': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def VGG7(classes, shape=(32,32)):
    return _VGG('VGG7', classes, shape)
    
def VGG9(classes, shape=(32,32)):
    return _VGG('VGG9', classes, shape)

def VGG11(classes, shape=(32,32)):
    return _VGG('VGG11', classes, shape)

def VGG13(classes, shape=(32,32)):
    return _VGG('VGG13', classes, shape)

def VGG16(classes, shape=(32,32)):
    return _VGG('VGG16', classes, shape)

def VGG19(classes, shape=(32,32)):
    return _VGG('VGG19', classes, shape)

##################################################
# Resnet50 network

class MyResNet50(nn.Module):
    def __init__(self, classes=1000):
        super(MyResNet50, self).__init__()
        self.model = resnet50()
        self.model.fc = nn.Linear(512, classes)

    def forward(self, x):
        return self.model.forward(x)

def _resnet18(classes, shape=None):
    net = resnet18()
    net.fc = nn.Linear(512, classes)
    
    return net

######################################## 
#   Neural Network Parameter Funcionality for FL
#   Taken from flower tutorial (Flower Basics)
######################################## 
def get_parameters(net) -> List[np.ndarray]:
    # Returns the parameters of the model as a list of numpy arrays
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    #Arturas Kairys Updates the given net with the given parameters
    # (retruns nothing as it will just modify the net object)
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k:torch.Tensor(v) for k, v in params_dict if len(v.shape) > 0})
    net.load_state_dict(state_dict, strict=True)
