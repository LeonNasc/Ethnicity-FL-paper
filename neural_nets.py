import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torcheval
import torcheval.metrics
from torchmetrics.functional.classification import multiclass_cohen_kappa


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
    optimizer = torch.optim.Adam(net.parameters())
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
        torch.manual_seed(0)

        if net is None:
            net = Net(classes).to(DEVICE)
        else:
            net = net.to(DEVICE)

        for epoch in range(epochs):
            train(net, trainloader, 1, DEVICE=DEVICE)
            print(DEVICE)
            metrics = test(net, valloader, classes=classes, DEVICE=DEVICE)

            # Get the current date and time
            current_time = datetime.datetime.now()
            print(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Epoch {epoch+1}: validation loss {metrics[0]}, accuracy {metrics[1]}")

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
##################################################
class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(IdentityBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels * 4)
        )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        identity = self.shortcut(identity)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ConvBlock(64, out_channels, stride=stride))
        for _ in range(1, blocks):
            layers.append(IdentityBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


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
