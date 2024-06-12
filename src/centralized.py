import torch
import src.partition_scripts as partition_scripts
from torchvision.models import resnet18
from src.neural_nets import centralized_training, get_parameters, MyResNet50

DATA_STORE = {
    "CIFAR10_IID": None,
    "CIFAR10_NonIID": None,
    "CIFAR100_IID": None,
    "CIFAR100_NonIID": None,
    "FedFaces_IID": None,
    "FedFaces_NonIID": None,

}

#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

centralized = True
experiments = ["CIFAR10", "CIFAR100", "CelebA", "FedFaces"]
epochs = 2

def run_centralized(experiment):
    def setup_net(classes):
        net = resnet18()
        #net.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) 
        net.fc = torch.nn.Linear(512, classes)
        return net
 
    match experiment:
        case "CIFAR10":
            print("Starting CIFAR10")
            DATA_STORE["CIFAR10"] = partition_scripts.partition_CIFAR_IID(1)
            dataloaders, valloaders, testloaders = DATA_STORE["CIFAR10"]
            net = setup_net(10)
            centralized_training(trainloader=dataloaders[0], valloader=valloaders[0], testloader=testloaders, net=net, epochs=epochs, classes=10, DEVICE=DEVICE)
        case "CIFAR100":
            print("Starting CIFAR10")
            DATA_STORE["CIFAR100"] = partition_scripts.partition_CIFAR_IID(1, "CIFAR100")
            dataloaders, valloaders, testloaders = DATA_STORE["CIFAR100"]
            net = MyResNet50(classes=100)
            centralized_training(trainloader=dataloaders[0], valloader=valloaders[0], testloader=testloaders, epochs=epochs, classes=100, DEVICE=DEVICE)
        case "CelebA":
            print("Starting CIFAR10")
            DATA_STORE["CelebA"] = partition_scripts.partition_CelebA_IID(1)
            dataloaders, valloaders, testloaders = DATA_STORE["CelebA"]
            net = MyResNet50(classes=2, shape=(64, 64))
            centralized_training(trainloader=dataloaders[0], valloader=valloaders[0], testloader=testloaders, epochs=epochs, net=net, DEVICE=DEVICE)
        case "FedFaces":
            print("Starting FedFaces")
            DATA_STORE["FedFaces"] = partition_scripts.partition_FedFaces_IID(1)
            dataloaders, valloaders, testloaders = DATA_STORE["FedFaces"]
            net = setup_net(4)
            centralized_training(trainloader=dataloaders[0], valloader=valloaders[0], net=net, testloader=testloaders, epochs=epochs, classes=4, DEVICE=DEVICE)
        case _:
            pass

if centralized:
    run_centralized(experiments[0])
    #run_centralized(experiments[1])
    #run_centralized(experiments[-1])
    #run_centralized(experiments[2])


