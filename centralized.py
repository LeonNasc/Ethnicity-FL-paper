import torch
import partition_scripts
from neural_nets import centralized_training, VGG7, get_parameters

DATA_STORE = {
    "CIFAR10_IID": None,
    "CIFAR10_NonIID": None,
    "CIFAR100_IID": None,
    "CIFAR100_NonIID": None,
    "FedFaces_IID": None,
    "FedFaces_NonIID": None,

}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

centralized = False
experiments = ["CIFAR10", "CIFAR100", "CelebA", "FedFaces"]
epochs = 400

def run_centralized(experiment):
    match experiment:
        case "CIFAR10":
            DATA_STORE["CIFAR10"] = partition_scripts.partition_CIFAR_IID(2)
            dataloaders, valloaders, testloaders = DATA_STORE["CIFAR10"]
            net = VGG7(classes=10)
            centralized_training(trainloader=dataloaders[0], valloader=valloaders[0], testloader=testloaders, net=net, epochs=epochs, classes=10, DEVICE=DEVICE)
        case "CIFAR100":
            DATA_STORE["CIFAR100"] = partition_scripts.partition_CIFAR_IID(2, "CIFAR100")
            dataloaders, valloaders, testloaders = DATA_STORE["CIFAR100"]
            net = VGG7(classes=100)
            centralized_training(trainloader=dataloaders[0], valloader=valloaders[0], testloader=testloaders, epochs=epochs, classes=100, DEVICE=DEVICE)
        case "CelebA":
            DATA_STORE["CelebA"] = partition_scripts.partition_CelebA_IID(2)
            dataloaders, valloaders, testloaders = DATA_STORE["CelebA"]
            net = VGG7(classes=2, shape=(64, 64))
            centralized_training(trainloader=dataloaders[0], valloader=valloaders[0], testloader=testloaders, epochs=epochs, net=net, DEVICE=DEVICE)
        case "FedFaces":
            DATA_STORE["FedFaces"] = partition_scripts.partition_FedFaces_IID(2)
            dataloaders, valloaders, testloaders = DATA_STORE["FedFaces"]
            net = VGG7(classes=4, shape=(64,64))
            centralized_training(trainloader=dataloaders[0], valloader=valloaders[0], net=net, testloader=testloaders, epochs=epochs, classes=3, DEVICE=DEVICE)
        case _:
            pass

if centralized:
    run_centralized(experiments[0])
    run_centralized(experiments[1])
    run_centralized(experiments[2])
    run_centralized(experiments[-1])


