import torch
import src.partition_scripts as partition_scripts
import datetime
import flwr as fl
import gc

from src.neural_nets import _resnet18, get_parameters
from logging import INFO
from flwr.common.logger import log
from src.Clients import FlowerClient, fit_config, weighted_average
from flwr_baselines.clients import FlowerClientFedNova, FlowerClientScaffold
from flwr_baselines.strategy import FedNovaStrategy, ScaffoldStrategy

today = datetime.datetime.today()
fl.common.logger.configure(identifier="FL Paper Experiment", filename=f"./logs/log_FLWR_{today.timestamp()}.txt")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_STORE = {}
CLIENT_OBJ = FlowerClient

class FedExperiment():

    def __init__(self, client_fn,strategy, name="New experiment"):
        self.client_fn = client_fn
        self.strategy = strategy
        self.name = name

    def simulate_FL(self, rounds, clients):
        log(INFO, "\n" + 10 * "========" + "\n" + self.name + " has started\n" + 10 * "========"  )
        metrics = fl.simulation.start_simulation(
                            client_fn=self.client_fn,
                            num_clients=clients,
                            config=fl.server.ServerConfig(num_rounds=rounds),
                            strategy=self.strategy,
                            client_resources=client_resources,
                        )
        log(INFO, "\n" + 10 * "========" + "\n" + self.name + " has ended\n" + 10 * "========"  )
        return metrics


# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
def  get_resources():
    if DEVICE.type == "cpu":
        client_resources = {"num_cpus": 8, "num_gpus": 0, "memory": 3*1024*1024*1024}

    return client_resources

client_resources = get_resources()


# A couple of client_fns for using with Flower, one for each dataset experiment
def client_fn_CIFAR10_IID(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    classes = 10
    # Create model
    net = _resnet18(classes=classes, shape=(32,32)).to(DEVICE)

    # Load data (CIFAR-10)
    trainloaders, valloaders,_ =  DATA_STORE["CIFAR10_IID"]
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return CLIENT_OBJ(net, trainloader, valloader, cid, DEVICE, classes).to_client()

def client_fn_CIFAR10_nonIID(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Create model
    classes = 10
    net = _resnet18(classes=10).to(DEVICE)

    # Load data (CIFAR-10)
    trainloaders, valloaders,_ =  DATA_STORE["CIFAR10_NonIID"]
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return CLIENT_OBJ(net, trainloader, valloader, cid, DEVICE, classes).to_client()

def client_fn_CIFAR100_nonIID(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    classes=100
    # Create model
    net = _resnet18(classes=100).to(DEVICE)

    # Load data (CIFAR-10)
    trainloaders, valloaders,_ =  DATA_STORE["CIFAR100_nonIID"]
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return CLIENT_OBJ(net, trainloader, valloader, cid, DEVICE, classes).to_client()

def client_fn_CIFAR100_IID(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    classes=100
    # Create model
    net = _resnet18(classes=100).to(DEVICE)

    # Load data (CIFAR-10)
    trainloaders, valloaders,_ =  DATA_STORE["CIFAR100_IID"]
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return CLIENT_OBJ(net, trainloader, valloader, cid, DEVICE, classes).to_client()


def client_fn_FedFaces_IID(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    classes=4
    # Create model
    net = _resnet18(classes=4, shape=(32,32)).to(DEVICE)

    # Load data (CIFAR-10)
    trainloaders, valloaders,_ =  DATA_STORE["FedFaces_IID"]
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return CLIENT_OBJ(net, trainloader, valloader, cid, DEVICE, classes).to_client()

def client_fn_FedFaces_nonIID(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    classes=4
    # Create model
    net = _resnet18(classes=4, shape=(32,32)).to(DEVICE)

    # Load data (CIFAR-10)
    trainloaders, valloaders,_ =  DATA_STORE["FedFaces_nonIID"]
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return CLIENT_OBJ(net, trainloader, valloader, cid, DEVICE, classes).to_client()


def run_FL_for_n_clients(the_client_fn, clients,sample_net, dataset_title, IID=True):
    all_metrics = {}
    text = "" if IID else "non-"
    #for strategy in setup_strategies(sample_net, fraction_fit=1, fraction_eval=1): #Original implementation
    strategies = [ScaffoldStrategy, FedNovaStrategy]
    for i, client_type in enumerate([FlowerClientScaffold, FlowerClientFedNova]):
        CLIENT_OBJ = client_type
        strategy = strategies[i] 
        experiment = FedExperiment(client_fn=the_client_fn, strategy=strategy, name=f"{dataset_title} - {str(strategy)} - {clients} clients - {text} IID Distribution")
        all_metrics[str(strategy)] = experiment.simulate_FL(TRAINING_ROUNDS, clients)
    
    return all_metrics

#######################################################################################################
if __name__ == "__main__":
    print(
        f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
    )

    TRAINING_ROUNDS = 2

    classes = 10 #+ 1
    sample_net = _resnet18(classes=classes, shape=(32,32))

    clients = 5
    DATA_STORE["CIFAR10_IID"]= partition_scripts.partition_CIFAR_IID(clients, "CIFAR10")
    run_FL_for_n_clients(client_fn_CIFAR10_IID, clients, sample_net, "CIFAR10")
    del DATA_STORE["CIFAR10_IID"]

    DATA_STORE["CIFAR10_nonIID"]= partition_scripts.partition_CIFAR_nonIID(clients, "CIFAR10")
    run_FL_for_n_clients(client_fn_CIFAR10_nonIID, clients, sample_net, "CIFAR10", IID=False)
    del DATA_STORE["CIFAR10_nonIID"]

    classes = 100 #+ 1
    DATA_STORE["CIFAR100_IID"]= partition_scripts.partition_CIFAR_IID(clients, "CIFAR100")
    run_FL_for_n_clients(client_fn_CIFAR100_IID, clients, sample_net, "CIFAR100")
    del DATA_STORE["CIFAR100_IID"]

    DATA_STORE["CIFAR100_nonIID"]= partition_scripts.partition_CIFAR_nonIID(clients, "CIFAR100")
    run_FL_for_n_clients(client_fn_CIFAR100_nonIID, clients, sample_net, "CIFAR100", IID=False)
    del DATA_STORE["CIFAR100_nonIID"]

    classes = 4 #+ 1
    DATA_STORE["FedFaces_IID"]= partition_scripts.partition_FedFaces_IID(clients)
    run_FL_for_n_clients(client_fn_FedFaces_IID, clients, sample_net, "FedFaces")
    del DATA_STORE["FedFaces_IID"]

    DATA_STORE["FedFaces_nonIID"]= partition_scripts.partition_FedFaces_nonIID(clients)
    run_FL_for_n_clients(client_fn_FedFaces_nonIID, clients, sample_net, "FedFaces", IID=False)
    del DATA_STORE["FedFaces_nonIID"]
