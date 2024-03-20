import torch
import partition_scripts
import datetime
import flwr as fl

from neural_nets import VGG7, get_parameters
from logging import INFO
from flwr.common.logger import log
from Clients import FlowerClient, weighted_average, fit_config

today = datetime.datetime.today()
fl.common.logger.configure(identifier="FL Paper Experiment", filename=f"./logs/log_FLWR_{today.timestamp()}.txt")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_STORE = {}

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
        client_resources = {"num_cpus": 2, "num_gpus": 0}
    else:
        client_resources = {"num_cpus": 2, "num_gpus": 1}

    return client_resources

client_resources = get_resources()

# Needed for initial params in fedAdam and FedYogi
#sample_net = VGG7(classes=2,shape=(64,64)) #Only for CelebA/FedFaces - Edit class numbers
#sample_net = VGG7(classes=100,shape=(32,32)) #Only for CIFAR10 - Edit class numbers

def setup_strategies(sample_net, fraction_fit=1, fraction_eval=1):
    fedAvg = fl.server.strategy.FedAvg(
    fraction_fit=fraction_fit,  
    fraction_evaluate=fraction_eval,  
    min_fit_clients=1,  
    min_evaluate_clients=1, 
    min_available_clients=1,
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average,
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(sample_net))
)

    fedProx = fl.server.strategy.FedProx(
    fraction_fit=fraction_fit,  
    fraction_evaluate=fraction_eval,  
    min_fit_clients=1,  
    min_evaluate_clients=1, 
    min_available_clients=1,
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average,
    proximal_mu= 0.5
)

    fedAvgM = fl.server.strategy.FedAvgM(
    fraction_fit=fraction_fit,  
    fraction_evaluate=fraction_eval,  
    min_fit_clients=1,  
    min_evaluate_clients=1, 
    min_available_clients=1,
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average,
)

    fedAdam = fl.server.strategy.FedAdam(
    fraction_fit=fraction_fit,  
    fraction_evaluate=fraction_eval,  
    min_fit_clients=1,  
    min_evaluate_clients=1, 
    min_available_clients=1,
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average,
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(sample_net))
)

    fedYogi = fl.server.strategy.FedYogi(
    fraction_fit=fraction_fit,  
    fraction_evaluate=fraction_eval,  
    min_fit_clients=1,  
    min_evaluate_clients=1, 
    min_available_clients=1,
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average,
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(sample_net))
)
    
    return (fedAvg, fedProx, fedAvgM, fedAdam, fedYogi)


# #### CIFAR10 setup: Client FNS ####
# A couple of client_fns for using with Flower, one for each dataset experiment
def client_fn_CIFAR10_IID(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Create model
    net = VGG7(classes=10).to(DEVICE)

    # Load data (CIFAR-10)
    trainloaders, valloaders,_ =  DATA_STORE["CIFAR10_IID"]
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader, cid, DEVICE).to_client()

def client_fn_CIFAR10_nonIID(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Create model
    net = VGG7(classes=10).to(DEVICE)

    # Load data (CIFAR-10)
    trainloaders, valloaders,_ =  DATA_STORE["CIFAR10_NonIID"]
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader, cid, DEVICE).to_client()

def client_fn_CIFAR100_nonIID(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Create model
    net = VGG7(classes=100).to(DEVICE)

    # Load data (CIFAR-10)
    trainloaders, valloaders,_ =  DATA_STORE["CIFAR100_nonIID"]
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader, cid, DEVICE).to_client()


def client_fn_CelebA_IID(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Create model
    net = VGG7(classes=2, shape=(64,64)).to(DEVICE)

    # Load data (CIFAR-10)
    trainloaders, valloaders,_ =  DATA_STORE["CelebA_IID"]
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader, cid, DEVICE).to_client()


def run_FL_for_n_clients(the_client_fn, clients,sample_net):
    all_metrics = {}
    for strategy in setup_strategies(sample_net):
        experiment = FedExperiment(client_fn=the_client_fn, strategy=strategy, name=f"CIFAR10 - {str(strategy)} - IID Distribution")
        all_metrics[str(strategy)] = experiment.simulate_FL(TRAINING_ROUNDS, clients)
    
    return all_metrics

if __name__ == "__main__":
    print(
        f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
    )

    TRAINING_ROUNDS = 100

    sample_net = VGG7(classes=10, shape=(32,32))
    clients = 10
    DATA_STORE["CIFAR10_IID"]= partition_scripts.partition_CIFAR_IID(clients, "CIFAR10")
    run_FL_for_n_clients(client_fn_CIFAR10_IID, clients, sample_net)

    clients = 50
    DATA_STORE["CIFAR10_IID"]= partition_scripts.partition_CIFAR_IID(clients, "CIFAR10")
    run_FL_for_n_clients(client_fn_CIFAR10_IID, clients, sample_net)

    clients = 100
    DATA_STORE["CIFAR10_IID"]= partition_scripts.partition_CIFAR_IID(clients, "CIFAR10")
    run_FL_for_n_clients(client_fn_CIFAR10_IID, clients, sample_net)
