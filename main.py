import torch
import partition_scripts
import datetime
import flwr as fl
import gc

from neural_nets import _resnet18, get_parameters
from logging import INFO
from flwr.common.logger import log
from Clients import FlowerClient, weighted_average_resnet_CIFAR_open_set, fit_config, weighted_average

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
        client_resources = {"num_cpus": 8, "num_gpus": 0, "memory": 3*1024*1024*1024}

    return client_resources

client_resources = get_resources()

# Needed for initial params in fedAdam and FedYogi
#sample_net = _resnet18(classes=100,shape=(32,32))

def setup_strategies(sample_net, fraction_fit=1, fraction_eval=1):

    fedTAvg = fl.server.strategy.FedTrimmedAvg(
    beta = 0.2,
    fraction_fit=fraction_fit,  
    fraction_evaluate=fraction_eval,  
    min_fit_clients=1,  
    min_evaluate_clients=1, 
    min_available_clients=1,
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average,
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(sample_net))
)
    fedAdaGrad = fl.server.strategy.FedAdagrad(
    fraction_fit=fraction_fit,  
    fraction_evaluate=fraction_eval,  
    min_fit_clients=1,  
    min_evaluate_clients=1, 
    min_available_clients=1,
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average,
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(sample_net))
)
    scaffold = SCAFFOLDStrategy(
    fraction_fit=fraction_fit,  
    fraction_evaluate=fraction_eval,  
    min_fit_clients=1,  
    min_evaluate_clients=1, 
    min_available_clients=1,
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average,
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(sample_net))
)

    '''Just for a short experiment
    fedAvg = fl.server.strategy.FedAvg(
    fraction_fit=fraction_fit,  
    fraction_evaluate=fraction_eval,  
    min_fit_clients=1,  
    min_evaluate_clients=1, 
    min_available_clients=1,
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average_resnet_CIFAR_open_set,
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(sample_net))
)

    fedProx = fl.server.strategy.FedProx(
    fraction_fit=fraction_fit,  
    fraction_evaluate=fraction_eval,  
    min_fit_clients=1,  
    min_evaluate_clients=1, 
    min_available_clients=1,
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average_resnet_CIFAR_open_set,
    proximal_mu= 0.5
)

    fedAvgM = fl.server.strategy.FedAvgM(
    fraction_fit=fraction_fit,  
    fraction_evaluate=fraction_eval,  
    min_fit_clients=1,  
    min_evaluate_clients=1, 
    min_available_clients=1,
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average_resnet_CIFAR_open_set,
)

    fedAdam = fl.server.strategy.FedAdam(
    fraction_fit=fraction_fit,  
    fraction_evaluate=fraction_eval,  
    min_fit_clients=1,  
    min_evaluate_clients=1, 
    min_available_clients=1,
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average_resnet_CIFAR_open_set,
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(sample_net))
)

    fedYogi = fl.server.strategy.FedYogi(
    fraction_fit=fraction_fit,  
    fraction_evaluate=fraction_eval,  
    min_fit_clients=1,  
    min_evaluate_clients=1, 
    min_available_clients=1,
    on_fit_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average_resnet_CIFAR_open_set,
    initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(sample_net))
    )
    return (fedAvg, fedProx, fedAvgM, fedAdam, fedYogi)
    '''   

    return [scaffold]

# #### CIFAR10 setup: Client FNS ####
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
    return FlowerClient(net, trainloader, valloader, cid, DEVICE, classes).to_client()

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
    return FlowerClient(net, trainloader, valloader, cid, DEVICE, classes).to_client()

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
    return FlowerClient(net, trainloader, valloader, cid, DEVICE, classes).to_client()

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
    return FlowerClient(net, trainloader, valloader, cid, DEVICE, classes).to_client()


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
    return FlowerClient(net, trainloader, valloader, cid, DEVICE, classes).to_client()

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
    return FlowerClient(net, trainloader, valloader, cid, DEVICE, classes).to_client()


def run_FL_for_n_clients(the_client_fn, clients,sample_net, dataset_title, IID=True):
    all_metrics = {}
    text = "" if IID else "non-"
    for strategy in setup_strategies(sample_net, fraction_fit=1, fraction_eval=1):
        experiment = FedExperiment(client_fn=the_client_fn, strategy=strategy, name=f"{dataset_title} - {str(strategy)} - {clients} clients - {text} IID Distribution")
        all_metrics[str(strategy)] = experiment.simulate_FL(TRAINING_ROUNDS, clients)
    
    return all_metrics

##################################################################################
# SCAFFOLD STRATEGY
#################################################################################
class SCAFFOLDStrategy(fl.server.strategy.FedAvg):
    def __init__(self, C, **kwargs):
        super().__init__(**kwargs)
        self.global_c = None
        self.C = C  # Control variate provided as a parameter

    def initialize_parameters(self, client_manager):
        """Initialize the parameters and control variates."""
        initial_parameters = super().initialize_parameters(client_manager)
        if initial_parameters is not None:
            self.global_c = [torch.zeros_like(torch.tensor(param)) for param in initial_parameters.parameters()]
        return initial_parameters

    def aggregate_fit(self, rnd, results, failures):
        """Aggregate fit results using SCAFFOLD logic."""
        aggregated_weights = super().aggregate_fit(rnd, results, failures)

        if aggregated_weights is not None:
            num_clients = len(results)
            new_global_c = [torch.zeros_like(param) for param in self.global_c]
            
            for _, fit_res in results:
                local_c = fit_res.metrics['client_c']
                for gc, lc in zip(new_global_c, local_c):
                    gc += torch.tensor(lc) / num_clients

            self.global_c = new_global_c
            return aggregated_weights, {"global_c": self.global_c}

        return aggregated_weights, {"global_c": self.global_c}

    def configure_fit(self, rnd, parameters, client_manager):
        """Configure the next round of training."""
        config = super().configure_fit(rnd, parameters, client_manager)
        
        if self.global_c is None:
            self.global_c = [torch.zeros_like(torch.tensor(param)) for param in parameters]

        fit_ins = fl.common.FitIns(parameters, {"global_c": self.global_c, "C": self.C})
        
        # Distribute global parameters and control variates to all clients
        client_instructions = []
        for client_proxy in client_manager.sample_clients(num_clients=len(client_manager.clients)):
            client_instructions.append((client_proxy, fit_ins))

        return client_instructions

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
