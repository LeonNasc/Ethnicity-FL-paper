import flwr as fl
from typing import List, Tuple
from logging import DEBUG
from flwr.common.logger import log
from neural_nets import get_parameters, set_parameters, train, test
from flwr.common import Metrics
from torcheval.metrics.functional import multiclass_f1_score

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader, cid, DEVICE):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.cid = cid
        self.round = 0
        self.device = DEVICE

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # Read values from config
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]

        # Use values provided by the config
        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=local_epochs, DEVICE=self.device)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy, f1, roc, kappa, precision, recall = test(self.net, self.valloader, DEVICE=self.device, classes=self.net.classes)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy), 
                                                    "f1": f1, 
                                                    "roc": roc, 
                                                    "kappa": kappa,
                                                    "precision": precision,
                                                    "recall":recall
                                                }


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return { "metrics": metrics, 
            "accuracy": sum(accuracies) / sum(examples)
            }

def fit_config(server_round: int):
    config = {
        "server_round": server_round,
        "local_epochs": 1,
    }
    return config

def display_parameters(arr):
    for i,layer in enumerate(arr):
        print(f"Layer {i}, shape {layer.shape}")
