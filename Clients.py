import flwr as fl
import csv
import partition_scripts
import numpy as np
from typing import List, Tuple
from logging import DEBUG
from flwr.common.logger import log
from neural_nets import get_parameters, set_parameters, train, test, _resnet18
from flwr.common import Metrics
from torcheval.metrics.functional import multiclass_f1_score

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader, cid, DEVICE, classes=10, open_set = False):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.cid = cid
        self.round = 0
        self.device = DEVICE
        self.classes = classes
        self.open_set = open_set

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
        train(self.net, self.trainloader, epochs=local_epochs, DEVICE=self.device, open_set = self.open_set)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy, f1, roc, kappa, precision, recall, weights = test(self.net, self.valloader, DEVICE=self.device, 
                                                                        classes=self.classes, open_set=self.open_set)

        return float(loss), len(self.valloader), {"accuracy": float(accuracy), 
                                                    "f1": f1.item(), 
                                                    "roc": roc.item(), 
                                                    "kappa": kappa.item(),
                                                    "precision": precision.item(),
                                                    "recall":recall.item(),
                                                    "weights": weights
                                                }


def weighted_average_resnet_CIFAR_open_set(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    # Metrics is a list of tuples (int, metrics), so c[n][1] is the nth metrics dict
    _,valdata,_ = partition_scripts.partition_CIFAR_IID(4, "CIFAR10")

    all_weights = [c[1]["weights"] for c in metrics]
    
    flat_labels = get_all_labels(valdata[0])
   
    outputs = {}
    for i, weight in enumerate(all_weights):
        current_outputs = get_FedOV_outputs(weight, valdata[0], i, outputs)

        
    results = get_votes(outputs)
    result_list = list(zip(results, flat_labels)) 
    save_tuples_to_csv(result_list, "res.csv")
    final_results = sum([1 if a[0] == a[1] else  0 for a in result_list])/len(flat_labels)

    # Aggregate and return custom metric (weighted average)
    return { #"metrics": metrics, 
                "outputs": final_results,
            }


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    f1 = [num_examples * m["f1"] for num_examples, m in metrics]
    kappa = [num_examples * m["kappa"] for num_examples, m in metrics]
    roc = [num_examples * m["roc"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples),
            "f1": sum(f1) / sum(examples),
            "kappa": sum(kappa) / sum(examples),
            "roc": sum(roc) / sum(examples),
            }

def get_all_labels(testdata):
    real_labels = [x[1] for x in testdata]
    flat_labels = []
    outputs = {}

    for r in real_labels:
        flat_labels += [t.item() for t in r]

    return flat_labels
 

def get_FedOV_outputs(weight, testdata, i, outputs):
    m = _resnet18(classes=11, shape=(32,32))
    set_parameters(m, weight)

    outputs[f"client_{i}"] = []
    for images, labels in testdata:
        current_outputs = m(images)
        outputs[f"client_{i}"].append(current_outputs)

    return outputs

def get_votes(outputs):
    results = []
    for n in range(len(list(outputs.values())[0])): #For the number of batches
        batch = np.zeros(outputs["client_0"][n].detach().numpy().shape)
        for client in outputs.values(): #for the clients
            batch += client[n].detach().numpy()

        for entry in batch:
            voting = np.argmax(entry[:-1]) #The last class is unknown, so we pick the maximum before this
            results.append(voting)

    return results


def fit_config(server_round: int):
    config = {
        "server_round": server_round,
        "local_epochs": 5,
    }
    return config

def display_parameters(arr):
    for i,layer in enumerate(arr):
        print(f"Layer {i}, shape {layer.shape}")



def save_tuples_to_csv(data, filename):
    """
    Saves a list of tuples to a CSV file.

    Parameters:
    data (list of tuples): The data to be saved to the CSV file.
    filename (str): The name of the CSV file.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

