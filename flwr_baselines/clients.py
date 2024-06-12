"""Defines the client class and support functions for SCAFFOLD."""

import os
from typing import Dict,OrderedDict

import flwr as fl
import torch
from flwr.common import Scalar
from torch.utils.data import DataLoader
from flwr_baselines.models import test, train_scaffold, train_fednova


# pylint: disable=too-many-instance-attributes
class FlowerClientFedNova(fl.client.NumPyClient):
    """Flower client implementing FedNova."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
    ) -> None:
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set the local model parameters using given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Scalar]):
        """Implement distributed fit function for a given client for FedNova."""
        self.set_parameters(parameters)
        a_i, g_i = train_fednova(
            self.net,
            self.trainloader,
            self.device,
            self.num_epochs,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
        )
        # final_p_np = self.get_parameters({})
        g_i_np = [param.cpu().numpy() for param in g_i]
        return g_i_np, len(self.trainloader.dataset), {"a_i": a_i}

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters."""
        self.set_parameters(parameters)
        loss, acc = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc)}


class FlowerClientScaffold(fl.client.NumPyClient):
    """Flower client implementing scaffold."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        cid: int,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        save_dir: str = "",
    ) -> None:
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        # initialize client control variate with 0 and shape of the network parameters
        self.client_cv = []
        for param in self.net.parameters():
            self.client_cv.append(torch.zeros(param.shape))
        # save cv to directory
        if save_dir == "":
            save_dir = "client_cvs"
        self.dir = save_dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set the local model parameters using given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config: Dict[str, Scalar]):
        """Implement distributed fit function for a given client for SCAFFOLD."""
        # the first half are model parameters and the second are the server_cv
        server_cv = parameters[len(parameters) // 2 :]
        parameters = parameters[: len(parameters) // 2]
        self.set_parameters(parameters)
        self.client_cv = []
        for param in self.net.parameters():
            self.client_cv.append(param.clone().detach())
        # load client control variate
        if os.path.exists(f"{self.dir}/client_cv_{self.cid}.pt"):
            self.client_cv = torch.load(f"{self.dir}/client_cv_{self.cid}.pt")
        # convert the server control variate to a list of tensors
        server_cv = [torch.Tensor(cv) for cv in server_cv]
        train_scaffold(
            self.net,
            self.trainloader,
            self.device,
            self.num_epochs,
            self.learning_rate,
            self.momentum,
            self.weight_decay,
            server_cv,
            self.client_cv,
        )
        x = parameters
        y_i = self.get_parameters(config={})
        c_i_n = []
        server_update_x = []
        server_update_c = []
        # update client control variate c_i_1 = c_i - c + 1/eta*K (x - y_i)
        for c_i_j, c_j, x_j, y_i_j in zip(self.client_cv, server_cv, x, y_i):
            c_i_n.append(
                c_i_j
                - c_j
                + (1.0 / (self.learning_rate * self.num_epochs * len(self.trainloader)))
                * (x_j - y_i_j)
            )
            # y_i - x, c_i_n - c_i for the server
            server_update_x.append((y_i_j - x_j))
            server_update_c.append((c_i_n[-1] - c_i_j).cpu().numpy())
        self.client_cv = c_i_n
        torch.save(self.client_cv, f"{self.dir}/client_cv_{self.cid}.pt")

        combined_updates = server_update_x + server_update_c

        return (
            combined_updates,
            len(self.trainloader.dataset),
            {},
        )

    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters."""
        self.set_parameters(parameters)
        loss, acc = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc)}