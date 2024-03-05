import numpy as np
import partition_scripts
import random
import matplotlib.pyplot as plt
from collections import Counter

def simple_array_test():
    xs = np.array([random.choice([0,1,2,3]) for _ in range(500)]).reshape(250,2)

    ys = np.array([0 for _ in range(100)] + [1 for _ in range(100)] + [2 for _ in range(50)] )

    a = partition_scripts._dirilecht_partition_data(xs, ys, 8, 2)

    for k in a:
        data_y = k[1]
        print(Counter(data_y).most_common())


def dataloaders_IID_test(cifar="CIFAR10"):
    a = partition_scripts.partition_CIFAR_IID(num_clients=5, CIFAR_TYPE=cifar)
    print(next(iter(a[0][0]))[0].shape)

def dataloaders_nonIID_test(beta=0.5, cifar="CIFAR10"):
    a = partition_scripts.partition_CIFAR_nonIID(num_clients=5, CIFAR_TYPE=cifar, beta=beta)
    print(next(iter(a[0][0]))[0].shape)

    return show_dists(a)

def dataloaders_fedfaces_IID_test():
    a = partition_scripts.partition_FedFaces_IID(5)
    
    return show_dists(a)

def dataloaders_fedfaces_nonIID_test(beta=0.5):
    a = partition_scripts.partition_FedFaces_nonIID(num_clients=5, beta=beta)
    return show_dists(a)

def show_dists(a):
    counters = []
    for dlr in a[0]:
        c = Counter()
        itr, train_labels = get_labels(dlr)

        try:
            while True:
                c.update(train_labels.tolist())
                _, train_labels = next(itr)
        except StopIteration:
            pass

        print(c.most_common(), c.total())
        counters.append(c)

    return counters

def get_labels(dlr):
    itr = iter(dlr)
    _, train_labels = next(itr)
    return itr,train_labels


def create_stacked_bar_graph(counters):
    """
    Creates a stacked bar graph for a list of Python counters.

    Args:
        counters (list): List of Python counters (collections.Counter objects).

    Returns:
        None (displays the plot)
    """
    # Extract labels and values from counters
    print("--------------------------------")
    print(counters)
    print("--------------------------------")

    categories = []
    
    for presences in [list(c.keys()) for c in counters]:
        categories = categories + presences
    categories = [int(x) for x in categories]
    categories = list(set(categories))

    num_counters = len(counters)

    values = {}
    for category in categories:
        values[category] = [counter.get(category,0) for counter in counters]

    labels = [f"Client {i+1}" for i in range(num_counters)]

    # Create a stacked bar plot
    fig, ax = plt.subplots(figsize=(15, 6))
    #colors = plt.cm.get_cmap('tab20', num_counters)
    bottom = np.zeros(len(categories))

    bottom = np.zeros(num_counters)
    for category, value in values.items():
        ax.bar(labels, value, label=category, bottom = bottom)
        bottom += value

    # Add labels and legend
    plt.xlabel("Clients")
    plt.ylabel("Counts")
    plt.title("Stacked Bar Graph for Client Class Distribution")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Show the plot
    plt.show()


def graph_and_distribution_tests():
    print(10* "=", "scenarios" , 10*"=")
    print("simple_distribution")
    c = show_dists(partition_scripts.partition_CIFAR_IID(5, "CIFAR100"))
    create_stacked_bar_graph(c)
    print("CIFAR, nonIID")
    c2 = dataloaders_nonIID_test()
    create_stacked_bar_graph(c2)
    print("FedFaces, IID")
    c3 = dataloaders_fedfaces_IID_test()
    create_stacked_bar_graph(c3)
    print("FedFaces, nonIID")
    c4 = dataloaders_fedfaces_nonIID_test()
    create_stacked_bar_graph(c4)


def celeb_IID_test():
    c = show_dists(partition_scripts.partition_CelebA_IID(5))
    create_stacked_bar_graph(c)

def celeb_nonIID_test():
    c = show_dists(partition_scripts.partition_CelebA_nonIID(5))
    create_stacked_bar_graph(c)

celeb_nonIID_test()