import numpy as np
import itertools
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


def dataloaders_nonIID_test():
    a = partition_scripts.partition_CIFAR_nonIID(num_clients=5, CIFAR_TYPE="CIFAR10", beta=0.5)

    return show_dists(a)

def dataloaders_fedfaces_IID_test():
    a = partition_scripts.partition_FedFaces_equally(5)
    
    return show_dists(a)

def dataloaders_fedfaces_nonIID_test():
    a = partition_scripts.partition_FedFaces_nonIID(num_clients=5, beta=0.5)
    return show_dists(a)

def show_dists(a):
    counters = []
    for dlr in a[0]:
        c = Counter()
        itr = iter(dlr)
        _, train_labels = next(itr)

        try:
            while True:
                c.update(train_labels.tolist())
                _, train_labels = next(itr)
        except StopIteration:
            pass

        print(c.most_common(), c.total())
        counters.append(c)

    return counters


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

    labels = list(itertools.chain([list(c.keys()) for c in counters]))
    print(labels)
    #labels = list(set(np.concatenate([c.keys() for c in counters])))
    num_counters = len(counters)
    values = np.array([list(counter.values()) for counter in counters])

    # Create a stacked bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.get_cmap('tab20', num_counters)
    bottom = np.zeros(len(labels))

    for i, counter in enumerate(counters):
        ax.bar(labels, values[i], bottom=bottom, label=f"Counter {i + 1}", color=colors(i))
        bottom += values[i]

    # Add labels and legend
    plt.xlabel("Categories")
    plt.ylabel("Counts")
    plt.title("Stacked Bar Graph for Python Counters")
    plt.legend()

    # Show the plot
    plt.show()

print(10* "=", "scenarios" , 10*"=")
print("simple_distribution")
c = show_dists(partition_scripts.partition_CIFAR_equally(5))
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

