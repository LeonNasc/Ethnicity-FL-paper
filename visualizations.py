import pandas as pd
import matplotlib.pyplot as plt

def plot_loss_accuracy(csv_file, experiment):
    # Read CSV file into a pandas DataFrame
    if csv_file[:-3] == "csv":
        data = pd.read_csv(csv_file)
        data = data[data["Experiment"].str.contains(experiment)]
    else:
        data = pd.DataFrame(parse_epochs_from_file(csv_file, experiment))
        data = data[data["Experiment"].str.contains(experiment)]
    
    # Extract data
    epochs = data['Epoch']
    loss = data['Loss']
    accuracy = data['Accuracy']
    
    # Plotting loss and accuracy on the same graph with different y-axes
    fig, ax1 = plt.subplots(figsize=(8, 8))
    
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_xticks(range(0, len(epochs), 50))
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, loss, color=color, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title('Loss and Accuracy over Epochs')
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_xticks(range(0, len(epochs), 50))
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, accuracy, color=color, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0,1)
    
    fig.tight_layout()
    plt.show()

def parse_epochs_from_file(file_path, exp_name):
    """
    Parses epoch information from a text file and converts it to a CSV format.

    Args:
        file_path (str): Path to the input text file.

    Returns:
        str: Equivalent CSV representation of the epochs.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            columns = ["Experiment","Epoch","Loss","Accuracy"]
            arr = []

            for line in lines:
                line = line.replace(":", "")
                parts = line.split(",")
                epoch_num = parts[0].strip().split(" ")[-1]
                loss = float(parts[1].split(" ")[-1])
                accuracy = float(parts[2].split(" ")[-1])
                arr.append((exp_name, epoch_num, loss, accuracy))

            df = pd.DataFrame(arr, columns=columns)

            return df
    except FileNotFoundError:
        return "Error: File not found. Please provide a valid file path."


#csv_file = 'results_centralized.csv'
#plot_loss_accuracy(csv_file, "CIFAR 10 ")


csv_file = './Training Results/CelebA Centralized.txt'
plot_loss_accuracy(csv_file, "CelebA")
