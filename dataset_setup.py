import tarfile, os

contents = os.listdir("./Datasets/raw/")

if len(contents) < 4: #If there are more than 3 items in Datasets/raw, the files already have been extracted

    raw_datasets = [x for x in contents  if ".tar.gz" in x]

    for dataset in raw_datasets:
        with tarfile.open(f'./Datasets/raw/{dataset}', 'r:gz') as tar:
            path = f'./Datasets/raw/{dataset.split(".")[0]}'
            
            if not os.path.exists(path):
                os.mkdir(path)

            tar.extractall(path)

        print(f'{dataset} extracted to Datasets/raw')

else:
    print("Files already have been extracted")