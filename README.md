This repository corresponds to my thesis on deforestation prediction using machine learning. A few files in particular are of importance:
- Resnet_CNN.py: The programmed ResNet50 model that was trained.
- LoadDataset.py: Loading the dataset, exporting results to neptune.io, model hyperparameters, executing the experiment runs.
- labels.csv: Regressive deforestation labels for each instance of the training and test dataset.
- LoadDatasetGIS.ipynb: A work-in-progress modification of the ResNet model that would take in GIS data through additional image channels.
- JobScript_Lisa.sh: File used to interact with the server, not significant for the thesis itself.
