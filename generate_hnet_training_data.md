# Overview

The `generate_hnet_training_data.py` script is a comprehensive data generation tool designed to create synthetic datasets for training and testing a neural network model named HNetGRU. This model is aimed at associating directions of arrival (DOAs) in various applications, such as signal processing or audio source localization. Here's a detailed, non-technical overview of how the script functions and its intended purpose:

## Purpose of the Script

The primary objective of this script is to generate synthetic training and testing data that the HNetGRU model can learn from and be evaluated against. By creating a diverse set of data points with known associations, the script ensures that the model is well-equipped to recognize and predict associations in real-world scenarios. This synthetic data aids in training the model to accurately map predicted DOAs to their corresponding reference DOAs, enhancing its predictive capabilities.

## Key Functionalities

### Setting Up the Environment and Parameters

- **Libraries and Functions**: The script utilizes essential Python libraries such as NumPy for numerical operations, SciPy for advanced mathematical computations, and Pickle for data serialization. It also defines utility functions like `sph2cart` to convert spherical coordinates (azimuth and elevation angles) to Cartesian coordinates, which are easier for the model to process.
- **Configuration Parameters**: Key parameters like `max_doas` (maximum number of DOAs) and `sample_range` (number of samples to generate for each DOA combination) are initialized. These parameters dictate the breadth and depth of the generated dataset.
`max_doas` is used to specify the number of sound sources the Hungarian Net should be able to process (with no regards with how well the _main_ neural network is able to predict DoAs from multiple sources).
`sample_range` is used to specify how many samples should be generated for each combination of reference and predicted DOAs. This parameter helps in controlling the size of the dataset and the diversity of the training and testing data.

### Generating Training and Testing Data

- **Iterating Over Angular Resolutions**: The script loops through various angular resolutions (e.g., 1°, 2°, up to 30°) to create data at different levels of granularity. This ensures that the model is exposed to data with varying directions of precision.
- **Combining Reference and Predicted DOAs**: For each resolution, the script varies the number of reference DOAs (`nb_ref`) and predicted DOAs (`nb_pred`). It systematically generates combinations where the number of reference and predicted DOAs ranges from 0 up to the defined maximum (`max_doas`). This exhaustive approach ensures that the model encounters a wide array of association scenarios during training.

### Creating Synthetic DOA Data

- **Random Angle Generation**: For each combination of reference and predicted DOAs, the script randomly samples azimuth and elevation angles within specified ranges. These angles represent the directional beams in applications like audio source localization.
- **Handling Edge Cases**: To maintain data consistency and prevent unrealistic scenarios, the script initializes some DOAs to a fixed value (e.g., 10) under certain conditions. This strategy helps the model handle cases where predictions might initially be random or inaccurate, promoting robustness in learning.

### Converting to Cartesian Coordinates

- **Spherical to Cartesian Conversion**: The randomly generated spherical coordinates (azimuth and elevation) are converted into Cartesian coordinates using the `sph2cart` function. This transformation simplifies the computation of distances between different DOAs, making it easier for the model to process spatial relationships.

### Computing Distance and Association Matrices

- **Distance Matrix Calculation**: The script computes a distance matrix using the Minkowski distance metric (specifically, the Euclidean distance with p=2). This matrix quantifies the spatial distances between every pair of reference and predicted DOAs.
- **Optimal Association via Hungarian Algorithm**: To establish the most accurate associations between reference and predicted DOAs, the script employs the Hungarian Algorithm (implemented via `linear_sum_assignment`). This algorithm finds the optimal pairing that minimizes the total distance, ensuring that each predicted DOA is matched to the most appropriate reference DOA.

### Shuffling for Data Diversity

- **Randomizing Data Order**: To prevent the model from learning any spurious patterns based on the order of the data, the script randomly shuffles the rows or columns of both the distance matrix and the association matrix. This randomization enhances the model's ability to generalize by ensuring that it doesn't become biased towards any specific ordering of DOAs.

### Storing and Saving the Data

- **Data Structuring**: For each generated sample, the script stores relevant information, including the number of reference and predicted DOAs, the shuffled distance and association matrices, and the Cartesian coordinates of both reference and predicted DOAs.
- **Serialization and Saving**: Once a substantial number of samples are generated, the script serializes the data using Pickle and saves it to designated files (`hung_data_train.pkl` for training and `hung_data_test.pkl` for testing). These files serve as the input datasets for training and evaluating the HNetGRU model.

### Output

Supervised training of HungarianNet is available thanks to (feat, labels) = (matrix distance D, association matrix A*) where A* is obtained deterministically using the Hungarian algorithm on the distance matrices D.