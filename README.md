# Character-Level Text Generation RNN for Species Names: README

This project focuses on training and evaluating a character-level RNN for generating species names. Below are instructions and descriptions of the software environment required.

## Training the Model

To train the model, execute `main_train.py`. This script handles the training process and includes several options:

- **NEW_TRAIN**: Set `NEW_TRAIN=True` to initiate a new training instance. Set it to `False` to continue training from a previous checkpoint.
- **Checkpoints**: The training checkpoints for each epoch will be automatically saved.
- **Google Colab Support**: If using Google Colab, set `GOOGLE_COLAB = True` and mount it to the directory containing the project's files. If not using Google Colab, set it to `False`, and the current working directory will be automatically used.
- **Training Loss Record**: The script records training loss in a CSV file and saves a visualization of training loss vs. epochs as a PNG file.

## Testing the Model

To test the trained model, run `main_test.py`. This script loads the best saved checkpoint and generates outputs for:

- Species names completion.
- New names generation.

The outputs of these experiments are saved as separate files for further analysis.

## Analyzing Results

After obtaining the output files, you can run `results_analysis.py` to analyze the results. This script provides:

- Overall performance metrics.
- Performance analysis vs. input lengths.
- Performance analysis vs. occurrences, presented as visualizations.

## Software Environment

The following software environment is required for running the scripts:

- **Python**: Version 3.9.12
- **TensorFlow**: Version 2.9.1
- **Additional Libraries**: 
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - Unidecode
- **TensorFlow Keras API**: Utilized for model training and evaluation.

Ensure that all the required software is installed and up-to-date to avoid any compatibility issues during execution of the scripts.
