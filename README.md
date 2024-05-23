## Word Relationship Prediction using NetworkX and RNNs

This repository explores the relationships between words using network graph analysis and builds a Recurrent Neural Network (RNN) model to predict the next word in a sequence.

### Dependencies

This code requires the following Python libraries:

* urllib.request
* gzip
* os
* string
* matplotlib.pyplot
* networkx
* numpy
* seaborn
* sklearn
* tensorflow.keras

###  Code Structure

The code is divided into several sections:

1. **Libraries:** Imports necessary libraries.
2. **Download the file if it doesn't exist:** Checks for the data file and downloads it if missing.
3. **Generating the Word Graph:**
   - Defines functions to create a graph where nodes represent words and edges represent a one-letter difference between words.
   - Calculates various properties of the graph including number of connected components, node degrees, shortest path lengths, and betweenness centrality.
4. **Show shortest paths between word pairs:**
   - Finds the shortest path between provided word pairs and visualizes the subgraph containing those words.
   - Calculates graph density.
5. **Create word sequences from the shortest paths between pairs of words:**
   - Generates sequences of words from the shortest paths between predefined pairs.
6. **Preprocess data for RNN model:**
   - Converts words into numerical indices using a `LabelEncoder`.
   - Reshapes the input data for the RNN model.
7. **Create the RNN model for missing word prediction:**
   - Defines a Sequential model with an embedding layer, an LSTM layer, and a dense output layer with softmax activation.
8. **Compile and Train the model:**
   - Compiles the model using the Adam optimizer, sparse categorical crossentropy loss, and accuracy metrics.
   - Trains the model on the generated sequences with validation split.
9. **Evaluate the model:**
   - Calculates accuracy, confusion matrix, precision, recall, and F1-score on the validation set.
   - Visualizes the confusion matrix and plots precision, recall, and F1-score.
10. **Save and Load the Model:**
   - Saves the trained model as "word_rnn_model.h5".
   - Loads the saved model for demonstration purposes.
