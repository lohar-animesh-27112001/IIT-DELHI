import pandas as pd  # Importing pandas for data manipulation and analysis
import numpy as np  # Importing numpy for numerical operations with arrays
import ast  # Importing ast for safely evaluating string representations of Python literals
from sklearn.metrics import classification_report  # Importing classification_report for evaluating the model's performance

class LinearChainCRF:
    def __init__(self, tag_index_map, tag_support_map, data_count):
        """
        Initializes the CRF model with tag index mapping and tag support map.
        :param tag_index_map: Mapping of tags to indices.
        :param tag_support_map: Fraction of support for each tag.
        """
        self.data_count = data_count
        self.tag_index_map = tag_index_map  # Stores the mapping of tags to indices
        self.index_tag_map = {i: tag for tag, i in tag_index_map.items()}  # Creates a reverse mapping from indices to tags
        self.n_tags = len(tag_index_map)  # Determines the number of unique tags
        self.tag_support_map = tag_support_map  # Stores the support information for each tag

        # Initialize weights for features (for transitions) and observations (for emissions)
        self.feature_weights = np.random.rand(self.n_tags, self.n_tags) * 0.01  # Small random values for feature weights
        self.observation_weights = np.random.rand(len(self.tag_index_map), len(self.tag_index_map)) * 0.01  # Small random values for observation weights

        # Enhance weights for low-support tags based on tag_support_map
        for tag in self.tag_index_map:  # Iterate over each tag in the mapping
            if self.tag_support_map[tag] < 0.1 and self.tag_support_map[tag] > 0.05 :  # Check if the support for the tag is below 0.3
                self.observation_weights[self.tag_index_map[tag], :] += 1.1*data_count  # Boost their importance significantly
            elif self.tag_support_map[tag] <= 0.05 and self.tag_support_map[tag] > 0.01 :
                self.observation_weights[self.tag_index_map[tag], :] += 1.3*data_count
            elif self.tag_support_map[tag] <= 0.01 :
                self.observation_weights[self.tag_index_map[tag], :] += 1.5*data_count
            else:
                self.observation_weights[self.tag_index_map[tag], :] += 1
        dict_tag = {}
        for tag in self.tag_index_map:
            dict_tag[tag] = []
        self.dict_tag = dict_tag

    def fit(self, X, Y, num_iterations, lr):
        """
        Trains the CRF model on the provided data.
        :param X: List of input sequences (list of words).
        :param Y: List of corresponding tags for each sequence.
        :param num_iterations: Number of training iterations.
        :param lr: Learning rate for weight updates.
        """
        for x, y in zip(X, Y):
            for i in range(0, len(y)):
                arr = self.dict_tag[y[i]]
                arr.append(x[i])
                self.dict_tag[y[i]] = arr
        for iteration in range(num_iterations):  # Loop over the number of iterations for training
            total_updates = 0  # Initialize a counter for total updates
            for x, y in zip(X, Y):  # Iterate over each input sequence and its corresponding tags
                predicted = self.viterbi(x)  # Get the predicted tag sequence using Viterbi
                true_features = self.extract_features(x, y)  # Extract features from the true tags
                predicted_features = self.extract_features(x, predicted)  # Extract features from the predicted tags

                # Update weights based on differences in true and predicted features
                if true_features.shape == self.feature_weights.shape and predicted_features.shape == self.feature_weights.shape:
                    self.feature_weights += lr * (true_features - predicted_features)  # Adjust feature weights
                    total_updates += np.sum(true_features != predicted_features)  # Count the number of updates

            # Output the number of total updates for the current iteration
            print(f"Iteration {iteration + 1}/{num_iterations} - Total updates: {total_updates}")

    def extract_features(self, x, y):
        """Extracts feature counts for the given sequence x, y."""
        features = np.zeros((self.n_tags, self.n_tags))  # Initialize a feature count matrix
        n_words = len(x)  # Get the number of words in the sequence

        for i in range(n_words):  # Loop over each word in the sequence
            current_tag_idx = self.tag_index_map[y[i]]  # Get index of the current tag
            features[current_tag_idx, current_tag_idx] += 1  # Increment the emission feature count for the current tag

            if i > 0:  # If this is not the first word, compute transition features
                prev_tag_idx = self.tag_index_map[y[i - 1]]  # Get index of the previous tag
                features[prev_tag_idx, current_tag_idx] += 1  # Increment the transition feature count from previous to current tag

        return features  # Return the feature count matrix

    def compute_log_likelihood(self, X, Y):
        """Computes the log likelihood of the observed sequence given the weights."""
        log_likelihood = 0.0  # Initialize log likelihood
        for x, y in zip(X, Y):  # Loop over each input sequence and its true tags
            features = self.extract_features(x, y)  # Extract features for the true tags
            emission_scores = np.dot(features, self.observation_weights)  # Compute emission scores
            transition_scores = np.dot(features, self.feature_weights)  # Compute transition scores

            # Calculate log likelihood with numerical stability to avoid overflow
            max_score = np.max(emission_scores + transition_scores)  # Find the max score for stability
            log_likelihood += np.sum(emission_scores + transition_scores) - max_score \
                              - np.log(np.sum(np.exp(emission_scores + transition_scores - max_score)))  # Update log likelihood

        return log_likelihood  # Return the calculated log likelihood

    def viterbi(self, x):
        """Runs the Viterbi algorithm to decode the best tag sequence."""
        n_words = len(x)  # Get the number of words in the sequence
        viterbi = np.full((n_words, self.n_tags), -np.inf)  # Initialize the Viterbi table with -inf for log probabilities
        backpointer = np.zeros((n_words, self.n_tags), dtype=int)  # Initialize backpointer table

        # Initialize first step of Viterbi algorithm
        for tag, idx in self.tag_index_map.items():  # Iterate over each tag and its index
            if(self.tag_support_map[tag] < 0.1 and self.tag_support_map[tag] > 0.05):  # If the tag is underrepresented
                viterbi[0, idx] = 0.00001*self.data_count  # Start with a high log probability for underrepresented tags
            elif(self.tag_support_map[tag] <= 0.05 and self.tag_support_map[tag] > 0.01):
                viterbi[0, idx] = 0.00002*self.data_count
            elif(self.tag_support_map[tag] <= 0.01):
                viterbi[0, idx] = .000003*self.data_count
            else:
                viterbi[0, idx] = 0  # Start with neutral log probability for the first word

        # Dynamic programming step to calculate Viterbi path
        for t in range(1, n_words):  # Loop over each word starting from the second
            for curr_tag, curr_index in self.tag_index_map.items():  # Iterate over each current tag
                max_prob, best_prev_tag = float('-inf'), -1  # Initialize for finding the best previous tag
                for prev_tag, prev_index in self.tag_index_map.items():  # Check each previous tag
                    score = viterbi[t - 1, prev_index] + self.feature_weights[prev_index, curr_index]  # Compute the score
                    if score > max_prob:  # If this score is better than the previous max
                        max_prob, best_prev_tag = score, prev_index  # Update max probability and best previous tag
                viterbi[t, curr_index] = max_prob  # Store the best score for the current tag
                backpointer[t, curr_index] = best_prev_tag  # Record the best previous tag

        # Backtrack to find the best path
        best_last_tag = np.argmax(viterbi[n_words - 1])  # Get index of the best last tag
        best_path = [best_last_tag]  # Initialize best path with the last tag

        for t in range(n_words - 1, 0, -1):  # Backtrack from the last word to the first
            best_last_tag = backpointer[t, best_last_tag]  # Move to the best previous tag
            best_path.insert(0, best_last_tag)  # Insert at the beginning of the path

        return [self.index_tag_map[i] for i in best_path]  # Convert indices back to tags and return

    def predict(self, X):
        """Predicts the sequences for the input data."""
        arr = [self.viterbi(x) for x in X]  # Apply Viterbi for each input sequence
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                a = {key: 0 for key in self.dict_tag}  # Initialize counts in a for each key
                for key in self.dict_tag:
                    if arr[i][j] in self.dict_tag[key]:
                        a[key] += 1  # Increment count for each tag that matches
                # Check if a has any counts
                present = False
                for key in a:
                    if a[key] != 0:
                        present = True
                        break
                if a[arr[i][j]] == 0 and present == True:
                    max_key = max(a, key=a.get)  # Find the key with the max count
                    arr[i][j] = max_key  # Update the prediction with the key with max count
        output = []
        for i in range(len(arr)):
            sentence_result = []
            for j in range(len(arr[i])):
                sentence_result.append((X[i][j], arr[i][j]))
            output.append(sentence_result)
        return arr, output

    @staticmethod
    def load_data(file_path, frac):
        """Loads and processes the data from the given CSV file."""
        try:
            data = pd.read_csv(file_path)  # Load the data
            data = data.sample(frac=frac).reset_index(drop=True)  # Shuffle and reset index of the dataset
            print(f"{file_path} loaded successfully.")  # Indicate successful data loading
            return data  # Return the loaded data
        except FileNotFoundError:
            print(f"Error: '{file_path}' not found.")  # Handle file not found error
            exit()  # Exit the program

    @staticmethod
    def parse_tags(tags):
        """Safely parses the tags from the dataset."""
        try:
            return ast.literal_eval(tags) if isinstance(tags, str) else tags  # Safely evaluate string representations
        except (SyntaxError, ValueError):
            print(f"Warning: Skipping malformed tag data: {tags}")  # Handle malformed tag data
            return []  # Return an empty list for malformed data

    @staticmethod
    def preprocess_data(file_path, frac):
        """Preprocesses the data from the file path, extracting sentences and tags."""
        data = LinearChainCRF.load_data(file_path, frac)  # Load the data using previously defined method
        X = data['Sentence'].apply(lambda s: s.split()).tolist()  # Split sentences into words
        Y = data['Tag'].apply(LinearChainCRF.parse_tags).tolist()  # Parse tags into lists using parsing function

        # Trim mismatched sentence-tag pairs to ensure they align properly
        for i in range(len(X)):  # Loop through each sentence
            if len(X[i]) != len(Y[i]):  # Check if lengths mismatch
                print(f"Warning: Sentence {i} length mismatch. Trimming to match.")  # Warn about the mismatch
                min_length = min(len(X[i]), len(Y[i]))  # Determine minimum length
                X[i], Y[i] = X[i][:min_length], Y[i][:min_length]  # Trim the longer one

        return X, Y  # Return processed input sentences and tags

    @staticmethod
    def calculate_tag_support(Y):
        """Calculates tag support based on frequency."""
        tag_counts = {}  # Dictionary to hold tag counts
        for tags in Y:  # Loop through all tag lists
            for tag in tags:  # Count each tag's occurrences
                tag_counts[tag] = tag_counts.get(tag, 0) + 1  # Increment count for each tag
        total_tags = sum(tag_counts.values())  # Calculate total number of tags
        return {tag: count / total_tags for tag, count in tag_counts.items()}  # Return fraction of support for each tag