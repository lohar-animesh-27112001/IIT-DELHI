from Linear_ChainCRF import LinearChainCRF
from sklearn.metrics import classification_report

# Usage example
if __name__ == "__main__":
    # Load and preprocess the training data
    X_train, Y_train = LinearChainCRF.preprocess_data('ner_train.csv', frac = 1)  # Preprocess training data
    tag_counts = LinearChainCRF.calculate_tag_support(Y_train)  # Calculate tag support for training data

    # Create tag index mapping
    unique_tags = sorted(set(tag for tags in Y_train for tag in tags))  # Get all unique tags
    tag_index_map = {tag: i for i, tag in enumerate(unique_tags)}  # Create mapping from tags to indices

    # Initialize and train the model
    count_data = len(X_train)
    model = LinearChainCRF(tag_index_map, tag_counts, count_data)  # Create instance of LinearChainCRF
    print("Starting training...")  # Indicate training start
    model.fit(X_train, Y_train, num_iterations = 10, lr = 0.01)  # Train the model on the data
    print("Training completed.")  # Indicate training completion

    # Load and preprocess test data
    X_test, Y_test = LinearChainCRF.preprocess_data('ner_test.csv', frac = 1)  # Preprocess test data

    # Make predictions
    # X_test = [['Barack', 'Obama', 'visited', 'India', 'last', 'week']]
    # Y_test = [['B-PER', 'I-PER', 'O', 'B-LOC', 'O', 'O']]
    Y_pred, output = model.predict([['Barack', 'Obama', 'visited', 'India', 'last', 'week']])
    print(output)
    
    Y_pred, output = model.predict(X_test)  # Get predictions for test data
    print("Predictions made.")  # Indicate predictions are ready

    # Flatten lists for evaluation
    Y_pred_flat = [tag for sentence in Y_pred for tag in sentence]  # Flatten predicted tags
    Y_test_flat = [tag for sentence in Y_test for tag in sentence]  # Flatten true tags

    # Evaluate performance
    print("Evaluating performance...")  # Indicate start of evaluation
    print(classification_report(Y_test_flat, Y_pred_flat, zero_division=0))  # Print classification report for evaluation