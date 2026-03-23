import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import argparse
from collections import Counter

# Disable OneDNN optimizations for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main(args):
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"Error: Data file '{args.data_path}' not found.")
        print("Please run the create_dataset.py script first to generate the data.")
        return

    # Load the data
    try:
        with open(args.data_path, 'rb') as f:
            data_dict = pickle.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    data = np.asarray(data_dict.get('data'))
    labels = np.asarray(data_dict.get('labels'))

    if data.size == 0 or labels.size == 0:
        print("Error: Dataset is empty. Please regenerate with create_dataset.py")
        return

    if len(data) != len(labels):
        print("Error: Data and labels length mismatch in dataset file.")
        return

    feature_size = int(data_dict.get('feature_size', data.shape[1]))

    # Split the data into training and test sets
    label_counts = Counter(labels)
    min_class_count = min(label_counts.values())
    stratify_labels = labels if min_class_count >= 2 else None

    if stratify_labels is None:
        print("Warning: Some classes have fewer than 2 samples; training split will not use stratification.")

    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=args.test_size,
        shuffle=True,
        stratify=stratify_labels,
        random_state=42
    )

    # Initialize the model with specified hyperparameters
    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)

    # Train the model
    model.fit(x_train, y_train)

    # Cross-validation for better performance estimation
    if args.cross_validate:
        cv_folds = min(5, min_class_count)
        if cv_folds >= 2:
            cv_scores = cross_val_score(model, x_train, y_train, cv=cv_folds)
            print(f'Cross-validation accuracy ({cv_folds}-fold): {cv_scores.mean() * 100:.2f}%')
        else:
            print('Skipping cross-validation: not enough samples per class.')

    # Test the model and print the accuracy
    y_predict = model.predict(x_test)
    score = accuracy_score(y_test, y_predict)
    print(f'{score * 100:.2f}% of samples were classified correctly!')

    # Save the model with metadata
    model_metadata = {
        'model': model,
        'accuracy': score,
        'classes': model.classes_.tolist(),
        'feature_size': feature_size,
        'hyperparameters': {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth
        }
    }

    with open(args.save_path, 'wb') as f:
        pickle.dump(model_metadata, f)
    
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a RandomForest classifier on hand gesture data.")
    parser.add_argument('--data_path', type=str, default='./data.pickle', help='Path to the data pickle file.')
    parser.add_argument('--save_path', type=str, default='model.p', help='Path to save the trained model.')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the forest.')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of the tree.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data to use for testing.')
    parser.add_argument('--cross_validate', action='store_true', help='Whether to perform cross-validation.')

    args = parser.parse_args()
    main(args)
