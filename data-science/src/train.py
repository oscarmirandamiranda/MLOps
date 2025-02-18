# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import mlflow
import mlflow.sklearn

def parse_args():
    '''Parse input arguments'''
    
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--n_estimators", type=int, default=100, 
                        help="The number of trees in the forest (default: 100)")
    parser.add_argument("--max_depth", type=int, default=None, 
                        help="The maximum depth of the tree. If None, nodes are expanded until all leaves are pure.")
    parser.add_argument("--model_output", type=str, help="Path to save the trained model")
   
    args = parser.parse_args()
    
    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''
    
    # Load datasets
    train_df = pd.read_csv(Path(args.train_data)/"train.csv")
    test_df = pd.read_csv(Path(args.test_data)/"test.csv")

    # Separate features and target variable for training and testing
    y_train = train_df['price']
    X_train = train_df.drop(columns=['price'])
    y_test = test_df['price']
    X_test = test_df.drop(columns=['price'])
    
    # Initialize and train a Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
    rf_model = rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    # Log model hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Compute regression metrics
    mse = mean_squared_error(y_test, rf_predictions)
    r2 = r2_score(y_test, rf_predictions)
    
    # Log metrics
    print(f'Mean Squared Error (MSE): {mse:.2f}')
    print(f'R² Score: {r2:.2f}')
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)

    # Output the trained model
    mlflow.sklearn.save_model(sk_model=rf_model, path=args.model_output)

if __name__ == "__main__":
    
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()

