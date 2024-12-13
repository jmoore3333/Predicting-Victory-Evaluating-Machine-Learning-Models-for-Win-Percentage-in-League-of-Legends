import pandas as pd
import json
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import joblib
import numpy as np

# Load credentials from the JSON file
with open('credentials.json') as f:
    credentials = json.load(f)

api_key = credentials['apiKey']
db_url = credentials['db_url']

# --- Adjustable Parameters for MLPClassifier ---
RANDOM_STATE = 42
TEST_SIZE = 0.3
HIDDEN_LAYER_SIZES = (100, 100, 100)  # Single hidden layer with 100 neurons
ACTIVATION = 'tanh'          # Activation function
SOLVER = 'adam'              # Optimizer
ALPHA = 0.001               # L2 penalty (regularization term)
BATCH_SIZE = 'auto'          # Size of minibatches
LEARNING_RATE = 'constant'   # Learning rate schedule
LEARNING_RATE_INIT = 0.0001   # Initial learning rate
MAX_ITER = 1000               # Maximum number of iterations
TOL = 1e-4                   # Tolerance for stopping criteria

# Function to fetch data from PostgreSQL database
def fetch_data_from_db():
    data = None
    try:
        engine = create_engine(db_url)
        query = "SELECT winning_team, metrics FROM match_results"
        data = pd.read_sql_query(query, engine)
    except Exception as e:
        print(f"Error fetching data from database: {e}")
    return data

# Function to preprocess the data
def preprocess_data(data):
    processed_data = []
    labels = []

    for index, row in data.iterrows():
        metrics = json.loads(row['metrics']) if isinstance(row['metrics'], str) else row['metrics']
        for frame in metrics:
            features = {
                'timestamp': frame['timestamp'],
                'team1_gold': frame['team1_gold'],
                'team2_gold': frame['team2_gold'],
                'gold_difference': frame['gold_difference'],
                'team1_xp': frame['team1_xp'],
                'team2_xp': frame['team2_xp'],
                'xp_difference': frame['xp_difference'],
                'team1_players_alive': frame['team1_players_alive'],
                'team2_players_alive': frame['team2_players_alive'],
                'team1_tower_kills': frame['team1_tower_kills'],
                'team2_tower_kills': frame['team2_tower_kills'],
                'team1_dragon_kills': frame['team1_dragon_kills'],
                'team2_dragon_kills': frame['team2_dragon_kills'],
                'dragon_kill_difference': frame['dragon_kill_difference'],
                'team1_barons': frame['team1_barons'],
                'team2_barons': frame['team2_barons'],
                'team1_elders': frame['team1_elders'],
                'team2_elders': frame['team2_elders'],
                'team1_grubs': frame['team1_grubs'],
                'team2_grubs': frame['team2_grubs'],
                'team1_rift_herald': frame['team1_rift_herald'],
                'team2_rift_herald': frame['team2_rift_herald'],
            }
            processed_data.append(features)
            labels.append(1 if row['winning_team'] == 100 else 0)

    df = pd.DataFrame(processed_data)
    df['winning_team'] = labels
    return df

# Fetching and preprocessing data
data = fetch_data_from_db()
processed_data = preprocess_data(data)

# Split the data into features and labels
X = processed_data.drop(columns=['winning_team', 'team1_gold', 'team2_gold', 'team1_players_alive', 'team2_players_alive'])
y = processed_data['winning_team']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'MLP_winrate_scaler.pkl')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
model = MLPClassifier(random_state=42)

# Set the parameter grid for hyperparameter tuning, including early stopping parameters
param_grid = {
    'hidden_layer_sizes': [(50, 50), (100, 100, 100), (150, 150), (200,)],
    'activation': ['relu', 'tanh', 'logistic', 'identity'],  # Added 'identity'
    'alpha': [0.0001, 0.001, 0.01, 0.1],  # Adjusted range of alpha
    'batch_size': [32, 64, 128, 'auto'],  # Including 'auto'
    'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],  # Adjusted learning rate range
    'max_iter': [1000, 2000, 3000],  # Increased max iterations for convergence
    'solver': ['adam', 'sgd', 'lbfgs', 'newton-cg'],  # Added 'lbfgs' and 'newton-cg'
    'early_stopping': [True],  # Enable early stopping
    'validation_fraction': [0.1, 0.2],  # Fraction of training data to use for validation
    'n_iter_no_change': [5, 10],  # Number of iterations with no improvement to stop
}

# Perform grid search to find the best combination of hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Print the best hyperparameters and the best cross-validation score
print("Best hyperparameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Use the best model from the grid search
best_model = grid_search.best_estimator_

# Predict the probabilities on the test set
y_pred_proba = best_model.predict_proba(X_test_scaled)

# Calculate the Brier score for the model
brier_score = brier_score_loss(y_test, y_pred_proba[:, 1])  # Assuming binary classification
print("Brier score:", brier_score)

# Optionally, visualize or check predictions (optional)
# predicted_classes = best_model.predict(X_test_scaled)
# print("Predicted classes:", predicted_classes)
