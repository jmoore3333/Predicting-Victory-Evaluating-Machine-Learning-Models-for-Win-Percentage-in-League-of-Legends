import pandas as pd
import json
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
import joblib
import numpy as np

# Load credentials from the JSON file
with open('credentials.json') as f:
    credentials = json.load(f)

api_key = credentials['apiKey']
db_url = credentials['db_url']

# --- Adjustable Parameters for SVR ---
RANDOM_STATE = 42
TEST_SIZE = 0.3
KERNEL = 'rbf'          # Kernel type for SVR
C = 1000.0                 # Regularization parameter
GAMMA = 'scale'         # Kernel coefficient

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
joblib.dump(scaler, 'SVR_winrate_scaler.pkl')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Initialize the SVR model with adjustable parameters
model = SVR(kernel=KERNEL, C=C, gamma=GAMMA)
model.fit(X_train, y_train)
joblib.dump(model, 'SVR_winrate_model.pkl')

# Evaluate the model
win_probabilities_test = model.predict(X_test)
win_probabilities_test = np.clip(win_probabilities_test, 0, 1)

# Calculate the Brier score
brier_score_test = brier_score_loss(y_test, win_probabilities_test)
print(f"Brier score on test data: {brier_score_test:.4f}")


# Optional: Display predicted values and actual labels for test data points
test_data_with_predictions = pd.DataFrame(X_test, columns=X.columns)
test_data_with_predictions['predicted_win_probability'] = win_probabilities_test
test_data_with_predictions['actual_winning_team'] = y_test.reset_index(drop=True)
print(test_data_with_predictions[['predicted_win_probability', 'actual_winning_team']].head())
