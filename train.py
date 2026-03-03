import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# 1. Load Data (Assumes extracted_data/tracks.csv exists)
try:
    df = pd.read_csv('extracted_data/tracks.csv', encoding='latin1', on_bad_lines='skip')
except FileNotFoundError:
    print("Error: extracted_data/tracks.csv not found!")
    exit()

# 2. Logic to define Moods based on Valence (Happiness) and Energy
# 0: Energetic, 1: Happy, 2: Sad, 3: Calm
def assign_mood(row):
    if row['valence'] >= 0.5 and row['energy'] >= 0.5: return 0
    elif row['valence'] >= 0.5 and row['energy'] < 0.5: return 1
    elif row['valence'] < 0.5 and row['energy'] >= 0.5: return 2
    else: return 3

df['mood_id'] = df.apply(assign_mood, axis=1)
mood_map = {0: 'Energetic', 1: 'Happy', 2: 'Sad', 3: 'Calm'}

# 3. Features & Target
features = ['danceability', 'energy', 'loudness', 'valence', 'tempo']
X = df[features]
y = df['mood_id']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scaling (Crucial for audio data consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 6. Training the Decision Tree
model = DecisionTreeClassifier(max_depth=12, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Export the "Brain"
joblib.dump(model, "models/mood_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(mood_map, "models/mapping.pkl")

print(f"Success! Model trained with {len(X_train)} songs.")