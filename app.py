import pandas as pd
import numpy as np
matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")
matches['date'] = pd.to_datetime(matches['date'])
matches = matches.sort_values('date')
innings_runs = (
    deliveries
    .groupby(['match_id', 'inning'])['total_runs']
    .sum()
    .reset_index()
)
first_innings = innings_runs[innings_runs['inning'] == 1].copy()
first_innings['target'] = first_innings['total_runs'] + 1
first_innings = first_innings[['match_id', 'target']]
df = matches.merge(first_innings, left_on='id', right_on='match_id')
df['chasing_team'] = np.where(
    df['toss_winner'] == df['team1'],
    np.where(df['toss_decision'] == 'bat', df['team2'], df['team1']),
    np.where(df['toss_decision'] == 'bat', df['team1'], df['team2'])
)
df['chasing_win'] = (df['winner'] == df['chasing_team']).astype(int)
df = df.sort_values('date')
df['prev_avg_target'] = (
    df.groupby('venue')['target']
      .transform(lambda x: x.shift().expanding().mean())
)
df = df.dropna(subset=['prev_avg_target'])
df['relative_target'] = df['target'] / df['prev_avg_target']
avg_winning_score = df[df['chasing_win'] == 1].groupby('venue')['target'].mean()
print("\n=== AVG WINNING CHASE TARGET PER STADIUM ===")
print(avg_winning_score)
df_model = df.copy()
df_model = pd.get_dummies(df_model, columns=['venue'], drop_first=True)
X = df_model[['relative_target'] + [c for c in df_model.columns if c.startswith('venue_')]]
y = df_model['chasing_win']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
print("\nModel Accuracy:", accuracy_score(y_test, model.predict(X_test)))
def predict_chasing_win(venue, target):
    venue = venue.lower()
    matches = df[df['venue'].str.lower().str.contains(venue)]
    if len(matches) == 0:
        print("Venue not found")
        print("Available venues:")
        print(df['venue'].unique())
        return

    selected_venue = matches['venue'].iloc[0]

    venue_data = df[df['venue'] == selected_venue]

    prev_avg = venue_data['target'].mean()
    relative_target = target / prev_avg

    input_data = pd.DataFrame({'relative_target': [relative_target]})

    for col in X.columns:
        if col.startswith('venue_'):
            input_data[col] = 0

    venue_col = f"venue_{selected_venue}"

    if venue_col in input_data.columns:
        input_data[venue_col] = 1

    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    prob = model.predict_proba(input_data)[0][1]

    print("\n==============================")
    print(f"Matched Venue: {selected_venue}")
    print(f"Target: {target}")
    print(f"Win Probability: {round(prob*100,2)}%")

    if prob >= 0.5:
        print("Prediction: Likely WIN ")
    else:
        print("Prediction: Likely LOSS ")

    print("==============================\n")


while True:
    venue_input = input("Enter Stadium Name (or type 'exit'): ")
    
    if venue_input.lower() == "exit":
        break

    target_input = int(input("Enter Target Score: "))

    predict_chasing_win(venue_input, target_input)

df.to_csv("processed_ipl_data.csv", index=False)