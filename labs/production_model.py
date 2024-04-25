import pandas as pd
from joblib import load
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

def load_data(filename):
    """Laddar in CSV-fil och returnerar en DataFrame."""
    return pd.read_csv(filename)

def load_model(model_path):
    """Laddar en sparad modell från en .pkl-fil."""
    return load(model_path)

def make_predictions(model, features):
    """Gör förutsägelser och returnerar sannolikheter samt förutsägelser."""
    probabilities = model.predict_proba(features)
    predictions = model.predict(features)
    return probabilities, predictions

def save_predictions(filename, probabilities, predictions):
    """Sparar sannolikheter och förutsägelser till en CSV-fil."""
    df = pd.DataFrame({
        'probability class 0': probabilities[:, 0],
        'probability class 1': probabilities[:, 1],
        'prediction': predictions
    })
    df.to_csv(filename, index=False)

def main():
    data_path = 'labs/labb1/test_samples.csv'  # Uppdatera sökvägen efter behov
    model_path = 'labs/labb1/voting_classifier_model.pkl'  # Uppdatera sökvägen efter behov

    # Ladda in testdata
    test_data = load_data(data_path)
    features = test_data.drop('target', axis=1)  # Anta att 'target' är kolumnen med sanna etiketter

    # Definiera och konfigurera VotingClassifier med 'soft' voting
    model = VotingClassifier(estimators=[
        ('lr', LogisticRegression(solver='liblinear', C=1, penalty='l2')),
        ('dt', DecisionTreeClassifier(max_depth=5, min_samples_split=5)),
        ('rf', RandomForestClassifier(n_estimators=50, max_features='sqrt'))
    ], voting='soft')
    model.fit(features, test_data['target'])  # Anta att vi har 'target' i test_data för demonstration

    # Gör förutsägelser
    probabilities, predictions = make_predictions(model, features)

    # Spara förutsägelser
    save_predictions('prediction.csv', probabilities, predictions)

    print("Förutsägelser har sparats till 'prediction.csv'.")

if __name__ == "__main__":
    main()
import os
print("Aktuell arbetskatalog:", os.getcwd())