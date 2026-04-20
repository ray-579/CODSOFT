# ================================
# 1. Import Libraries
# ================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import seaborn as sns
import matplotlib.pyplot as plt

# ================================
# 2. Load Dataset
# ================================
data = pd.read_csv("A:/Churn_Modelling.csv")

print(data.head())

# ================================
# 3. Drop Irrelevant Columns
# ================================
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# ================================
# 4. Encode Categorical Data
# ================================

# Gender → Binary
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# Geography → One-Hot Encoding
data = pd.get_dummies(data, columns=['Geography'], drop_first=True)

# ================================
# 5. Features & Target
# ================================
X = data.drop('Exited', axis=1)
y = data['Exited']

# ================================
# 6. Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# 7. Feature Scaling
# ================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# 8. Initialize Models
# ================================
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# ================================
# 9. Train & Evaluate
# ================================
for name, model in models.items():
    print(f"\n========== {name} ==========")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# ================================
# 10. Feature Importance (Random Forest)
# ================================
rf = models["Random Forest"]

importances = rf.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:\n", importance_df)

sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance (Random Forest)")
plt.show()