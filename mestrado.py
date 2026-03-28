import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve)

def trainAndTestSplit(X_encoded, y):
    print("\n Separando dados em treino e teste (80% treino, 20% teste)")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Grupo Treino: {X_train.shape}")
    print(f"Grupo Teste: {X_test.shape}")
    print(f"Proporção classes treino - 0: {(y_train==0).sum()}, 1: {(y_train==1).sum()}")
    return X_train, X_test, y_train, y_test

def logicalRegression(dataFrame):
    dataFrame['income'] = (dataFrame['income'] == '>50K').astype(int)

    X = dataFrame.drop('income', axis=1)
    y = dataFrame['income']

    categorical_cols = X.select_dtypes(include=['object']).columns
    print(f"São colunas categóricas: {categorical_cols.tolist()}")

    label_encoders = {}
    X_encoded = X.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    print(f"Codificação realizada em {len(categorical_cols)} colunas categóricas usando LabelEncoder")

    X_train, X_test, y_train, y_test = trainAndTestSplit(X_encoded, y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Dados padronizados por StandardScaler")

    print("TREINAMENTO DO MODELO DE REGRESSÃO LOGÍSTICA")
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1).fit(X_train_scaled, y_train)

    print("Modelo treinado com sucesso!")
    print(f"Coeficientes de regularização: {model.C}")

    training = model.predict(X_train_scaled)
    testing = model.predict(X_test_scaled)
    conf = model.predict_proba(X_test_scaled)[:, 1]

    print("\nACURÁCIA:")
    print(f"Treino: {accuracy_score(y_train, training):.4f}")
    print(f"Teste:  {accuracy_score(y_test, testing):.4f}")

    print("\nAUC-ROC:")
    print(f"Treino: {roc_auc_score(y_train, model.predict_proba(X_train_scaled)[:, 1]):.4f}")
    print(f"Teste:  {roc_auc_score(y_test, conf):.4f}")

    print("\n RELATÓRIO DE CLASSIFICAÇÃO (Teste):")
    print(classification_report(y_test, testing, 
                            target_names=['<=50K', '>50K']))

    print("\n MATRIZ DE CONFUSÃO (Teste):")
    cm = confusion_matrix(y_test, testing)
    print(cm)

    fpr, tpr, _ = roc_curve(y_test, conf)

    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], linestyle='--')
    plt.title("Curva ROC - Regressão Logística")
    plt.show()

    feature_importance = pd.DataFrame({
        'Feature': X_encoded.columns,
        'Coeficiente': model.coef_[0]
    }).sort_values('Coeficiente', key=abs, ascending=False)
    print(feature_importance.to_string(index=False))

    print("ANÁLISE COMPLETA DE REGRESSÃO LOGÍSTICA FINALIZADA!")

path = "adult.csv"
df = pd.read_csv(path, na_values=["?"])

df_clean = df.dropna()
print(f"Removidas linhas com valores faltantes: \n antes: {df.shape[0]} depois: {df_clean.shape[0]} removidas: {df.shape[0] - df_clean.shape[0]}")
logicalRegression(df_clean)

