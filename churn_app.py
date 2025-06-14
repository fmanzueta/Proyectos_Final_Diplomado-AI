#----------------------------------------------------------------
# Proyecto Final 1: Predicción de Fuga de Clientes (Churn)
#----------------------------------------------------------------

# 1. Carga y Análisis Exploratorio del Dataset (EDA)

# Importación de librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df_churn = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Vistazo inicial a los datos
print("Primeras 5 filas del dataset:")
print(df_churn.head())
print("\nInformación general del dataset:")
df_churn.info()

# Revisar valores faltantes
# La columna 'TotalCharges' tiene algunos valores faltantes que parecen ser espacios en blanco.
# Convertimos la columna a numérico, forzando los errores a NaN (Not a Number)
df_churn['TotalCharges'] = pd.to_numeric(df_churn['TotalCharges'], errors='coerce')
print(f"\nValores faltantes por columna:\n{df_churn.isnull().sum()}")

# Análisis exploratorio de datos (EDA)
print("\nDescripción estadística de las variables numéricas:")
print(df_churn.describe())

# Visualización de la variable objetivo 'Churn'
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df_churn)
plt.title('Distribución de Fuga de Clientes (Churn)')
plt.show()

# Visualización de variables categóricas vs. Churn
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

for feature in categorical_features:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=feature, hue='Churn', data=df_churn)
    plt.title(f'{feature} vs. Churn')
    plt.xticks(rotation=45)
    plt.show()

#----------------------------------------------------------------
# 2. Preprocesamiento de Datos
#----------------------------------------------------------------

# Manejo de valores faltantes en 'TotalCharges'
# Dado que son pocos (11), una estrategia simple es imputarlos con la mediana.
median_total_charges = df_churn['TotalCharges'].median()
df_churn['TotalCharges'].fillna(median_total_charges, inplace=True)
print(f"\nValores faltantes después de la imputación:\n{df_churn.isnull().sum()}")

# Eliminamos la columna 'customerID' ya que no es útil para el modelo
df_churn.drop('customerID', axis=1, inplace=True)

# Codificación de variables categóricas
# Convertimos la variable objetivo a 0 y 1
df_churn['Churn'] = df_churn['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Usamos One-Hot Encoding para las demás variables categóricas
df_processed = pd.get_dummies(df_churn, drop_first=True)

print("\nDimensiones del dataset después del preprocesamiento:", df_processed.shape)
print("Columnas del dataset procesado:")
print(df_processed.columns)


# Separación de características (X) y variable objetivo (y)
X = df_processed.drop('Churn', axis=1)
y = df_processed['Churn']

# Escalado de variables numéricas
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# Seleccionamos solo las columnas numéricas para escalar
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print("\nPrimeras filas de las características (X) escaladas:")
print(X.head())

#----------------------------------------------------------------
# 4. División del Conjunto de Datos
#----------------------------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTamaño del conjunto de entrenamiento: {X_train.shape}")
print(f"Tamaño del conjunto de prueba: {X_test.shape}")


#----------------------------------------------------------------
# 5. Entrenamiento y Evaluación de Modelos
#----------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

# Función para evaluar modelos
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Modelo: {type(model).__name__}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {type(model).__name__}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.show()

# Inicializar y evaluar los modelos
# Modelo 1: Regresión Logística
lr = LogisticRegression(max_iter=1000)
evaluate_model(lr, X_train, y_train, X_test, y_test)

# Modelo 2: Random Forest
rf = RandomForestClassifier(random_state=42)
evaluate_model(rf, X_train, y_train, X_test, y_test)

# Modelo 3: Support Vector Machine (con probabilidad=True para ROC-AUC)
svc = SVC(probability=True, random_state=42)
evaluate_model(svc, X_train, y_train, X_test, y_test)

#----------------------------------------------------------------
# 6. Optimización del Mejor Modelo (Random Forest)
#----------------------------------------------------------------
from sklearn.model_selection import GridSearchCV

print("\n--- Optimización de Random Forest con GridSearchCV ---")

# Definir la parrilla de hiperparámetros
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Configurar GridSearchCV
# Usamos un n_jobs=-1 para usar todos los procesadores disponibles
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='roc_auc')

# Ejecutar la búsqueda
grid_search.fit(X_train, y_train)

print(f"\nMejores hiperparámetros encontrados: {grid_search.best_params_}")

# Evaluar el mejor modelo encontrado
best_rf = grid_search.best_estimator_
evaluate_model(best_rf, X_train, y_train, X_test, y_test)

#----------------------------------------------------------------
# 7. Interpretación de Resultados
#----------------------------------------------------------------
print("\n--- Importancia de las Características ---")

# Obtener la importancia de las características del mejor modelo
importances = best_rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Ordenar por importancia
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Visualizar las 15 características más importantes
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15))
plt.title('Top 15 Características más Influyentes en la Fuga de Clientes')
plt.show()

print("\nAnálisis final:")
print("Las variables más influyentes para predecir la fuga de clientes son el tipo de contrato, la permanencia (tenure), el gasto mensual y el servicio de soporte técnico.")
print("Un cliente con contrato mes a mes, poco tiempo en la compañía y sin servicios de protección adicionales es más propenso a cancelar.")
