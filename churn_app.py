import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# ----------------------------------------------------------------------
# Configuración de la página
# ----------------------------------------------------------------------
st.set_page_config(page_title="Predicción de Fuga de Clientes", layout="wide")

st.title("🚀 App de Predicción de Fuga de Clientes (Churn)")
st.write("""
Esta aplicación demuestra el proceso completo de un proyecto de Machine Learning, 
desde el análisis de datos hasta un modelo predictivo interactivo para predecir si un cliente cancelará su servicio.
""")

# ----------------------------------------------------------------------
# Carga de datos y caché
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # Preprocesamiento inicial
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    return df

df = load_data()

# ----------------------------------------------------------------------
# Función para entrenar el modelo (cacheada para eficiencia)
# ----------------------------------------------------------------------
@st.cache_resource
def train_model(df):
    # Copia para evitar modificar el dataframe cacheado
    df_model = df.copy()
    
    # Preprocesamiento para el modelo
    df_model['Churn'] = df_model['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    df_processed = pd.get_dummies(df_model, drop_first=True)
    
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    
    # Guardar columnas para la predicción en vivo
    model_columns = X.columns
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Usamos RandomForest, que tuvo buen desempeño
    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
    model.fit(X_train, y_train)
    
    return model, scaler, X_test, y_test, model_columns

model, scaler, X_test, y_test, model_columns = train_model(df.copy())

# ----------------------------------------------------------------------
# Sección Principal: Exploración y Resultados del Modelo
# ----------------------------------------------------------------------
st.header("1. Análisis Exploratorio y Visualización de Datos")

# Opción para mostrar el dataset
if st.checkbox("Mostrar el dataset completo"):
    st.dataframe(df)

# Visualizaciones
col1, col2 = st.columns(2)
with col1:
    st.subheader("Distribución de Churn")
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=df, ax=ax)
    st.pyplot(fig)
with col2:
    st.subheader("Churn por Tipo de Contrato")
    fig, ax = plt.subplots()
    sns.countplot(x='Contract', hue='Churn', data=df, ax=ax)
    st.pyplot(fig)

# ----------------------------------------------------------------------
st.header("2. Rendimiento del Modelo de Clasificación")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

# Métricas
col1, col2 = st.columns(2)
col1.metric("Accuracy del Modelo", f"{accuracy:.2%}")
col2.metric("ROC-AUC Score", f"{roc_auc:.4f}")

# Reporte y Matriz de confusión
with st.expander("Ver detalles del rendimiento (Reporte y Matriz de Confusión)"):
    st.text("Reporte de Clasificación:")
    st.text(classification_report(y_test, y_pred))

    st.text("Matriz de Confusión:")
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    st.pyplot(fig)

# Importancia de características
st.subheader("Importancia de las Características")
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': model_columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
plt.title('Top 10 Características más Influyentes')
st.pyplot(fig)

# ----------------------------------------------------------------------
# Barra Lateral: Predicción Interactiva
# ----------------------------------------------------------------------
st.sidebar.header("🔮 Realizar una Predicción de Churn")
st.sidebar.write("Introduce los datos de un cliente para predecir si cancelará.")

# Inputs del usuario
tenure = st.sidebar.slider("Permanencia (meses)", 1, 72, 12)
monthly_charges = st.sidebar.slider("Cargos Mensuales ($)", 18.0, 120.0, 70.0)
total_charges = st.sidebar.number_input("Cargos Totales ($)", value=float(monthly_charges * tenure))
contract = st.sidebar.selectbox("Tipo de Contrato", ['Month-to-month', 'One year', 'Two year'])
tech_support = st.sidebar.selectbox("Soporte Técnico", ['No', 'Yes', 'No internet service'])
online_security = st.sidebar.selectbox("Seguridad Online", ['No', 'Yes', 'No internet service'])
payment_method = st.sidebar.selectbox("Método de Pago", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

if st.sidebar.button("Predecir Churn"):
    # Crear un dataframe con los inputs
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })
    
    # Crear un dataframe dummy con todas las columnas posibles para one-hot encoding
    input_df_processed = pd.DataFrame(columns=model_columns)
    input_df_processed.loc[0] = 0 # Inicializar con ceros
    
    # Asignar valores numéricos
    input_df_processed['tenure'] = tenure
    input_df_processed['MonthlyCharges'] = monthly_charges
    input_df_processed['TotalCharges'] = total_charges

    # Aplicar One-Hot Encoding a las variables categóricas
    if contract != 'Month-to-month':
        col_contract = f'Contract_{contract}'
        if col_contract in input_df_processed.columns:
            input_df_processed[col_contract] = 1
            
    if tech_support == 'Yes':
        if 'TechSupport_Yes' in input_df_processed.columns:
            input_df_processed['TechSupport_Yes'] = 1
            
    if online_security == 'Yes':
        if 'OnlineSecurity_Yes' in input_df_processed.columns:
            input_df_processed['OnlineSecurity_Yes'] = 1
            
    # Llenar otros campos categóricos... (simplificado por brevedad)

    # Escalar las variables numéricas con el scaler ya ajustado
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df_processed[numerical_cols] = scaler.transform(input_df_processed[numerical_cols])

    # Realizar la predicción
    prediction = model.predict(input_df_processed)[0]
    prediction_proba = model.predict_proba(input_df_processed)[0]

    st.sidebar.subheader("Resultado de la Predicción:")
    if prediction == 1:
        st.sidebar.error("El cliente tiene ALTA probabilidad de cancelar (Churn).")
        st.sidebar.metric("Probabilidad de Churn", f"{prediction_proba[1]:.2%}")
    else:
        st.sidebar.success("El cliente tiene BAJA probabilidad de cancelar.")
        st.sidebar.metric("Probabilidad de Permanencia", f"{prediction_proba[0]:.2%}")