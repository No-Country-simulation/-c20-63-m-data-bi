import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import numpy as np

from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report


st.title("👋Hello DS👋")

st.info("🚀Esta Aplicación construye un modelo de Machine Learning🚀")

with st.expander('Data'):
    st.write('**Raw Data**')
    df = pd.read_csv('carrito.csv', index_col=False)
    df = df.drop(df.columns[0], axis=1)
    df

    st.write('**X**')
    X = df.drop(['Reason_for_abandonment'], axis=1)
    X

    st.write('**y**')
    y = df['Reason_for_abandonment']
    y

# Center Markdown
# Write the title in a centered format
st.markdown("""
    <div style="text-align:center">
        <h2>Número de compras último año por Edad</h2>
    </div>
""", unsafe_allow_html=True)

# Group by Age_range and Reason_for_abandonment, and count the Number_of_purchases_in_last_year
count_df = df.groupby(['Age_range', 'Reason_for_abandonment']).size().reset_index(name='Frequency')

# Create the bar chart using Altair
chart = alt.Chart(count_df).mark_bar().encode(
    x='Age_range',
    y='Frequency',
    color='Reason_for_abandonment'
).properties(width=600, height=400)

# Render the chart in Streamlit
st.altair_chart(chart, use_container_width=True)

# Variables de Entrada
with st.sidebar:
    st.header("Variables de Entrada")
    gender = st.selectbox('Gender', ('Male', 'Female'))
    age_range = st.slider('Age_range', 19,80)
    number_purchases = st.slider('Number_of_purchases_in_last_year', 5, 45)
    friendly_interface = st.slider('User_friendly_interface', 0, 4)
    web_app = st.selectbox('Better_website_application', ('One_platform',
       'Two_platforms', 'Three_platforms','Four_or_more_platforms'))
    
    # Crear DataFrame
    data = {'Gender':gender,
            'Age_range':age_range,
            'Number_of_purchases_in_last_year':number_purchases,
            'User_friendly_interface':friendly_interface,
            'Better_website_application':web_app
    }
    input_df = pd.DataFrame(data, index=[0])
    input_datos = pd.concat([input_df, X], axis=0)

with st.expander('**Varibles Entrada**'):
    st.write('**Fila Demostración**')
    input_df
    st.write('**Datos Combinados**')
    input_datos
  
# PREPARACIÓN DE DATOS
with st.expander('**Varibles Prueba y Entrenamiento**'):
    #Encode y
    y_mapper = {'Better alternative offer':0,
                'Change in price':1,
                'Lack of trust':2,
                'No preferred mode of payment':3,
                'Promo code not applicable':4
                }
    def target_encode(val):
        return y_mapper[val]

    y = y.apply(target_encode)

    # Perform train-test split
    SEED = 88
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    st.write('**Train-Test Split**')
    st.write(f'X_train shape: {X_train.shape}')
    st.write(f'X_test shape: {X_test.shape}')
    st.write(f'y_train shape: {y_train.shape}')
    st.write(f'y_test shape: {y_test.shape}')

    # Encode categorical variables
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    st.write('**Encoded X_train**')
    st.dataframe(X_train.head())

    st.write('**Encoded X_test**')
    st.dataframe(X_test.head())
    # old
    # Find the size of the smallest minority class
    minority_class_size = pd.Series(y_train).value_counts().min()
    # Ajustar que sea un número menor que la cantidad menor
    k_neighbors_value = minority_class_size - 1
    # Definir el algoritmo para Oversampling
    method = SMOTE(random_state=SEED, k_neighbors=k_neighbors_value)
    # Ajustando SMOTE
    X_train_resampled, y_train_resampled = method.fit_resample(X_train, y_train)

    st.write('**Tratamiento Desbalance**')
    st.write(f'X_train con SMOTE: {X_train_resampled.shape}')
    st.write(f'y_train con SMOTE: {y_train_resampled.shape}')

    # Definir Scaler y ajustarlo
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # Inicializar XGBoost
    xgb = XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', num_class=5)

    # Definir parámetros de XGBoost para GridSearchCV
    params_xgb = {'max_depth': [4,5], #Intenté con más parámetros pero traté de nivelar los resultados de Cross Validation
                'learning_rate': [0.01, 0.05, 0.1, 0.2],  #Intenté con más parámetros pero traté de nivelar los resultados de Cross Validation
                'colsample_bytree': [0.77, 0.88],  # Similar a subsample
                'colsample_bylevel': [0.7],
                'n_estimators': np.arange(100, 150, 10),  # XGBoost es más rápido ajustar de manera adecuada
                'gamma': [0, 0.1],
                'random_state': [SEED]}

    random_search = RandomizedSearchCV(estimator=xgb,
                                   param_distributions=params_xgb,
                                   n_iter=50,  # Number of parameter settings sampled
                                   cv=5,
                                   scoring='f1_weighted',
                                   n_jobs=-1,
                                   random_state=SEED)
    random_search.fit(X_train_scaled, y_train_resampled)

    best_params = random_search.best_params_
    best_score = random_search.best_score_

    # Definir XGBoost con los siguientes Hiperparámetros
    xgb = XGBClassifier(
        objective='multi:softmax',
        eval_metric='mlogloss',
        num_class=5,
        max_depth=5,
        learning_rate=0.1,
        colsample_bytree=0.77,
        colsample_bylevel=0.7,
        n_estimators=140,
        gamma=0,
        random_state=SEED
    )

    # Ajustar XGBoost en los Sets de Entrenamiento
    xgb.fit(X_train_scaled, y_train_resampled)

    # Hacer predicciones en el Set de Prueba
    y_pred = xgb.predict(X_test_scaled)

    # Calcular probabilidades de ambas clases en variable Objetivo
    y_proba = xgb.predict_proba(X_test_scaled)

    # Imprimiendo el resultado de Predecir las variable Attrited como Verdadera
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr') # Add multi_class parameter

    # Generate the classification report
    report = classification_report(y_test, y_pred)

    # Display the classification report in Streamlit
    st.text("**Reporte de Clasificación**:")
    st.text(report)

with st.expander('**Predicción Categoría**'):
    # Encode categorical variables in input_df
    input_df = pd.get_dummies(input_df)

    # Ensure the columns in input_df match the columns in X_train
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

    # Scale input_df using the same scaler
    input_df_scaled = scaler.transform(input_df)

    # Get the predicted probabilities for input_df
    input_proba = xgb.predict_proba(input_df_scaled)

    # Print the predicted probabilities
    st.write(input_proba)

    # Get the class with the highest probability
    predicted_class = np.argmax(input_proba)

    # Map the predicted class to the corresponding reason for abandonment
    reason_mapper = {0: 'Better alternative offer',
                    1: 'Change in price',
                    2: 'Lack of trust',
                    3: 'No preferred mode of payment',
                    4: 'Promo code not applicable'}

    predicted_reason = reason_mapper[predicted_class]

    # Print the predicted reason
    st.write('**Predicted Reason for Abandonment**')
    st.write(predicted_reason)
    
