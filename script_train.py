import os
import datetime
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer, classification_report
from xgboost import XGBClassifier
import joblib

# Cargar los datos
df = pd.read_csv('data/train.csv')

# Definir columnas continuas y discretas
continuous_cols = [
    'battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt',
    'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time'
]

discrete_cols = [
    'blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi'
]

# Separar características y etiqueta
X = df.drop('price_range', axis=1)
y = df['price_range']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=11)

# Preprocesamiento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_cols),
        ('cat', OneHotEncoder(), discrete_cols)
    ]
)

# Modelo inicial
dt_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=7, random_state=11))
])

# Entrenar el modelo de DecisionTree
dt_model.fit(X_train, y_train)

# Predecir y evaluar DecisionTree
y_pred = dt_model.predict(X_test)
print("Decision Tree Classifier Report")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Crear directorios para guardar modelos y resultados
model_dir = "/modelos"
results_dir = "/resultados"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Guardar los resultados en CSV con la fecha actual
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
report_filename = f'{results_dir}/classification_report_{current_date}.csv'
report_df.to_csv(report_filename, index=True)

# Guardar el modelo entrenado
model_filename = f'{model_dir}/decision_tree_model_{current_date}.pkl'
joblib.dump(dt_model, model_filename)

# Hiperparámetros para XGBoost
param_dist = {
    'classifier__n_estimators': np.arange(50, 201, 10),
    'classifier__max_depth': np.arange(3, 15, 1),
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__gamma': np.arange(0, 1.1, 0.1)
}

# Pipeline para XGBoost
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
])

# Búsqueda de hiperparámetros
scorer = make_scorer(f1_score, average='macro')
random_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=param_dist,
    n_iter=2,
    scoring=scorer,
    cv=3,
    verbose=1,
    random_state=11,
    n_jobs=-1
)

# Entrenar la búsqueda de hiperparámetros
random_search.fit(X_train, y_train)

# Mostrar los mejores hiperparámetros
best_params = random_search.best_params_
print("Mejores hiperparámetros:", best_params)

# Guardar el mejor modelo de XGBoost
best_model_filename = f'{model_dir}/xgb_best_model_{current_date}.pkl'
joblib.dump(random_search.best_estimator_, best_model_filename)

# Cargar y evaluar el mejor modelo de XGBoost
y_pred = random_search.best_estimator_.predict(X_test)
print("XGBoost Classifier Report")
report_xgb = classification_report(y_test, y_pred, output_dict=True)
report_xgb_df = pd.DataFrame(report_xgb).transpose()

# Guardar el classification report de XGBoost en CSV con la fecha
report_xgb_filename = f'{results_dir}/xgb_classification_report_{current_date}.csv'
report_xgb_df.to_csv(report_xgb_filename, index=True)
