import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from joblib import dump
from energy_consumption_architecture.utils.paths import data_dir
from xgboost import XGBRegressor

# Function to calculate the correlation matrix and plot a heatmap
def matriz_correlacion(dataset, target):
    corr_matrix = dataset.corr()
    corr_matrix[target].sort_values(ascending=False)
    cm_red_blue = mpl.colormaps['RdBu']
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    f, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap=cm_red_blue, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Heatmap with Numerical Values and Colors')
    plt.show()

# Calcular el VIF para cada característica
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data
# Remover iterativamente características con VIF alto
def remove_high_vif_features(df, threshold=10):
    while True:
        vif_data = calculate_vif(df)
        max_vif = vif_data['VIF'].max()
        if max_vif > threshold:
            feature_to_remove = vif_data.loc[vif_data['VIF'] == max_vif, 'feature'].values[0]
            df = df.drop(columns=[feature_to_remove])
        else:
            break
    return df, vif_data
# Extraer características temporales
def extract_time_features(df):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    return df
# Dividir en entrenamiento y prueba para series de tiempo
def time_series_train_test_split(df, target, test_size=0.2):
    n_test = int(len(df) * test_size)
    train = df[:-n_test]
    test = df[-n_test:]
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_test, y_test = test.drop(columns=[target]), test[target]
    return X_train, X_test, y_train, y_test
# Entrenar y evaluar un modelo
def train_evaluate_model(model, X_train, X_test, y_train, y_test):
    X_train_array, X_test_array = X_train.values, X_test.values  # Convertir a matrices NumPy
    y_train_array, y_test_array = y_train.values, y_test.values
    
    model.fit(X_train_array, y_train_array)
    y_train_pred = model.predict(X_train_array)
    y_test_pred = model.predict(X_test_array)
    
    rmse_train = np.sqrt(mean_squared_error(y_train_array, y_train_pred))
    mae_train = mean_absolute_error(y_train_array, y_train_pred)
    r2_train = r2_score(y_train_array, y_train_pred)
    
    rmse_test = np.sqrt(mean_squared_error(y_test_array, y_test_pred))
    mae_test = mean_absolute_error(y_test_array, y_test_pred)
    r2_test = r2_score(y_test_array, y_test_pred)
    
    return {
        "Model": model,
        "Train RMSE": rmse_train,
        "Train MAE": mae_train,
        "Train R2": r2_train,
        "Test RMSE": rmse_test,
        "Test MAE": mae_test,
        "Test R2": r2_test
    }
# Evaluar todos los modelos para un cluster
def evaluate_models_for_cluster(X_train, X_test, y_train, y_test, models):
    results = []
    for model_name, model in models.items():
        metrics = train_evaluate_model(model, X_train, X_test, y_train, y_test)
        metrics["Model Name"] = model_name
        results.append(metrics)
    results_df = pd.DataFrame(results)
    return results_df
def detect_overfitting(metrics_df, threshold_ratio=1.5):
    overfitted_models = []
    
    for index, row in metrics_df.iterrows():
        train_rmse = row["Train RMSE"]
        test_rmse = row["Test RMSE"]
        model_name = row["Model Name"]
        
        # Verificar si train_rmse es cero para evitar división por cero
        if train_rmse == 0:
            # Si el RMSE de prueba es significativamente mayor que cero, considerar el modelo sobreentrenado
            if test_rmse > threshold_ratio:
                overfitted_models.append(model_name)
        else:
            # Evaluar si la métrica de prueba es significativamente mayor que la de entrenamiento
            if test_rmse / train_rmse > threshold_ratio:
                overfitted_models.append(model_name)
    
    return overfitted_models
# Seleccionar el mejor modelo basado en RMSE, MAE y R²
def select_best_model(metrics_df, overfitted_models):
    # Excluir los modelos detectados como sobreentrenados
    metrics_df = metrics_df[~metrics_df["Model Name"].isin(overfitted_models)]
    
    # Ordenar los modelos primero por RMSE de prueba, luego por MAE de prueba, y finalmente por R² de prueba
    sorted_metrics = metrics_df.sort_values(by=["Test RMSE", "Test MAE", "Test R2"], ascending=[True, True, False])
    best_model = sorted_metrics.iloc[0]  # Seleccionar la primera fila como el mejor modelo
    return best_model
# Pipeline principal para el entrenamiento y selección de modelos por cluster
def pipeline_for_clusters(df, target, models, test_size=0.2, vif_threshold=10,threshold_ratio=2):
    metrics_by_cluster = []
    best_models_by_cluster = []

    # Procesar por cada cluster
    for cluster in df['Cluster'].unique():
        print(f"Processing Cluster {cluster}")
        
        # Filtrar datos del cluster y extraer características temporales
        cluster_data = df[df['Cluster'] == cluster].copy()
        cluster_data = extract_time_features(cluster_data)
        
        # Eliminar la columna 'Cluster' del conjunto de datos
        cluster_data = cluster_data.drop(columns=['Cluster'])
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = time_series_train_test_split(cluster_data, target, test_size)
        
        # Remover características con alto VIF
        #X_train, vif_data = remove_high_vif_features(X_train, threshold=vif_threshold)
        #X_test = X_test[X_train.columns]  # Asegurarse de que X_test tenga las mismas columnas que X_train
        
        # Evaluar todos los modelos para el cluster
        results_df = evaluate_models_for_cluster(X_train, X_test, y_train, y_test, models)
        
        # Detectar modelos sobreentrenados
        overfitted_models = detect_overfitting(results_df,threshold_ratio)
        
        # Seleccionar el mejor modelo del cluster
        best_model = select_best_model(results_df, overfitted_models)
        best_models_by_cluster.append(best_model)
        
        # Agregar resultados de todos los modelos al resumen general
        results_df["Cluster"] = cluster
        metrics_by_cluster.append(results_df)
        
        # Mostrar el mejor modelo para el cluster
        print(f"Best model for Cluster {cluster}:\n{best_model}\n")
    
    # Combinar todas las métricas en un solo DataFrame
    metrics_df = pd.concat(metrics_by_cluster, ignore_index=True)
    metrics_df.drop(columns=["Model"],inplace=True)
    best_models_df = pd.DataFrame(best_models_by_cluster)
    return metrics_df, best_models_df
