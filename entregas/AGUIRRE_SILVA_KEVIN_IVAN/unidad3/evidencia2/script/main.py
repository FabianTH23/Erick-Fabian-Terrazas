import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 1. Preparación de datos (Simulación de un dataset)
# -----------------------------------------------------------------

# Crear un DataFrame simulado
np.random.seed(42)
data = {
    'Area_m2': np.random.randint(50, 200, 100),
    'Num_habitaciones': np.random.randint(1, 5, 100),
    'Antiguedad_anios': np.random.randint(1, 50, 100),
    'Distancia_centro_km': np.random.uniform(0.5, 20, 100)
}
df = pd.DataFrame(data)

# La columna 'Precio' (Variable Dependiente) se genera como una función lineal + ruido
# Precio = 2000 * Area + 10000 * Habitaciones - 500 * Antiguedad - 1000 * Distancia + ruido
df['Precio'] = (
    2000 * df['Area_m2'] + 
    10000 * df['Num_habitaciones'] - 
    500 * df['Antiguedad_anios'] - 
    1000 * df['Distancia_centro_km'] + 
    np.random.normal(0, 50000, 100) # Añadir ruido
)

# Definición de variables de entrada (X) y salida (y)
X = df[['Area_m2', 'Num_habitaciones', 'Antiguedad_anios', 'Distancia_centro_km']]
y = df['Precio']

# División entrenamiento/prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\n\nDatos de entrenamiento:", X_train.shape, y_train.shape)
print("Datos de prueba:", X_test.shape, y_test.shape)

# 2. Entrenamiento del modelo
# -----------------------------------------------------------------
print("\n\n--- Entrenando Modelo de Regresión Lineal ---")
modelo_rl = LinearRegression()
modelo_rl.fit(X_train, y_train)

# Predicciones sobre el conjunto de prueba
y_pred = modelo_rl.predict(X_test)

# 3. Cálculo de las métricas seleccionadas
# -----------------------------------------------------------------

# Error Cuadrático Medio (MSE)
mse = mean_squared_error(y_test, y_pred)
# Raíz del Error Cuadrático Medio (RMSE)
rmse = np.sqrt(mse)
# Error Absoluto Medio (MAE)
mae = mean_absolute_error(y_test, y_pred)
# Coeficiente de Determinación (R²)
r2 = r2_score(y_test, y_pred)

print("\n--- Métricas de Evaluación del Modelo ---")
print(f"Error Cuadrático Medio (MSE): {mse:,.2f}")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:,.2f}")
print(f"Error Absoluto Medio (MAE): {mae:,.2f}")
print(f"Coeficiente de Determinación (R²): {r2:.4f}")

# Ejemplo de predicción para un caso
ejemplo = pd.DataFrame({
    'Area_m2': [130], 'Num_habitaciones': [3], 
    'Antiguedad_anios': [5], 'Distancia_centro_km': [3.0]
})
prediccion_ejemplo = modelo_rl.predict(ejemplo)
print(f"\nPredicción para la vivienda de ejemplo: ${prediccion_ejemplo[0]:,.2f}\n\n")