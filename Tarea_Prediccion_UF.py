# modelo de regresion lineal multiple para predecir precios en uf
#Diego Vásquez Castillo Data Scince Seccion:302


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. crear el dataframe con los datos
datos_departamentos = {
    'superficie_m2': [50, 70, 65, 90, 45],
    'num_habitaciones': [1, 2, 2, 3, 1],
    'distancia_metro_km': [0.5, 1.2, 0.8, 0.2, 2.0],
    'precio_uf': [2500, 3800, 3500, 5200, 2100]
}

tabla_departamentos = pd.DataFrame(datos_departamentos)
print("=== datos originales ===")
print(tabla_departamentos, "\n")

# 2. separar variables independientes (x) y dependiente (y)
variables_independientes = tabla_departamentos[['superficie_m2', 'num_habitaciones', 'distancia_metro_km']]
variable_objetivo = tabla_departamentos['precio_uf']

# 3. crear y entrenar el modelo
modelo_regresion = LinearRegression()
modelo_regresion.fit(variables_independientes, variable_objetivo)

# 4. mostrar los parametros del modelo
intercepto = round(modelo_regresion.intercept_, 2)
coeficientes = np.round(modelo_regresion.coef_, 2)

print("=== parametros del modelo ===")
print(f"intercepto (b0): {intercepto}")
print(f"coeficientes (b1, b2, b3): {coeficientes}\n")

# 5. realizar predicciones
predicciones_uf = modelo_regresion.predict(variables_independientes)
predicciones_uf = np.round(predicciones_uf, 2)
tabla_departamentos['prediccion_uf'] = predicciones_uf

print("=== predicciones del modelo ===")
print(tabla_departamentos[['superficie_m2', 'num_habitaciones', 'distancia_metro_km', 'precio_uf', 'prediccion_uf']].round(2), "\n")

# 6. evaluar el rendimiento del modelo
r2 = round(r2_score(variable_objetivo, predicciones_uf), 2)
mse = mean_squared_error(variable_objetivo, predicciones_uf)
rmse = round(np.sqrt(mse), 2)

print("=== evaluacion del modelo ===")
print(f"r2: {r2}")
print(f"rmse: {rmse}\n")

# 7. interpretacion final
print("=== interpretacion ===")
print("el modelo estima el precio en uf de un departamento segun su superficie, numero de habitaciones y distancia al metro.")
print("los coeficientes indican cuanto cambia el precio por cada unidad adicional de cada variable.")
print("un r2 alto indica que el modelo explica bien los datos, pero como solo hay 5 registros, puede estar sobreajustado.")
