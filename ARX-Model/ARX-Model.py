import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error, median_absolute_error, max_error
from sklearn.preprocessing import StandardScaler

# ==================================================================================================
# Funktionen
# ==================================================================================================

def gleitender_mittelwert_allgemein(signal, vergangenheit=4, zukunft=0):
    """
    Berechnet den gleitenden Mittelwert über (vergangenheit + 1 + zukunft) Punkte.
    Alle Ränder, die nicht vollständig gefüllt werden können, erhalten NaN.
    """
    result = np.full_like(signal, fill_value=np.nan, dtype=np.float64)
    for i in range(vergangenheit, len(signal) - zukunft):
        window = signal[i - vergangenheit : i + zukunft + 1]
        result[i] = np.mean(window)
    return result

# --------------------------------------------------------------------------------------

def scale_to_range(signal, target_min=-1, target_max=1):
    """
    Skalierung auf [target_min, target_max].
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    if min_val == max_val:
        return np.full_like(signal, fill_value=(target_max + target_min) / 2)
    scaled = (signal - min_val) / (max_val - min_val)  # [0,1]
    scaled = scaled * (target_max - target_min) + target_min
    return scaled

# --------------------------------------------------------------------------------------

def standardize_to_target(signal, target_mean=0, target_std=1):
    """
    Standardisiert ein Signal auf einen beliebigen Mittelwert und eine beliebige Standardabweichung.
    Gibt zur Kontrolle den resultierenden Mittelwert und die Standardabweichung aus.

    Args:
        signal (np.array): 1D-Array mit Rohdaten.
        target_mean (float): Gewünschter Mittelwert nach Standardisierung (default = 0).
        target_std (float): Gewünschte Standardabweichung nach Standardisierung (default = 1).

    Returns:
        np.array: Transformiertes Signal mit target_mean und target_std.
    """
    scaler = StandardScaler()
    standardized = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
    transformed = standardized * target_std + target_mean

    print(f"→ Nach Transformation: Mittelwert = {np.mean(transformed):.4f}, Stdabw = {np.std(transformed):.4f}")

    return transformed

# --------------------------------------------------------------------------------------

def create_arx_matrix_multi(y, inputs, ny=1, nu=1):
    """
    Erzeugt Regressionsmatrix X und Zielvektor Y für ein ARX-Modell mit
    mehreren Eingängen.
    
    y: Ausgangsdaten (z. B. y_scaled)
    inputs: Liste der Eingänge (z. B. [u1_scaled, u2_scaled, ...])
    ny: Anzahl vergangener y-Werte
    nu: Anzahl vergangener u-Werte + aktuellen u-Wert
    """
    N = len(y)
    X_list = []
    Y_list = []

    # Wir beginnen ab max(ny, nu), da vorher die Vergangenheit fehlt
    for t in range(max(ny, nu), N):
        row = []
        # Autoregressive Anteile: y(t-1), y(t-2), ..., y(t-ny)
        for i in range(1, ny + 1):
            row.append(y[t - i])

        # Exogene Anteile: für jeden Eingang u(t), u(t-1), ..., u(t-nu)
        for u in inputs:
            for j in range(nu + 1):
                row.append(u[t - j])

        X_list.append(row)
        Y_list.append(y[t])

    X_arr = np.array(X_list)
    Y_arr = np.array(Y_list)
    return X_arr, Y_arr


# --------------------------------------------------------------------------------------------------












# ==================================================================================================
# Einlesen und validieren
# ==================================================================================================

u_df = pd.read_csv('u.csv', header=None)
u1 = u_df.iloc[:, 0].values
u2 = u_df.iloc[:, 1].values
u3 = u_df.iloc[:, 2].values

y_df = pd.read_csv('y.csv', header=None)
y = y_df.values.flatten()

print("u1:", u1[:5])
print("u2:", u2[:5])
print("u3:", u3[:5])
print("y :", y[:5])
print("=============================================================================================")

# ==================================================================================================
# Glätten und entstehende NaN Werte löschen
# ==================================================================================================

u1_glatt = gleitender_mittelwert_allgemein(u1, 4, 0)
u2_glatt = gleitender_mittelwert_allgemein(u2, 4, 0)
u3_glatt = gleitender_mittelwert_allgemein(u3, 4, 0)
y_glatt  = gleitender_mittelwert_allgemein(y,  4, 0)

valid_mask = ~np.isnan(u1_glatt) & ~np.isnan(u2_glatt) & ~np.isnan(u3_glatt) & ~np.isnan(y_glatt)
u1_clean = u1_glatt[valid_mask]
u2_clean = u2_glatt[valid_mask]
u3_clean = u3_glatt[valid_mask]
y_clean  = y_glatt[valid_mask]

print("u1_clean:", u1_clean[:5])
print("u2_clean:", u2_clean[:5])
print("u3_clean:", u3_clean[:5])
print("y_clean :",  y_clean[:5])
print("=============================================================================================")

# ==================================================================================================
# Skalieren
# ==================================================================================================

u1_scaled = scale_to_range(u1_clean, -1, 1)
u2_scaled = scale_to_range(u2_clean, -1, 1)
u3_scaled = scale_to_range(u3_clean, -1, 1)
y_scaled  = scale_to_range(y_clean, -1, 1)

# ===================================================================================
# Falls normalisieren auf Erwartungswert und Standartabweichung
# u1_scaled = standardize_to_target(u1_clean, 0, 1)
# u2_scaled = standardize_to_target(u2_clean, 0, 1)
# u3_scaled = standardize_to_target(u3_clean, 0, 1)
# y_scaled  = standardize_to_target(y_clean,  0, 1)
# ===================================================================================

print("u1_scaled:", u1_scaled[:5])
print("u2_scaled:", u2_scaled[:5])
print("u3_scaled:", u3_scaled[:5])
print("y_scaled :",  y_scaled[:5])
print("=============================================================================================")

# ==================================================================================================
# ARX-Modell manuell erstellen und trainieren (Least Squares)
# ==================================================================================================

# 1) Erzeuge Regressionsmatrix und Zielvektor
ny = 1  # y(t-1), ...
nu = 1  # u(t), u(t-1), ...
verstärkungsfaktor = 1.0

X, Y = create_arx_matrix_multi(y_scaled, [u1_scaled, u2_scaled, u3_scaled], ny=ny, nu=nu)

# 2) Split in Train/ Test
split_ratio = 0.8
split_index = int(len(Y) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = Y[:split_index], Y[split_index:]

# 3) Bau der phi-Matrix 
phi_matrix_train = X_train

#    y_t_train = ...
y_t_train = y_train  # (Werte bis zum Split)

# 4) Manuelle Summenbildung (Nenner/ Zähler)
nenner_temp = np.zeros((phi_matrix_train.shape[1], phi_matrix_train.shape[1]))
zaehler_temp = np.zeros(phi_matrix_train.shape[1])

for i in range(phi_matrix_train.shape[0]):
    phi_i = phi_matrix_train[i, :].reshape(-1, 1)  # Spaltenvektor
    nenner_temp += (phi_i @ phi_i.T)
    zaehler_temp += (phi_i.flatten() * y_t_train[i])

theta = np.linalg.inv(nenner_temp) @ zaehler_temp

print(f"Theta : {theta}")
print("=============================================================================================")

# =========================================================================================
# Testdaten vorbereiten
phi_matrix_test = X_test
y_test_real = y_test  # (Werte ab dem Split)

# Vorhersagen
y_predict_train = verstärkungsfaktor*(phi_matrix_train @ theta)
y_predict_test  = verstärkungsfaktor*(phi_matrix_test  @ theta)

# =========================================================================================
rmse_train = root_mean_squared_error(y_train, y_predict_train)
rmse_test  = root_mean_squared_error(y_test, y_predict_test)

mae_train = mean_absolute_error(y_train, y_predict_train)
mae_test  = mean_absolute_error(y_test, y_predict_test)

# Ausgabe der Fehlermaße
print("Train RMSE:", rmse_train)
print("Test  RMSE:", rmse_test)
print("Train MAE :", mae_train)
print("Test  MAE :", mae_test)
print("=============================================================================================")

# =========================================================================================
# Visualisierung der Ergebnisse
t_train = np.arange(1, y_predict_train.shape[0] + 1)
t_test  = np.arange(split_index + 1, split_index + y_predict_test.shape[0] + 1)

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(t_train, y_t_train, label='Ausgang (Messdaten)', linewidth=1)
plt.plot(t_train, y_predict_train, label='Ausgang (Vorhersage)', linewidth=1, linestyle = '--')
plt.legend()
plt.xlabel('Zeit')
plt.ylabel('Skalierte Daten')
plt.grid()
plt.title('Trainingsdaten')

plt.subplot(2, 1, 2)
plt.plot(t_test, y_test_real, label='Ausgang (Messdaten)', linewidth=1)
plt.plot(t_test, y_predict_test, label='Ausgang (Vorhersage)', linewidth=1, linestyle = '--')
plt.legend()
plt.xlabel('Zeit')
plt.ylabel('Skalierte Daten')
plt.grid()
plt.title('Testdaten')

plt.tight_layout()
plt.show()
