import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# ==================================================================================================
# Funktionen
# ==================================================================================================

def gleitender_mittelwert_allgemein(signal, vergangenheit=4, zukunft=0):
    """
    Berechnet den gleitenden Mittelwert über (vergangenheit + 1 + zukunft) Punkte.
    Alle Ränder, die nicht vollständig gefüllt werden können, erhalten NaN.
    
    Args:
        signal (np.array): Das Signal (1D).
        vergangenheit (int): Anzahl der vergangenen Werte (z. B. 4).
        zukunft (int): Anzahl der zukünftigen Werte (z. B. 0 für rein rückwärtsgerichtet).
    
    Returns:
        np.array: Das geglättete Signal (NaN an Rändern).
    """
    window_size = vergangenheit + zukunft + 1
    result = np.full_like(signal, fill_value=np.nan, dtype=np.float64)
    
    for i in range(vergangenheit, len(signal) - zukunft):
        window = signal[i - vergangenheit : i + zukunft + 1]
        result[i] = np.mean(window)
    
    return result

# =================================================================================================

def scale_to_range(signal, target_min=-1, target_max=1):
    """
    Skaliert ein 1D-Signal in einen beliebigen Wertebereich [target_min, target_max].
    
    Args:
        signal (np.array): Das zu skalierende Signal.
        target_min (float): Untere Grenze des Zielbereichs.
        target_max (float): Obere Grenze des Zielbereichs.
    
    Returns:
        np.array: Das skalierte Signal.
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    
    # Falls alle Werte gleich sind, auf das Mittel setzen (Vermeidung von Division durch 0)
    if min_val == max_val:
        return np.full_like(signal, fill_value=(target_max + target_min) / 2)

    scaled = (signal - min_val) / (max_val - min_val)  # [0, 1]
    scaled = scaled * (target_max - target_min) + target_min  # auf Zielbereich strecken
    return scaled

# ==================================================================================================

def create_arx_matrix(y, u, ny=2, nu=2, delay=4):
    """
    Erstellt Regressionsmatrix für ein ARX-Modell.
    
    y: Ausgangszeitreihe (z. B. y_scaled)
    u: Eingang (z. B. u_scaled)
    ny: Anzahl vergangener y-Werte
    nu: Anzahl vergangener u-Werte
    delay: Eingang-Verzögerung
    """
    N = len(y)
    X = []
    Y = []

    for t in range(max(ny, nu + delay), N):
        row = []

        # vergangene y-Werte
        for i in range(1, ny + 1):
            row.append(y[t - i])

        # vergangene u-Werte (mit delay)
        for j in range(delay, delay + nu):
            row.append(u[t - j])

        X.append(row)
        Y.append(y[t])
    
    return np.array(X), np.array(Y)

# ==================================================================================================

# --------------------------------------------------------------------------------------------------

# ==================================================================================================
# Einlesen und validieren
# ==================================================================================================

# Einlesen der Eingangsdaten (u.csv)
u_df = pd.read_csv('u.csv', header=None)    # header = none sorgt dafür dass erste Zeile nicht als header genommen wird (Weil es keinen gibt)
u = u_df.values.flatten()  # In 1D-Array umwandeln

# Einlesen der Ausgangsdaten (y.csv)
y_df = pd.read_csv('y.csv', header=None)
y = y_df.values.flatten()  # In 1D-Array umwandeln

# Ausgabe zur Kontrolle
print("u:", u[:10])  # Zeige erste 10 Werte
print("y:", y[:10])
print("=============================================================================================")

# ==================================================================================================
# Glätten und entstehende NaN Werte löschen
# ==================================================================================================

u_glatt = gleitender_mittelwert_allgemein(u, 4, 0)
y_glatt = gleitender_mittelwert_allgemein(y, 4, 0)

# Entfernt alle Positionen, bei denen u_glatt oder y_glatt NaN ist (Zeitliche Verschiebung um 4 Werte!)
valid_mask = ~np.isnan(u_glatt) & ~np.isnan(y_glatt)

u_glatt_clean = u_glatt[valid_mask]
y_glatt_clean = y_glatt[valid_mask]


# Ausgabe zur Kontrolle
print("u_glatt_clean:", u_glatt_clean[:8])
print("y_glatt_clean:", y_glatt_clean[:8])
print("=============================================================================================")

# ==================================================================================================
# Skalieren auf beliebige Werte (Hier auf -1/1)
# ==================================================================================================

u_scaled = scale_to_range(u_glatt_clean, -1, 1)
y_scaled = scale_to_range(y_glatt_clean, -1, 1)

# Ausgabe zur Kontrolle
print("u_scaled:", u_scaled[:5])
print("y_scaled:", y_scaled[:5])
print("=============================================================================================")

# ==================================================================================================
# ARX-Modell trainieren und Vorhersage machen und plotten
# ==================================================================================================

# ARX-Modell erstellen
X, Y = create_arx_matrix(y_scaled, u_scaled, ny=2, nu=2, delay=4)

# Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# Modell trainieren
model = LinearRegression()
model.fit(X_train, y_train)

# Vorhersage
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Fehler berechnen
rmse_train = root_mean_squared_error(y_train, y_pred_train)
rmse_test = root_mean_squared_error(y_test, y_pred_test)

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

print("Train RMSE:", rmse_train)
print("Test RMSE :", rmse_test)
print("Train MAE :", mae_train)
print("Test MAE  :", mae_test)

# Ergebnis plotten
plt.figure(figsize=(12, 5))
plt.plot(y_test, label='Tatsächlicher Ausgang (y_test)', linewidth=2)
plt.plot(y_pred_test, label='Vorhersage (ŷ_test)', linestyle='--', linewidth=2)
plt.title("ARX-Modell: Vorhersage vs. Messung")
plt.xlabel("Zeit (Samples)")
plt.ylabel("Skalierter Ausgang")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

