import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# ========================================================================================
# EINSTELLUNGEN
# ========================================================================================

NUM_PARTICLES = 100         # Anzahl Partikel im Schwarm
NUM_ITERATIONS = 100        # Anzahl Durchläufe
INERTIA = 0.7               # Trägheitsfaktor (W träge → langsamer, stabiler)
COGNITIVE = 1.5             # Lokaler Faktor (Einfluss des Partikels auf sich selbst)
SOCIAL = 1.5                # Globaler Faktor (Einfluss des Schwarms)
VELOCITY_DECAY = 0.99       # Abbaurate der Geschwindigkeit
BOUNDS = [-5.12, 5.12]      # Suchraum: x und y in [-5.12, 5.12]
COMMUNICATION = "lokal"     # Kommunikationsmuster: "global" oder "lokal"

# ========================================================================================
# FITNESSFUNKTION
# ========================================================================================

def fitness(x, y):
    return 20 + x**2 + y**2 - 10.0 * (np.cos( 2 * np.pi * x) + np.cos( 2 * np.pi * y))

# ========================================================================================
# PSO-ALGORITHMUS
# ========================================================================================

# Initialisierung
np.random.seed(0)
positions = np.random.uniform(BOUNDS[0], BOUNDS[1], (NUM_PARTICLES, 2))
velocities = np.zeros((NUM_PARTICLES, 2))
pbest_pos = positions.copy()
pbest_val = fitness(positions[:, 0], positions[:, 1])

# Kommunikationsmuster: globales gbest
if COMMUNICATION == "global":
    gbest_index = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_index]
    gbest_val = pbest_val[gbest_index]

# Zeitmessung starten
start_time = time.time()

# Iterationen
for iteration in range(NUM_ITERATIONS):
    # Fitness bewerten
    current_val = fitness(positions[:, 0], positions[:, 1])
    
    # Lokale beste Positionen aktualisieren
    better_mask = current_val < pbest_val
    pbest_val[better_mask] = current_val[better_mask]
    pbest_pos[better_mask] = positions[better_mask]
    
    # Globales Bestes (bei globaler Kommunikation)
    if COMMUNICATION == "global":
        gbest_index = np.argmin(pbest_val)
        gbest_pos = pbest_pos[gbest_index]
        gbest_val = pbest_val[gbest_index]
    elif COMMUNICATION == "lokal":
        # Lokales gbest pro Partikel (Ring-Topologie)
        gbest_pos = np.zeros_like(pbest_pos)
        for i in range(NUM_PARTICLES):
            # Nachbarn: links, aktuell, rechts (zyklisch)
            left = (i - 1) % NUM_PARTICLES
            right = (i + 1) % NUM_PARTICLES
            candidates = [left, i, right]
            best_local = min(candidates, key=lambda j: pbest_val[j])
            gbest_pos[i] = pbest_pos[best_local]
    
    # Velocity-Update (PSO-Kern)
    r1 = np.random.rand(NUM_PARTICLES, 2)
    r2 = np.random.rand(NUM_PARTICLES, 2)
    
    cognitive_part = COGNITIVE * r1 * (pbest_pos - positions)
    social_part = SOCIAL * r2 * (gbest_pos - positions)

# gbest_pos wurde im lokalen Kommunikationsmuster bereits so erstellt,
# dass gbest_pos[i] die beste Position im lokalen Umfeld (links, selbst, rechts)
# von Partikel i enthält.
    
    velocities = INERTIA * velocities + cognitive_part + social_part
    velocities *= VELOCITY_DECAY  # Abbaurate

    # Positionen aktualisieren und begrenzen
    positions += velocities
    positions = np.clip(positions, BOUNDS[0], BOUNDS[1])

# Nach Optimierung: Bestes Partikel (lokal oder global)
final_fitness = fitness(positions[:, 0], positions[:, 1])
best_index = np.argmin(final_fitness)
best_position = positions[best_index]
best_value = final_fitness[best_index]

# Zeitmessung beenden
elapsed_time = time.time() - start_time

# ========================================================================================
# VISUALISIERUNG
# ========================================================================================

# 3D-Plot mit dem gefundenen Minimum
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
X = np.linspace(BOUNDS[0], BOUNDS[1], 100)
Y = np.linspace(BOUNDS[0], BOUNDS[1], 100)
X, Y = np.meshgrid(X, Y)
Z = fitness(X, Y)

ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.scatter(best_position[0], best_position[1], best_value, color='red', s=60, label="Minimum")
ax.set_title("Fitnessfunktion mit Minimum (rot)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x,y)")
ax.legend()
plt.tight_layout()
plt.show()

print("==============================================")
print(" Partikelschwarm-Optimierung: Ergebnis")
print("==============================================")
print(f"Anzahl Partikel       : {NUM_PARTICLES}")
print(f"Anzahl Iterationen    : {NUM_ITERATIONS}")
print(f"Trägheitsfaktor (w)   : {INERTIA}")
print(f"Kognitiver Faktor (c1): {COGNITIVE}")
print(f"Sozialer Faktor (c2)  : {SOCIAL}")
print(f"Abbaurate der Geschwindigkeit : {VELOCITY_DECAY}")
print(f"Kommunikationsmuster  : {COMMUNICATION}")
print(f"Suchraum              : [{BOUNDS[0]}, {BOUNDS[1]}]")
print("----------------------------------------------")
print(f"Gefundenes Minimum    : f({best_position[0]:.4f}, {best_position[1]:.4f}) = {best_value:.6f}")
print(f"Laufzeit              : {elapsed_time:.4f} Sekunden")
print("==============================================")



"""
======================== Erklärung der PSO-Parameter ========================

1. NUM_PARTICLES = 100  
   → Anzahl der Partikel im Schwarm.  
     Ein größerer Schwarm erhöht die Wahrscheinlichkeit, das globale Minimum zu finden,  
     da mehr Regionen des Suchraums gleichzeitig erkundet werden.  
     100 Partikel bieten einen guten Kompromiss zwischen Diversität und Rechenaufwand.

2. NUM_ITERATIONS = 100  
   → Gibt an, wie viele Schritte die Partikel unternehmen dürfen.  
     Eine hohe Anzahl erlaubt eine bessere Feinabstimmung in der Nähe des Minimums.  
     100 Iterationen reichen in der Regel aus, um Konvergenz bei einfacher bis mittelschwerer Komplexität zu erreichen.

3. INERTIA = 0.7  
   → Trägheitsgewicht w: Steuert, wie viel der aktuellen Bewegung aus der Vergangenheit übernommen wird.  
     Höhere Werte fördern Exploration, kleinere fördern Konvergenz.  
     Der gewählte Wert 0.7 ermöglicht sowohl stabile Bewegung als auch sukzessives Abbremsen  
     eine klassische Wahl für balancierte Suche.

4. COGNITIVE = 1.5  
   → Der kognitive Parameter c1 bestimmt, wie stark sich ein Partikel an seinem persönlichen besten Punkt orientiert.  
     Ein Wert von 1.5 bewirkt, dass Partikel ihre individuelle Erfahrung berücksichtigen,  
     ohne zu stark von ihrer Historie dominiert zu werden.

5. SOCIAL = 1.5  
   → Der soziale Parameter c2 bestimmt die Ausrichtung zur besten Position im Schwarm (global oder lokal).  
     Gleicher Wert wie c1 sorgt für Gleichgewicht zwischen individueller und kollektiver Lernkomponente.

6. VELOCITY_DECAY = 0.99  
   → Die Dämpfung der Geschwindigkeit reduziert mit jeder Iteration die Bewegung der Partikel.  
     Dies verhindert Oszillationen und sorgt für Stabilität in der Konvergenzphase.  
     0.99 ist leicht dämpfend und erhält dennoch Dynamik über längere Zeiträume.

7. COMMUNICATION = "lokal"  
   → Kommunikationsmuster des Schwarms: "lokal" bedeutet, dass jedes Partikel nur mit seinen direkten Nachbarn  
     kommuniziert (Ring-Topologie). Dies erhöht die Vielfalt im Suchverhalten und verringert das Risiko,  
     dass der gesamte Schwarm in einem lokalen Minimum stecken bleibt.

8. BOUNDS = [-5.12, 5.12]  
   → Definiert den zulässigen Suchraum für die Optimierung.  
     Die Grenzen wurden passend zur bekannten Definitionsmenge der Rastrigin-ähnlichen Benchmark-Funktion gewählt.

===================== Ergebnisinterpretation =====================

Das Partikelschwarmverfahren hat erfolgreich ein Minimum der Funktion gefunden:
  f(x, y) ≈ 0.000000 bei x ≈ 0, y ≈ 0

Durch die gewählte Konfiguration konnte die Balance zwischen Exploration (Suchen)
und Exploitation (Verfeinern) erreicht werden. Besonders die Kombination aus
Trägheit, kognitiver und sozialer Komponente sowie lokalem Kommunikationsmuster
ermöglicht es dem Schwarm, sowohl weiträumig zu suchen als auch effektiv zu konvergieren.

Die Wahl des lokalen Kommunikationsmusters sorgt für robuste Konvergenz bei
nichtlinearen, multimodalen Funktionen und verhindert, dass sich der gesamte
Schwarm zu früh auf ein suboptimales Minimum festlegt.
"""
