# Szenengraphvisualisierung
- Umsetzung mittels graphviz python library

## 3DSSG
- mittels des vom repo bereitgestellten DataLoaders wurden die 
Daten geladen und anschließend eine Inferenz ausgeführt und 
die Klassenvorhersagen der Objekte und Prädikate berechnet
- Graphvisualisierung mit Predictions und GT-Daten (Ordner mit Namen old_vis wurden folgendermaßen visualisiert):
  - rot: falsche Knotenvorhersage
  - grün: korrekte Knotenvorhersage
  - grün: korrekte Kantenvorhersage
  - rot: falsche Kantenvorhersage; Kante ist in GT-Daten nicht vorhangen
  - orange: fehlende Kantenvorhersage: keine Kante vorhergesagt, aber nach GT ist Kante vorhanden
  - blau: Kante in GT und Prediction vorhanden, aber Vorhersage nicht zwangsläufig korrekt
- Graphvisualisierung mit Predictions und GT-Daten:
  - neue fein-granularere Visualisierung, da zwischen zwei Fällen unterschieden wird bei der 
  Kantenvorhersage: Kante wurde vorhergesagt in GT und Pred, aber die Klassenpredictions stimmen nicht
  überein vs. Kante wurde vorhergesagt in GT und Pred und mind. eine Prediction stimmt mit GT überein
  - rot: falsche Knotenvorhersage
  - grün: korrekte Knotenvorhersage
  - grün: korrekte Kantenvorhersage
  - rot: falsche Kantenvorhersage; Kante ist in GT-Daten nicht vorhangen
  - orange: fehlende Kantenvorhersage: keine Kante vorhergesagt, aber nach GT ist Kante vorhanden
  - blau: Kante in GT und Prediction vorhanden und mindestens eine Vorhersage stimmt überein
  - schwarz: Kante in GT und Prediction vorhanden, aber Klassenvorhersagen stimmen in keiner einzigen Vorhersage mit GT überein 

Code zum Ausführen von Inferenz, Berechnungen für Klassenvorhersagen und
zur Szenengraph-Visualisierung von graphviz:
```
python inference_single_sample.py --config configs/config_3DSSG_full_l160.yaml 
```
- Methode, um Graph zu visualisieren mit GT-Daten: **graphVisualization_extended** in *inference_single_sample.py*

### Inferenz auf Testdaten
- Inferenz auf Testdaten
- ID: f2c76fe5-2239-29d0-8593-1a2555125595
  - Visualisierung des Graphen: 3DSSG/scene_graph_visualization/3dssg_f2c76fe5-2239-29d0-8593-1a2555125595_graphvis_with_gt
  - 10 von 22 Objektvorhersagen stimmen
  - es gibt eine alte Visualisierung (Ordner beginnt mit old_vis) und eine neue Visualisierung
- ID: c7895f2b-339c-2d13-8248-b0507e050314
  - keine so guten node predictions, dafür gute edge-predictions (wenn edge vorhergesagt wurde)
  - nur neue Visualisierung vorhanden

### Inferenz auf Trainingsdaten
- ID: scan_id = "7ab2a9c7-ebc6-2056-8973-f34c559f7e0d"
- ID: scan_id = "bf9a3de7-45a5-2e80-81a4-fd6126f6417b"
- ID: scan_id = "f62fd5fd-9a3f-2f44-883a-1e5cf819608e"