# 3D Semantic Scene Graph Estimations
Ergänzungen zur bestehenden README.

## data
Im Ordner Data befinden sich auch die eigenen Datensätze (Punktwolken mit Instanzsegmentierung z.B. von S3DIS).

## experiments
Neben den vortrainierten Modelle nbefinden sich auch die Ergebnisse der Inferenzen
von 3DSSG, S3DIS (und potentiell auch den eigenen Daten).

ToDo:
- hier auch die Ergebnisse der Szenengraph-Generierung ablegen

## ssg/dataset
Enthält einen eigenen Dataloader namens 'custom_dataloader_3DSSG.py'.
Zur Inferenz eigener Datensätze (z.B. von S3DIS-Instanzsegmentierung). 

## configs
Enthält angepasste config-Dateien für den S3DIS-Datensatz (entsprechende Dateien enthalten
ein custom im Namen). Wurzelkonfigurationsdatei ist 'configs/config_custom_3DSSG_full_l160.yaml'. 

## Visualisierung von Punktwolken
Mittels Befehl 'python visualize_pointcloud.py'. Visualisiert Punktwolke (mit und ohne
Bounding Boxen) aus 3RScan-Datensatz durch Open3D.
Ändern von 'scan_id', um eine andere Punktwolke zu visualisieren.
- Problem: Bounding Boxen stimmen mit Punktwolke nicht überein
