# 3D Semantic Scene Graph Estimations
Ergänzungen zur bestehenden README.

## data
Im Ordner Data befinden sich auch die eigenen Datensätze (Punktwolken mit Instanzsegmentierung z.B. von S3DIS).
- data/custom_data: Ordner
  - enthält classes.txt, relationships.txt
  - enthält test_scans.txt: Liste mit Namen der Ordner, die die Punktwolke enthalten
- data/custom_data/Area_6_lounge_1: Ordner mit Punktwolke aus S3DIS-Datensatz
  - Area_6_lounge_1_pred_extended_postprocessed.txt: Punktwolke mit eigener Instanzsegmentierung
  - instance_gt.txt: Punktwolke mit GT Instanzsegmentierung

Für weitere eigene Datensätze: Hinzufügen eines Ordners, Inhalt
ist z.B. Punktwolke mit eigener Instanzsegmentierung, die mit
``_pred_extended_postprocessed.txt`` endet

## experiments
Neben den vortrainierten Modellen befinden sich auch die Ergebnisse der Inferenzen
von 3DSSG, S3DIS (und potentiell auch den eigenen Daten).

ToDo:
- hier auch die Ergebnisse der Szenengraph-Generierung ablegen
## experiments/custom_dataset/custom_3DSSG_full_l160
- enthält vortrainierte Modell
- hier werden die Ergebnisse von batch_inference.py abgelegt, 
fall pth_experiment in batch_inference.py entsprechend gesetzt wurde


| Dateiname | Beschreibung |
|----------|----------|
| sample.pt    | HeteroDataBatch-Objekt   | 
| obj_cls_predictions.pt    | Objekt-Klassenvorhersagen nach Anwendung von Softmax   | 
| obj_logits.pt    | Logits der Objekt-Klassenvorhersagen   | 
| rel_cls_predictions.pt    | Prädikat-Klassenvorhersagen nach Anwendung von Sigmoid und Schwellwert   |
| rel_logits.pt    | Logits der Prädikat-Klassenvorhersagen   | 

- dropped_edges: wenn zufällig Kanten verworfen werden im DataLoader, sodass es lediglich 512 Kanten gibt (s3dis sample mit eigener Instanzsegmentierung)
- no_dropped_edges_bad_results: keine gedroppten Kanten, s3dis sample mit eigener Instanzsegmentierung
- no_dropped_edges_gt_bad_results: s3dis sample mit GT Instanzsegmentierung

## batch_inference.py
- führt Inferenz auf eigenen Datensatz aus
- nutzt eigenen Dataloader 'custom_dataloader_3DSSG.py'
- nutzt eigene Config-File
Zum Ausführen des Codes:
```
python batch_inference.py --config configs/config_custom_3DSSG_full_l160.yaml
```

## ssg/dataset
Enthält einen eigenen Dataloader namens 'custom_dataloader_3DSSG.py'.
Zur Inferenz eigener Datensätze (z.B. von S3DIS-Instanzsegmentierung). 
- init-params:
  - cfg: Konfigurationsdatei z.B. 'config configs/config_custom_3DSSG_full_l160.yaml'
  - mode: z.B. eval
  - gt_inst_seg: True, falls Punktwolke mit GT Instanzsegmentierung geladen werden soll, by default: false
- Punktwolke mit GT Instanzsegmentierung: erwartete Dateiendung '_gt.txt'
- Punktwolken werden als txt-Datei erwartet, als numpy-array einlesbar,
mit fortlaufender Instanzsegmentierung (von 0 bis N als IDs bei (N+1) Instanzen)


## configs
Enthält angepasste config-Dateien für den S3DIS-Datensatz (entsprechende Dateien enthalten
ein custom im Namen). Wurzelkonfigurationsdatei ist `configs/config_custom_3DSSG_full_l160.yaml'. 

## Visualisierung von Punktwolken
Mittels Befehl 'python visualize_pointcloud.py'. Visualisiert Punktwolke (mit und ohne
Bounding Boxen) aus 3RScan-Datensatz durch Open3D.
Ändern von 'scan_id', um eine andere Punktwolke zu visualisieren.
- Problem: Bounding Boxen stimmen mit Punktwolke nicht überein

