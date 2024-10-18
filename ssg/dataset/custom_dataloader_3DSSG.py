import os

import torch
import torch.utils.data as data
from torch_geometric.data import HeteroData

import numpy as np

from ssg import define

class Custom3DSSGDataset(data.Dataset):
    def __init__(self, config, mode) -> None:
        """
        Initialisiert das Custom3DSSGDataset.

        Diese Methode liest die Konfigurationsparameter und initialisiert die
        Attribute für den Dataset. Sie lädt die Klassennamen, Beziehungsnamen 
        und die ausgewählten Scans aus den entsprechenden Dateien. Die 
        Konfiguration wird genutzt, um verschiedene Dataset-Parameter zu 
        setzen, darunter die maximale Anzahl an Kanten und die Dimensionen 
        der Punkte.

        Parameter:
        -----------
        config : object
            Konfigurationsobjekt, das Parameter für das Dataset enthält, 
            einschließlich der maximalen Anzahl an Kanten.

        mode : str
            Modus, in dem das Dataset betrieben wird (z.B. 'train', 'test', 
            'val'). Dies kann für spätere Anpassungen des Verhaltens des 
            Datasets verwendet werden.

        Attributes:
        -----------
        cfg : object
            Die Konfiguration für das Dataset.

        path : str
            Der Pfad zu dem Verzeichnis, das die Datendateien enthält.

        pcd_file_ending : str
            Dateiendung für die Punktwolken-Dateien.

        test_scans_file : str
            Name der Datei, die die ausgewählten Scans enthält.

        multi_rel_outputs : bool
            Flag, das angibt, ob Mehrfachbeziehungen in den Daten erlaubt 
            sind.

        max_edges : int
            Maximale Anzahl an Kanten im Graphen.

        dim_pts : int
            Die Dimensionen der Punkte (standardmäßig 3 für 3D-Punkte).

        relationship_names : list
            Eine sortierte Liste von Beziehungsnamen.

        class_names : list
            Eine sortierte Liste von Klassennamen.

        selected_scans : list
            Eine Liste der ausgewählten Scans, die geladen werden.
        """ 
        super().__init__()
        '''read config'''
        self.cfg = config
        self.mode = mode
        self.path = self.cfg.data.path #"./data/custom_data"
        self.pcd_file_ending = self.cfg.data.pcd_file_ending #"_pred_extended_postprocessed.txt"
        self.test_scans_file = self.cfg.data.test_scans_file  # "test_scans.txt" 
        self.obj_gt_data_file = "instance_gt.txt"# TODO in config nennen
        self.multi_rel_outputs = self.cfg.model.multi_rel #True
        self.max_edges = self.cfg.data.max_num_edge #512 # self.cfg.data.max_num_edge
        self.node_feature_dim = self.cfg.model.node_feature_dim # 256
        self.edge_feature_dim = self.cfg.model.edge_feature_dim # 256
        self.dim_pts = 3
        '''read classes'''
        classes_path = os.path.join(self.path, 'classes.txt')
        relationships_path = os.path.join(self.path, 'relationships.txt')

        class_names = []
        with open(classes_path, "r") as file:
            for line in file:
                line = line.rstrip().lower()
                class_names.append(line)

        relationship_names = []
        with open(relationships_path, "r") as file:
            for line in file:
                line = line.rstrip().lower()
                relationship_names.append(line)
        
        if not self.multi_rel_outputs:
            if define.NAME_NONE not in relationship_names:
                relationship_names.append(define.NAME_NONE)
        elif define.NAME_NONE in relationship_names: # entfernt None als Relationship, da multi_rel erlaubt ist
            relationship_names.remove(define.NAME_NONE)

        self.relationship_names = sorted(relationship_names)
        self.class_names = sorted(class_names)
        '''load test scans'''
        selected_scans_path = os.path.join(self.path, self.test_scans_file) 
        self.selected_scans = []
        with open(selected_scans_path, "r") as file:
            for line in file:
                line = line.strip()
                self.selected_scans.append(line)
        
    
    def __len__(self):
        """
        Gibt die Anzahl der ausgewählten Scans im Dataset zurück.

        Returns:
        --------
        int
            Die Anzahl der in `selected_scans` gespeicherten Scans.
        """
        return len(self.selected_scans) 

    def __getitem__(self, idx):
        # Schritt 1: ply file anhand des idx auslesen als np Array
        scan_name = self.selected_scans[idx]
        self.pcd_file_ending = "_gt.txt" # wieder spaeter auskommentieren, dass ist die perfekt instanzsegmentierte pointcloud (gt)
        path_scan_pcd = os.path.join(self.path, scan_name, scan_name + self.pcd_file_ending) 
        # print(f"scan_name: {scan_name}")
        # print(f"path_scan_pcd: {path_scan_pcd}")
        pcd_np = np.loadtxt(path_scan_pcd)
        # print(f"pcd_np: {pcd_np}")
        # print(f"pcd_np.shape: {pcd_np.shape}")

        # Schritt 2: np Array so bearbeiten, dass es passt (also alles separat speichern und z.B. als int konvertieren)
        pcd_pts = pcd_np[:, :3].copy()
        pcd_colors = pcd_np[:, 3:6].astype(np.uint8).copy()
        pcd_sem_label = pcd_np[:, 6].astype(np.uint8).copy()
        pcd_inst_label = pcd_np[:, 7].astype(np.uint8).copy()
        available_sem_labels = set(pcd_sem_label.flatten())
        oid = set(pcd_inst_label.flatten())
        num_instances = len(oid)
        # print(f"pcd_pts.shape: {pcd_pts.shape}")
        #print(f"pcd_pts.dtype: {type(pcd_pts)}")
        # print(f"pcd_colors.shape: {pcd_colors.shape}")
        # print(f"pcd_colors: {pcd_colors}")
        # print(f"pcd_sem_label.shape: {pcd_sem_label.shape}")
        # print(f"pcd_inst_label.shape: {pcd_inst_label.shape}")
        # print(f"available_sem_labels: {available_sem_labels}")
        # print(f"len(available_sem_labels): {len(available_sem_labels)}")
        # print(f"oid: {oid}")
        # print(f"len(oid): {len(oid)}")

        # Schritt 3: node: Objekte auslesen + pro Objekt: Transforms auf Punkte
        #   gehe ueber jedes Objekt
        #       hole die Punkte von dem Objekt
        #       berechne bboxes
        #       reduziere Punkte auf z.B. 256
        #       Transformiere Punkte (Normierung, Zentrum)
        #       speichere Punkt ab in einem np array
        node_oid = np.array(list(oid))
        # print(f"node_oid: {node_oid}")
        # print(f"type(node_oid): {type(node_oid)}")
        node_pts, bboxes = self.__sample_points(pcd_pts, pcd_inst_label, node_oid)
        # bboxes: Liste aus Bounding Boxen; pro Objekt Eintrag mit [min_bbox, max_bbox]
        # node_pts: Dimension num_obj x 256 x 3
        #           pro Objekt 256 Punkte mit xyz-Koordinaten

        # Schritt 4: edge_index: Edges erstellen (alle Möglichkeiten, keine edge zu sich selbst)
        edge_index = self.__sample_3D_node_edges(node_oid)

        
        # Schritt 5: node to node: pro Edge: Transforms auf Punkte
        rel_pts = self.__sample_rel_points(node_oid, bboxes, edge_index, pcd_pts, pcd_inst_label)

        # Schritt 6: zufaelliges droppen von edges
        #rel_pts_filtered, edge_index_filtered = self.__drop_edges(edge_index, rel_pts) # einkommentieren, wenn man Filterung haben moechte
        rel_pts_filtered = rel_pts # no filter
        edge_index_filtered = np.array(edge_index) # no filter
        
        # prüfen, ob es mehr als 512 edges gibt
        #   wenn ja: waehle 512 edges aus
        #           hierfuer 512 zufaellige Indizes auswaehlen
        #           dann aus rel_pts diese nehmen
        #           und dann noch die entsprechenden edge_indizes auswaehlen
        
        # Schritt 7: edge_index in richtige Form bringen
        # TODO: edge_index wirklich in edge_index Form umwandeln

        # Schritt 8: GT Daten Objekte 
        # Beginn GT Daten: auskommentieren, falls man braucht
        # inst_gt_data_file = os.path.join(self.path, scan_name, self.obj_gt_data_file)
        # inst_gt_data = []
        # with open(inst_gt_data_file, "r") as file:
        #     for line in file:
        #         line = line.rstrip().lower()
        #         inst_gt_data.append(self.class_names.index(line))
        # inst_gt_data = np.array(inst_gt_data)
        # inst_gt_data = torch.from_numpy(inst_gt_data)
        # Ende GT Datan: alles in dem Abschnitt bis hierhin auskommentieren, falls man GT Daten braucht
        

        # return: HeteroData Objekt mit (vorerst) node(pts, oid), edge_index, node to node (pts)
        # Schritt 9: node, edge_index, node to node als Tensor transformieren und in ein 
        #            HeteroData-Objekt packen
        ''''Store Data in HeteroData Object'''
        pcd = HeteroData()
        pcd['scan_id'] = scan_name
        #print(f"type(node_pts): {type(node_pts)}")
        pcd['node'].x = torch.zeros(len(node_oid), 1) # dummy
        pcd['node'].oid = torch.from_numpy(node_oid)
       #pcd['node'].y = inst_gt_data # auskommentieren, wenn man gt daten hat
        pcd['node'].pts = node_pts
        pcd['edge'].pts = rel_pts_filtered
        pcd['node', 'to', 'node'].edge_index = torch.from_numpy(edge_index_filtered).t().contiguous().to(torch.long) 
        #print(f"pcd: {pcd}")
        # print(f"pcd['node'].oid: {pcd['node'].oid}")
        #print(f"pcd['node'].y: {pcd['node'].y}")
        #print(f"edge_index_filtered: {edge_index_filtered}")
        #print(f"pcd['node', 'to', 'node'].edge_index: {pcd['node', 'to', 'node'].edge_index}")
        #print(f"type(pcd['node'].pts): {type(pcd['node'].pts)}")
        #print(f"pcd['node'].oid): {pcd['node'].oid}")
        #print(f"type(pcd['edge'].pts): {type(pcd['edge'].pts)}")
        #print(f"type(pcd['node', 'to', 'node'].edge_index): {type(pcd['node', 'to', 'node'].edge_index)}")
        return pcd

    def __sample_points(self, pcd_pts, pcd_inst_label, node_oid):
        # Schritt 3: node: Objekte auslesen + pro Objekt: Transforms auf Punkte
        #   gehe ueber jedes Objekt
        #       hole die Punkte von dem Objekt
        #       berechne bboxes
        #       reduziere Punkte auf z.B. 256
        #       Transformiere Punkte (Normierung, Zentrum)
        #       speichere Punkt ab in einem np array
        num_objects = len(node_oid)
        bboxes = list()
        node_pts = torch.zeros(num_objects, self.node_feature_dim, self.dim_pts)
        #print(f"node_pts.shape: {node_pts.shape}")
        for obj_id in node_oid: # iteriere ueber jedes Objekt
            '''Punktmenge fuer das Ojekt mit ID obj_id bestimmen'''
            #print(f"obj: {obj}")
            mask_obj = pcd_inst_label[:] == obj_id # Maske fuer die Punktmenge des Objekts
            #print(f"result: {result}")
            obj_pts = pcd_pts[mask_obj, :].copy() # extrahiere Punktemenge, die zum Objekt gehoert
            '''Bounding Box (min und max) bestimmen'''
            min_bbox = np.min(obj_pts, axis=0)
            max_bbox = np.max(obj_pts, axis=0)
            bboxes.append([min_bbox, max_bbox])
            '''Reduziere/Erhoehe Anzahl der Punkte pro Objekt auf die Anzahl von self.feature_node_dim'''
            # waehle zufaellig 256 viele Punkte aus
            #print(f"obj_pts.shape: {obj_pts.shape}")
            indices = None
            if obj_pts.shape[0] > self.node_feature_dim:
                indices = np.random.choice(obj_pts.shape[0], size=self.node_feature_dim, replace=False)
            elif obj_pts.shape[0] < self.node_feature_dim:
                indices = np.random.choice(obj_pts.shape[0], size=self.node_feature_dim, replace=True)
            #print(f"indices: {len(indices)}")
            obj_pts = torch.from_numpy(obj_pts[indices,:])
            #print(f"node_pts: {node_pts}")
            '''Transformation der Punkte'''
            obj_pts = self.norm_tensor(obj_pts)
            node_pts[obj_id] = obj_pts
            #print(f"node_pts[obj_id].shape: {node_pts[obj_id].shape}")
        node_pts = node_pts.permute(0, 2, 1)
        return (node_pts, bboxes)
    
    def norm_tensor(self, points): # Funktion aus dataloader_SGFN.p
        assert points.ndim == 2
        assert points.shape[1] == 3
        centroid = torch.mean(points, dim=0)  # N, 3
        points -= centroid  # n, 3, npts
        # find maximum distance for each n -> [n]
        furthest_distance = points.pow(2).sum(1).sqrt().max()
        points /= furthest_distance
        return points
    
    def __sample_3D_node_edges(self, node_oid):
        edge_index = []
        for src in node_oid: # src: source node
            for trgt in node_oid: # trgt: target node
                if src != trgt:
                    edge = (src, trgt)
                    edge_index.append(edge)
        # print(f"edge_index: {edge_index}")
        # print(f"len(edge_index): {len(edge_index)}")
        return edge_index
    
    def __sample_rel_points(self, node_oid, bboxes, edge_index, pcd_pts, pcd_inst_label): 
        num_edges = len(edge_index)
        rel_pts = torch.zeros(num_edges, self.edge_feature_dim, 4)
        pts = pcd_pts.copy()
        edge_counter = 0
        for edge_idx in range(len(edge_index)):
            edge = edge_index[edge_idx]
            #print(f"edge: {edge}")
            src_oid = edge[0]
            trgt_oid = edge[1]
            src_min_bbox = bboxes[src_oid][0]
            src_max_bbox = bboxes[src_oid][1]
            trgt_min_bbox = bboxes[trgt_oid][0]
            trgt_max_bbox = bboxes[trgt_oid][1]
            #src_pts = pcd_pts[pcd_inst_label[:] == src_oid, :].copy()
            #trgt_pts = pcd_pts[pcd_inst_label[:] == trgt_oid, :].copy()
            src_mask = pcd_inst_label[:] == src_oid
            src_mask = src_mask.astype(np.int32) * 1 # src hat ID 1
            trgt_mask = pcd_inst_label[:] == trgt_oid
            trgt_mask = trgt_mask.astype(np.int32)* 2 # trgt hat ID 2
            # alles andere hat ID 0
            #src_mask = np.array([1 if x == 1 else 0 for x in src_mask]) # too slow
            #trgt_mask = np.array([2 if x == 1 else 0 for x in trgt_mask]) # too slow
            mask = np.expand_dims(src_mask + trgt_mask, axis=1)
            # if src_oid == 0:
            #     print(f"mask: {mask}")
            '''alle Punkte bestimmen, die sich innerhalb der BBoxen befinden und ihnen korrektes Label zuweisen'''
            min_box = np.minimum(src_min_bbox, trgt_min_bbox)
            max_box = np.maximum(src_max_bbox, trgt_max_bbox)
            #pts_with_mask = np.concatenate((pts, mask), axis=1)
            #print(f"pts_with_mask: {pts_with_mask}")
            filter_mask = (pts[:, 0] > min_box[0]) * (pts[:, 0] < max_box[0]) \
                * (pts[:, 1] > min_box[1]) * (pts[:, 1] < max_box[1]) \
                * (pts[:, 2] > min_box[2]) * (pts[:, 2] < max_box[2]) 
            #pts_filtered = pts_with_mask[filter_mask, :]
            points4d = np.concatenate([pts, mask], axis=1)
            #print(f"filter_mask: {filter_mask}")
            pointset = points4d[filter_mask, :]
            #pts_within_joint_bboxes =     
            '''256 Punkte auswaehlen (mit ihren labeln) aus dieser Punktmenge'''
            #print(f"pointset.shape: {pointset.shape}")
            num_pts = pointset.shape[0]
            indices = None
            if num_pts < self.edge_feature_dim:
                indices = np.random.choice(num_pts, self.edge_feature_dim, replace=True)
            else: 
                indices = np.random.choice(num_pts, self.edge_feature_dim, replace=False)
            selected_pointset = pointset[indices, :]
            selected_pointset = torch.from_numpy(selected_pointset)
            '''Normalisieren'''
            selected_pointset[:, :3] = self.zero_mean(selected_pointset[:, :3], False)
            rel_pts[edge_counter] = selected_pointset
            #print(f"rel_pts[edge_counter]: {rel_pts[edge_counter]}")
            #print(f"rel_pts[edge_counter].shape: {rel_pts[edge_counter].shape}")
            #rel_pts.append(selected_pointset)
            edge_counter += 1
        rel_pts = rel_pts.permute(0, 2, 1)
        #print(f"rel_pts.shape: {rel_pts.shape}")
        return rel_pts
    
    def zero_mean(self, point, normalize: bool):# Funktion aus dataloader_SGFN.py
        mean = torch.mean(point, dim=0)
        point -= mean.unsqueeze(0)
        ''' without norm to 1  '''
        if normalize: # wird nicht ausgefuehrt
            print(f"normalize is true")
            # find maximum distance for each n -> [n]
            furthest_distance = point.pow(2).sum(1).sqrt().max()
            point /= furthest_distance
        return point
    
    def __drop_edges(self, edge_index, rel_pts):
        # Schritt 6: zufaelliges droppen von edges
        # prüfen, ob es mehr als 512 edges gibt
        #   wenn ja: waehle 512 edges aus
        #           hierfuer 512 zufaellige Indizes auswaehlen
        #           dann aus rel_pts diese nehmen
        #           und dann noch die entsprechenden edge_indizes auswaehlen
        num_edges = len(edge_index)
        if num_edges > self.max_edges: 
            indices = np.random.choice(num_edges, self.max_edges, replace=False)
            indices = np.sort(indices)
            #print(f"type(indices): {type(indices)}")
            #print(f"type(edge_index): {type(edge_index)}")
            edge_index = np.array(edge_index)
            edge_index_filtered = edge_index[indices]
            rel_pts_filtered = rel_pts[indices, :, :]
            #print(f"rel_pts_filtered.shape: {rel_pts_filtered.shape}")
            #print(f"len(edge_index_filtered): {len(edge_index_filtered)}")
            #print(f"len")
            #print(f"rel_pts: {rel_pts}")
            #print(f"edge_index: {edge_index}")
            return rel_pts_filtered, edge_index_filtered
        else:
            return rel_pts, edge_index
        