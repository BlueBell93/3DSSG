import os
import logging
import codeLib
import ssg
import ssg.config as config
from ssg.checkpoints import CheckpointIO
import cProfile
import matplotlib
import torch
import torch_geometric
import networkx as nx
import torch
from torch import nn
import codeLib.utils.string_numpy as snp
import json
import torch
import graphviz
import numpy as np
import matplotlib.pyplot as plt

# disable GUI
matplotlib.pyplot.switch_backend('agg')
# change log setting
matplotlib.pyplot.set_loglevel("CRITICAL")
logging.getLogger('PIL').setLevel('CRITICAL')
logging.getLogger('trimesh').setLevel('CRITICAL')
logging.getLogger("h5py").setLevel(logging.INFO)
logger_py = logging.getLogger(__name__)



def get_scanid2idx_dict(dataset):
    """"
    Generates a dictionary that maps scan ids to their corresponding indices in the dataset

    Args:
        dataset(object): an object of type ssg.dataset.SGFNDataset like dataset_train = config.get_dataset(cfg, 'train')

    Returns:
        dict: a dict where keys are scan ids and values are the corresponding indices
                in the dataset
    """
    scanid2idx_dict = dict()
    for index in range(len(dataset)):
        current_scan_id = snp.unpack(dataset.scans, index)  # self.scans[idx]
        scanid2idx_dict[current_scan_id] = index
    return scanid2idx_dict


def generate_sample_info(scan_id, dataset):
    """
        Generates and saves sample information as a JSON file for a given scan ID from a dataset.
    """
    scanid2idx_dict = get_scanid2idx_dict(dataset)
    index = scanid2idx_dict[scan_id]
    sample = dataset.__getitem__(index)

    print(sample)

    sample_info = {}
    sample_info["node"] = {}
    sample_info["node"]["x"] = (sample['node'].x).tolist()
    sample_info["node"]["y"] = (sample['node'].y).tolist()
    sample_info["node"]["oid"] = (sample['node'].oid).tolist()
    sample_info["node"]["pts"] = (sample['node'].pts).tolist()
    sample_info["node"]["desp"] = (sample['node'].desp).tolist()

    print(f"sample_info[node][x]: {sample['node'].x}")
    print(f"sample_info[node][y]: {sample['node'].y}")
    print(f"sample_info[node][oid]: {sample['node'].oid}")
    print(f"sample_info[node][pts]: {sample['node'].pts}")
    print(f"sample_info[node][desp]: {sample['node'].desp}")

    sample_info["node_gt"] = {}
    sample_info["node_gt"]["x"] = (sample['node_gt'].x).tolist()
    sample_info["node_gt"]["clsIdx"] = sample['node_gt'].clsIdx

    print(f"sample_info[node_gt][x]: {sample['node_gt'].x}")
    print(f"sample_info[node_gt][clsIdx]: {sample['node_gt'].clsIdx}")

    sample_info["edge"] = {}
    sample_info["edge"]["pts"] = (sample['edge'].pts).tolist()

    print(f"sample_info[edge][pts]: {sample['edge'].pts}")

    sample_info["node_gt to node"] = {}
    sample_info["node_gt to node"]["edge_index"] = (sample['node_gt', 'to', 'node'].edge_index).tolist()

    print(f"sample_info[node_gt to node][edge_index]: {sample['node_gt', 'to', 'node'].edge_index}")

    sample_info["node_gt to node_gt"] = {}
    sample_info["node_gt to node_gt"]["clsIdx"] = sample['node_gt', 'to', 'node_gt'].clsIdx
    sample_info["node_gt to node_gt"]["edge_index"] = (sample['node_gt', 'to', 'node_gt'].edge_index).tolist()

    print(f"sample_info[node_gt to node_gt][clsIdx]: {sample['node_gt', 'to', 'node_gt'].clsIdx}")
    print(f"sample_info[node_gt to node_gt][edge_index]: {sample['node_gt', 'to', 'node_gt'].edge_index}")
    

    sample_info["node to node"] = {}
    sample_info["node to node"]["edge_index"] = (sample['node', 'to', 'node'].edge_index).tolist()
    sample_info["node to node"]["y"] = (sample['node', 'to', 'node'].y).tolist()

    print(f"sample_info[node to node][edge_index]: {sample['node', 'to', 'node'].edge_index}")
    print(f"sample_info[node to node][y]: {sample['node', 'to', 'node'].y}")

def graphVisualization_extended(oid, node_pred, edge_pred, node_gt, edge_gt, edge_index, scan_id, obj_names, rel_names, file_name = "graphviz.pdf"):
    """
    Visualizes a scene graph using Graphviz, comparing the predicted and ground truth (GT) object classes and 
    relationships between objects. Nodes represent objects, and edges represent relationships. Colors indicate 
    whether predictions match the ground truth.

    Args:
        oid (torch.Tensor): Tensor containing object IDs for each node in the graph. Transferred from GPU to CPU.
        node_pred (torch.Tensor): Tensor containing predicted class indices for each object. Transferred from GPU to CPU.
        edge_pred (torch.Tensor): Tensor containing predicted relationship labels (binary vector) for each edge. 
                                Transferred from GPU to CPU.
        node_gt (torch.Tensor): Tensor containing ground truth class indices for each object. Transferred from GPU to CPU.
        edge_gt (torch.Tensor): Tensor containing ground truth relationship labels (binary vector) for each edge. 
                                Transferred from GPU to CPU.
        edge_index (torch.Tensor): Tensor defining the source and target nodes for each edge (shape: [2, num_edges]).
                                Transferred from GPU to CPU.
        scan_id (str): Identifier for the scan/scene being visualized.
        obj_names (list of str): List of object class names, used for labeling the nodes with their predicted and GT classes.
        rel_names (list of str): List of relationship class names, used for labeling edges with their predicted and GT relationships.
        file_name (str, optional): Output PDF file name. Default is "graphviz.pdf".

    Returns:
        None: The graph is saved as a PDF and optionally viewed in the default viewer.

    Visualization Details:
        - Nodes represent objects and are labeled with both the predicted and GT class names.
        - The color of each node indicates whether the predicted class matches the GT class:
            - Green: Correct prediction (node matches GT).
            - Default: Incorrect prediction.
        - Edges represent relationships between objects and are labeled with both the predicted and GT relationship names.
        - The color of each edge indicates the match between predicted and GT relationships:
            - Green: Predicted relationships fully match the GT relationships.
            - Blue: Partial match between predicted and GT relationships.
            - Orange: GT relationship exists, but no prediction.
            - Red: Incorrect prediction (relationship predicted where none exists in GT).
        - The graph can be printed or saved as a PDF with the given file name.

    Notes:
        - Only edges with at least one predicted or ground truth relationship are displayed.
        - The graph is rendered in memory and printed to the console; file rendering is currently commented out.
    """
    
    oid = oid.cpu().numpy()
    node_pred = node_pred.cpu().numpy()
    edge_pred = edge_pred.cpu().numpy()
    node_gt = node_gt.cpu().numpy()
    edge_gt = edge_gt.cpu().numpy()
    edge_index = edge_index.cpu().numpy()
    rel_names = np.array(rel_names)
    oid2idx = [(obj_id, idx) for idx, obj_id in enumerate(oid)]
    idx2oid = [(idx, obj_id) for idx, obj_id in enumerate(oid)]
    #print(f"oid: {oid}")
    #print(f"node_pred: {node_pred}")
    #print(f"oid2idx: {oid2idx}")

    color_node_correct_prediction = "green" # falls node predictions übereinstimmen
    color_node_wrong_prediction = "red" # falls node predictions nicht übereinstimmen
    color_edge_correct_prediction = "green" # falls edge predictions übereinstimmen
    color_edge_no_correct_prediction = "black" # falls edge in GT und Pred vorhanden sind, aber nicht übereinstimmen
    color_edge_partly_correct_prediction = "blue" # falls die edge prediction teilweise übereinstimmt mit GT-Daten
    color_edge_missing_prediction = "orange" # falls edge nicht vorhergesagt wird, es sie aber gibt
    color_edge_wrong_prediction = "red" # falls eine edge, die es nach GT Daten nicht gibt, existiert

    # Graph generieren
    comment = "3DSSG: " + scan_id
    dot = graphviz.Digraph(scan_id, comment=comment)  
    # nodes hinzufuegen
    for obj_id, idx in oid2idx: 
        # GT Daten
        obj_name_gt = obj_names[node_gt[idx]]
        # Predicted Daten
        obj_name_pred = obj_names[node_pred[idx]]
        text = "ID " + str(obj_id) + ": Pred: " + obj_name_pred + " (GT: " + obj_name_gt + ")"
        if obj_name_pred == obj_name_gt:
            dot.node(str(obj_id), text, style = "filled", color = color_node_correct_prediction)
        else:
            dot.node(str(obj_id), text, style = "filled", color = color_node_wrong_prediction)
    # edges hinzufuegen
    num_edges = edge_index.shape[1]
    for idx in range(num_edges):
        src_idx = edge_index[0, idx]
        trgt_idx = edge_index[1, idx]
        src_id = oid[src_idx]
        trgt_id = oid[trgt_idx]
        # GT Daten
        predicate_gt = edge_gt[idx] # holen der Zeile von den 26 Klassenvorhersagen
        indices_gt = np.where(predicate_gt == 1)[0]
        # Prediction Daten
        predicate_pred = edge_pred[idx] # holen der Zeile von den 26 Klassenvorhersagen
        indices_pred = np.where(predicate_pred == 1)[0]
        # Zeichnen
        num_indices_gt = len(indices_gt)
        num_indices_pred = len(indices_pred)
        if num_indices_gt > 0 and num_indices_pred > 0: # Fall 1: Edge in GT und Pred vorhanden
                predicate_names_gt = rel_names[indices_gt]
                predicate_names_pred = rel_names[indices_pred]
                label_text = "Pred: "
                label_text += ', '.join(predicate_names_pred) 
                label_text += " (GT: "
                label_text += ', '.join(predicate_names_gt) 
                label_text += ")"
                if sorted(predicate_names_gt) == sorted(predicate_names_pred): # Fall 1.1: alles ist perfekt
                    dot.edge(str(src_id), str(trgt_id), label_text, color=color_edge_correct_prediction)
                elif set(predicate_names_gt).intersection(set(predicate_names_pred)): # Fall 1.2 es ist nicht unbedingt perfekt, aber mind. eine Vorhersage
                    dot.edge(str(src_id), str(trgt_id), label_text, color=color_edge_partly_correct_prediction)
                else:
                    dot.edge(str(src_id), str(trgt_id), label_text, color=color_edge_no_correct_prediction) # edge in GT und Pred vorhanden, aber keine korrekte Prediction
                pass # fall 1
        elif num_indices_gt > 0: # Fall 2: Edge in GT vorhanden, in Pred nicht
            predicate_names_gt = rel_names[indices_gt]
            label_text = "Pred: None (GT:"
            label_text += ', '.join(predicate_names_gt) 
            label_text += ")"
            dot.edge(str(src_id), str(trgt_id), label_text, color=color_edge_missing_prediction)
        elif num_indices_pred > 0: # Fall 3: Edge nicht in GT vorhanden, aber in Pred
            predicate_names_pred = rel_names[indices_pred]
            label_text = "Pred: "
            label_text += ', '.join(predicate_names_pred) 
            label_text += " (GT: None)"
            dot.edge(str(src_id), str(trgt_id), label_text, color=color_edge_wrong_prediction)
    print(dot)
    dir = "3dssg_" + scan_id + "_graphvis_with_gt" #+ file_name # "_graphviz.pdf"
    dot.render(directory=dir, view=True) # auskommentieren, falls gespeichert werden soll


def graphVisualization(oid, node_gt, edge_gt, edge_index, scan_id, obj_names, rel_names, file_name = "graphviz.pdf"):
    """
    Visualizes a graph of object relationships in a scene using Graphviz, based on the ground truth object classes and 
    relationships. The output is saved as a PDF file.

    Args:
        oid (torch.Tensor): Tensor containing object IDs for each node in the graph. Transferred from GPU to CPU.
        node_gt (torch.Tensor): Tensor containing ground truth class indices for each object. Transferred from GPU to CPU.
        edge_gt (torch.Tensor): Tensor containing ground truth relationship labels (binary vector) for each edge. 
                                Transferred from GPU to CPU.
        edge_index (torch.Tensor): Tensor defining the source and target nodes for each edge (shape: [2, num_edges]).
                                Transferred from GPU to CPU.
        scan_id (str): Identifier for the scan/scene being visualized.
        obj_names (list of str): List of object class names, used for labeling the nodes with their ground truth classes.
        rel_names (list of str): List of relationship class names, used for labeling edges with their ground truth relationships.
        file_name (str, optional): Output PDF file name. Default is "graphviz.pdf".

    Returns:
        None: The graph is saved as a PDF and optionally viewed in the default viewer.
    
    Notes:
        - The function visualizes each object (node) with its ID and ground truth class.
        - The edges (relationships between objects) are labeled with the ground truth relationship class, if any.
        - Only the edges where at least one ground truth relationship exists are displayed.
        - The output file is saved as a PDF, and the graph is optionally opened for viewing.
    """
    
    oid = oid.cpu().numpy()
    node_gt = node_gt.cpu().numpy()
    edge_gt = edge_gt.cpu().numpy()
    edge_index = edge_index.cpu().numpy()
    rel_names = np.array(rel_names)
    #print(f"oid: {oid}")
    oid2idx = [(obj_id, idx) for idx, obj_id in enumerate(oid)]
    idx2oid = [(idx, obj_id) for idx, obj_id in enumerate(oid)]
    #print(f"oid2idx: {oid2idx}")

    # Graph generieren
    comment = "3DSSG: " + scan_id
    dot = graphviz.Digraph(scan_id, comment=comment)  
    # nodes hinzufuegen
    for obj_id, idx in oid2idx:
        obj_name = obj_names[node_gt[idx]]
        text = "ID " + str(obj_id) + ": (GT: " + obj_name + ")"
        dot.node(str(obj_id), text)
    #print(dot)
    # edges hinzufuegen
    num_edges = edge_index.shape[1]
    #print(f"num_edges: {num_edges}")
    for idx in range(num_edges):
        src_idx = edge_index[0, idx]
        trgt_idx = edge_index[1, idx]
        src_id = oid[src_idx]
        trgt_id = oid[trgt_idx]
        predicate = edge_gt[idx] # holen der Zeile von den 26 Klassenvorhersagen
        indices = np.where(predicate == 1)[0]
        # if len(indices) == 0:
        #     print(f"no edge_connection")
        if len(indices) > 0: 
            predicate_names = rel_names[indices]
            label_text = "(GT:"
            label_text += ' '.join(predicate_names) 
            label_text += ")"
            dot.edge(str(src_id), str(trgt_id), label_text)
        # if idx == 0:
        #     print(f"predicate_names: {predicate_names}")
    #print(dot)
    dir = "3rscan_" + scan_id + "_gt_graphviz.pdf"
    dot.render(directory=dir, view=True)

def main():
    # Train Dataset
    #scan_id = "7ab2a9c7-ebc6-2056-8973-f34c559f7e0d"
    #scan_id = "bf9a3de7-45a5-2e80-81a4-fd6126f6417b"
    #scan_id = "f62fd5fd-9a3f-2f44-883a-1e5cf819608e"
    scan_id = "6a36053f-fa53-2915-9579-3938283bc154"
    #scan_id = "baf0a8fb-26d4-2033-8a28-2001356bbb9a"
    #scan_id = '0cac75e8-8d6f-2d13-8fc4-acdbf00437c8'
    #scan_id = 'c7895f78-339c-2d13-82bb-cc990cbbc90f'
    # scan_id='b05fdd96-fca0-2d4f-88c3-d9dfda85c00e'

    # Test Dataset
    #scan_id = 'f2c76fe5-2239-29d0-8593-1a2555125595' # hier stimmt es überein # das hier habe ich genommen als Beispiel fuer die Praesentation
    #scan_id = 'c7895f2b-339c-2d13-8248-b0507e050314' # hier sind die objekt predictions nicht so gut, habe ich auch genommen
 
    cfg = ssg.Parse()
    # Shorthands
    out_dir = os.path.join(cfg['training']['out_dir'], cfg.name) # stores modes.pth and the eval results like confusion matrix
    # Output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # set random seed
    codeLib.utils.util.set_random_seed(cfg.SEED)
    cfg.data.load_cache = False
    n_workers = cfg['training']['data_workers']
    ''' create dataset and loaders '''
    dataset_train = config.get_dataset(cfg, 'train')
    dataset_test = config.get_dataset(cfg, 'test')
    dataset_val = config.get_dataset(cfg, 'validation')
    # logger_py.info('create loader')
    test_loader = torch_geometric.loader.DataLoader( # dataloader vor test dataset
        dataset_test,
        batch_size=cfg.training.batch,
        num_workers=n_workers,
        pin_memory=True
    )
    train_loader = torch_geometric.loader.DataLoader( # dataloader vor train dataset
        dataset_train,
        batch_size=cfg.training.batch,
        num_workers=n_workers,
        pin_memory=True
    )
    val_loader = torch_geometric.loader.DataLoader( # dataloader vor val dataset
        dataset_val, batch_size=1, num_workers=n_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )
    ''' Create model '''
    relationNames = dataset_train.relationNames
    classNames = dataset_train.classNames
    num_obj_cls = len(dataset_train.classNames)
    num_rel_cls = len(
         dataset_train.relationNames) if relationNames is not None else 0
    print(f"classNames: {classNames}")
    print(f"relationNames: {relationNames}")
    model = config.get_model(
         cfg, num_obj_cls=num_obj_cls, num_rel_cls=num_rel_cls)
    checkpoint_io = CheckpointIO(out_dir, model=model)
    load_dict = checkpoint_io.load('model_best.pt', device=cfg.DEVICE) # dict_keys(['epoch_it', 'it', 'loss_val_best', 'patient', 'optimizer', 'scheduler', 'smoother'])
    it = load_dict.get('it', -1)
    # Anfang einkommentieren, falls man Infos zum Modell und zu den Objekt/Prädikatsklassen in der Konsole möchte
    # print("--------------------------------------------------------------------------------------------------------------")
    # print(f"SGPN Model {model}")
    # print(f"num_obj_cls = {num_obj_cls}, num_rel_cls = {num_rel_cls}")
    # print()
    # print(f"relationNames = {relationNames}")
    # print()
    # print(f"classNames = {classNames}")
    # print("--------------------------------------------------------------------------------------------------------------")
    # Ende einkommentieren, für Modell- und Klassenvorhersagen
    
    #print("Inference Single Sample")
    #generate_sample_info(scan_id, dataset_train) # Zeile auskommentieren, falls man HeteroData-Objekt ausgegeben haben moechte in Konsole
    pr = cProfile.Profile()
    pr.enable()
    scanid2idx_dict = get_scanid2idx_dict(dataset_train)
    model.eval()
    #print(f"Laenge Dataset Train {len(dataset_train)}")
    with torch.no_grad():
        sample_index = scanid2idx_dict[scan_id]
        sample = dataset_train.__getitem__(sample_index)
        print(f"sample: {sample}")
        sample = sample.to(cfg.DEVICE)
        cls_pred, rel_pred = model(sample)
        # print(f"cls_pred: {cls_pred}")
        # print(f"rel_pred: {rel_pred}")
        # print(f"cls_pred.shape: {cls_pred.shape}")
        # print(f"rel_pred.shape: {rel_pred.shape}")
    # prediction classes of the objects from logits
    softmax = nn.Softmax(dim=1)
    pred_probab_cls = torch.softmax(cls_pred, dim=1)#.mean(dim=0)
    pred_values_cls, pred_indices_cls = pred_probab_cls.max(1)
    # prediction classes of predicates from logits
    pred_probab_edge_cls = torch.sigmoid(rel_pred)
    pred_edge_cls = pred_probab_edge_cls > 0.5

    # Visualisierung der logits
    logit_values = rel_pred.flatten().cpu().detach().numpy()
    plt.hist(logit_values, bins=50)
    plt.title('Logits Distribution of 3DSSG Scan with id '+ scan_id)
    file_path = "./experiments/logit_values_relationship_hist_3dssg_" + scan_id + ".png"
    print(f"file_path: {file_path}")
    #plt.savefig(file_path, dpi=300, bbox_inches="tight")
    # plt.show()
    min_value = torch.min(rel_pred)
    print("Kleinster Logit Wert Relationship:", min_value.item())
    max_value = torch.max(rel_pred)
    print("Groesster Logit Wert Relationship:", max_value.item())
    mean_value = torch.mean(rel_pred)
    print("Durchschnittswert Logit Wert Relationship:", mean_value.item())
    
    # Visualisierung der predictions nach sigmoid
    #rel_probs = torch.sigmoid(rel_pred)
    prob_values = pred_probab_edge_cls.flatten().cpu().detach().numpy()
    file_path = "./experiments/sigmoid_prediction_values_relationship_hist_3dssg_" + scan_id + ".png"
    plt.hist(prob_values, bins=50)
    plt.title('Sigmoid Probabilities Distribution')
    #plt.savefig(file_path, dpi=300, bbox_inches="tight")
    # plt.show()
    min_value = torch.min(pred_probab_edge_cls)
    print("Kleinster Sigmoid Wert Relationship:", min_value.item())
    max_value = torch.max(pred_probab_edge_cls)
    print("Groesster Sigmoid Wert Relationship:", max_value.item())
    mean_value = torch.mean(pred_probab_edge_cls)
    print("Durchschnittswert Sigmoid Wert Relationship:", mean_value.item())
    num_true = torch.sum(pred_edge_cls).item() 
    num_false = pred_edge_cls .numel() - num_true
    print(f"num_true: {num_true}")
    print(f"num_false: {num_false}")

    #graphVisualization(sample['node'].oid, sample['node'].y, sample['node', 'to', 'node'].y, sample['node', 'to', 'node'].edge_index, sample.scan_id, classNames, relationNames)    
    
    graphVisualization_extended(sample['node'].oid, pred_indices_cls, pred_edge_cls, sample['node'].y, sample['node', 'to', 'node'].y, sample['node', 'to', 'node'].edge_index, sample.scan_id, classNames, relationNames)

    # print("Print Class Predictions Nodes##################################################################################")
    # print(f"cls_pred {cls_pred}")
    # cls_size = cls_pred.size
    # cls_shape = cls_pred.shape
    # #print(f"prediction_classes after softmax of oid=0: {pred_probab_cls[0]}")
    # print(f"node.pd: {pred_probab_cls}")
    # print(f"cls_pred size = {cls_size}, shape = {cls_shape}")
    # print()
    # print(f"Predicted values and indizes of classes")
    # print(f"values: {pred_values_cls}")
    # print(f"indices: {pred_indices_cls}")
    # print(f"Ground Truth Class Indizes: {sample['node_gt'].clsIdx}")
    # print(f"Class Indizes node.y: {sample['node'].y}")
    # print(f"Object ID node.oid: {sample['node'].oid}")
    # print(f"Compare indices with ground truth (node.y):")
    # result = torch.eq(pred_indices_cls, sample['node'].y)
    # print(result)
    # print(f"Number of values where prediction and ground truth is equal: {torch.sum(result)}")
    # print(f"Total number of nodes: {result.numel()}")
    # print(f"Proportion of correct predictions = {torch.sum(result)/result.numel()}")

    # print()
    # print("Print Predicate Predictions Edges##################################################################################")
    # print(f"rel_pred {rel_pred}")
    # rel_size = rel_pred.size
    # rel_shape = rel_pred.shape
    # print(f"rel_pred size = {rel_size}, shape = {rel_shape}")
    # print(f"pred_probab_edge_cls: {pred_probab_edge_cls}")
    # print(f"Prediction edge classes: {pred_edge_cls}")
    # print(f"Ground Truth: {sample['node','to','node'].y}")
    # print(f"Ground Truth Size: {sample['node','to','node'].y.size()}")
    # result_edge = torch.eq(pred_edge_cls, sample['node','to','node'].y)
    # print(result_edge)
    # numbers_right = torch.sum(result_edge)
    # print(numbers_right)
    # print(f"Total numbers: {result_edge.numel()}")
    # print(f"Proportion: {torch.sum(result_edge).item()/result_edge.numel()}")
    # print(f"Edge index node to node: {sample['node', 'to', 'node'].edge_index}")
    # print(f"Edge index node_gt to node: {sample['node_gt', 'to', 'node'].edge_index}")
    # print(f"test {torch.sum(pred_edge_cls, axis=1)}, size: {torch.sum(pred_edge_cls, axis=1).size()}")
    # print(f"END#######################################################################################################")


    #     for X in val_loader:
    #         print(X)
    #         print("node y")
    #         print(f"{X['node'].y}")
    #         print("Ground Truth class Indices")
    #         print(f"{X['node_gt'].clsIdx}")
    #         X = X.to(cfg.DEVICE)
    #         cls_pred, rel_pred = model(X)
    #         #predictions = model(X)
    #         break

    # print("--------------------------------------------------------------------------------------------------------------")
    # #print(f"cls_pred {cls_pred}")
    # #print(f"rel_pred {rel_pred}")
    # print(f"cls_pred {cls_pred}")
    # cls_size = cls_pred.size
    # cls_shape = cls_pred.shape
    # print(f"cls_pred size = {cls_size}, shape = {cls_shape}")
    # #print(f"type predictions {type(predictions)}") # ein Tupel, also wie erwartet
    # rel_size = rel_pred.size
    # rel_shape = rel_pred.shape
    # print(f"rel_pred {rel_pred}")
    # print(f"rel_pred size = {rel_size}, shape = {rel_shape}")
    # # logits
    # softmax = nn.Softmax(dim=1)
    # pred_probab_cls = softmax(cls_pred)
    # #print(pred_probab_cls)
    # print(f"tensor dim {pred_probab_cls.dim()}")
    # print(f"tensor dim {pred_probab_cls.shape}")
    # pred_values_cls, pred_indices_cls = torch.max(pred_probab_cls, dim=1)
    # pred_probab_rel = softmax(rel_pred)
    # pred_values_rel, pred_indices_rel = torch.max(pred_probab_rel, dim=1)
    # print(f"Predicted values and indizes of classes")
    # print(f"values: {pred_values_cls}")
    # print(f"indices: {pred_indices_cls}")
    # #print(f"Predicted values and indizes")
    # print("DONE")

if __name__ == '__main__':
    main()
