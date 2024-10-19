import os
import graphviz
import numpy as np
import torch
import matplotlib.pyplot as plt
class GraphVis():
    def __init__(self, oid, obj_pred, obj_gt, rel_pred, edge_index, cls_names, rel_names, draw_gt = True) -> None:
        # moeglichen Klassen aus txt file laden
        self.oid = oid
        self.obj_pred = obj_pred 
        self.obj_gt = obj_gt
        self.rel_pred = rel_pred
        self.edge_index = edge_index
        self.draw_gt = draw_gt
        self.cls_names = cls_names
        self.rel_names = rel_names
    
    def draw_graph(self, filtered_oid, caption="Area_6_lounge_1") -> None:
        # erstelle Daten
        # Daten fuer die Objekte
        # nodes
        filtered_obj_pred_id = self.obj_pred[filtered_oid]
        filtered_obj_gt_id = self.obj_gt[filtered_oid]
        #print(f"filtered_obj_pred_id: {filtered_obj_pred_id }")
        #print(f"filtered_obj_gt_id: {filtered_obj_gt_id}")
        cls_names = np.array(self.cls_names)
        rel_names = np.array(self.rel_names)
        filtered_obj_pred_cls_name = np.array(self.cls_names)[filtered_obj_pred_id]
        filtered_obj_gt_cls_name = np.array(self.cls_names)[filtered_obj_gt_id]
        #print(f"filtered_obj_pred_cls_name : {filtered_obj_pred_cls_name}")
        #print(f"filtered_obj_gt_cls_name: {filtered_obj_gt_cls_name}")
        # edges
        edges = [] # Teilmenge von edge_index, die die filtered_oids enthalten
        # ich benoetige die Indizes aus den gefilterten edge_index  
        edge_relationship_triple = [] # (id1, id2, rel_id, rel_name) 
        filtered_oid_set = set(filtered_oid)
        #print(f"type(self.edge_index): {type(self.edge_index)}")
        #print(f"self.edge_index.shape: {self.edge_index.shape}")
        zipped_edge_index = tuple(zip(self.edge_index[0], self.edge_index[1]))
        #print(f"zipped_edge_index: {tuple(zipped_edge_index)}")
        indices = [i for i, (u, v) in enumerate(zipped_edge_index) if u in filtered_oid_set and v in filtered_oid_set]
        print(f"indices: {indices}")
        # src, trgt, label_id, label_name
        for idx in range(len(indices)):
            index = indices[idx]
            src = self.edge_index[0][index]
            trgt = self.edge_index[1][index]
            pred = self.rel_pred[index]
            print(f"pred: {pred}")

        # baue Graph auf
        dot = graphviz.Digraph(caption)
        # Nodes
        for index in range(len(filtered_oid)):
            id = filtered_oid[index]
            #print(f"id instance: {id}")
            obj_pred_cls_name = self.obj_pred[id]
            obj_pred_cls_name = cls_names[obj_pred_cls_name]
            obj_gt_cls_name = self.obj_gt[id]
            obj_gt_cls_name = cls_names[obj_gt_cls_name]
            label = str(id) + ": " + obj_pred_cls_name + " (GT:" + obj_gt_cls_name +")" 
            dot.node(str(id), label=label)
        # Edges
        print(dot.source)

def graphVisualization_extended(oid, node_pred, edge_pred, node_gt, edge_gt, edge_index, scan_id, obj_names, rel_names, file_name = "graphviz.pdf"):
        print(f"inside draw method")
        num_true = torch.sum(edge_pred).item() 
        num_false = edge_pred.numel() - num_true
        print(f"num_true: {num_true}")
        print(f"num_false: {num_false}")
        
        oid = oid.cpu().numpy()
        node_pred = node_pred.cpu().numpy()
        edge_pred = edge_pred.cpu().numpy()
        #node_gt = node_gt.cpu().numpy()
        if edge_gt is not None:
            edge_gt = edge_gt.cpu().numpy()
        edge_index = edge_index.cpu().numpy()
        rel_names = np.array(rel_names)
        #print(f"oid: {oid}")
        print(f"node_pred: {node_pred}")
        oid2idx = [(obj_id, idx) for idx, obj_id in enumerate(oid)]
        idx2oid = [(idx, obj_id) for idx, obj_id in enumerate(oid)]
        #print(f"oid2idx: {oid2idx}")

        color_node_correct_prediction = "green" # falls node predictions übereinstimmen
        color_node_wrong_prediction = "red"
        color_edge_correct_prediction = "green"
        color_edge_partly_correct_prediction = "blue"
        color_edge_missing_prediction = "orange"
        color_edge_wrong_prediction = "red"

        # Graph generieren
        comment = "s3dis: " + scan_id
        dot = graphviz.Digraph(scan_id, comment=comment)  
        # nodes hinzufuegen
        for obj_id, idx in oid2idx:
            # GT Daten
            obj_name_gt = node_gt[idx] #obj_names[node_gt[idx]]
            # Predicted Daten
            obj_name_pred = obj_names[node_pred[idx]]
            text = "ID " + str(obj_id) + ": Pred: " + obj_name_pred + " (GT: " + obj_name_gt + ")"
            if obj_name_pred == obj_name_gt:
                dot.node(str(obj_id), text, style = "filled", color = color_node_correct_prediction)
            else:
                dot.node(str(obj_id), text, style = "filled", color = color_node_wrong_prediction)
        #print(dot)
        # edges hinzufuegen
        num_edges = edge_index.shape[1]
        #print(f"num_edges: {num_edges}")
        # print(f"num_edges: {num_edges}")
        print(f"edge_index.shape : {edge_index.shape}")
        print(f"edge_pred.shape : {edge_pred.shape}")
        print(f"oid : {oid}")
        #rel_cls_predictions  = pred_probab_edge_cls > 0.5
        contains_true = np.any(edge_pred)
        print(f"contains_true: {contains_true}")
        true_count = np.sum(edge_pred)
        print(f"true_count: {true_count}")
        true_indices = np.argwhere(edge_pred)
        print(f"true_indices: {true_indices}")
        print(f"len(true_indices): {len(true_indices)}")
        counter = 0
        if edge_gt is None:
            print(f"i am here inside edge_gt is None")
            for idx in range(num_edges):
                src_idx = edge_index[0, idx]
                trgt_idx = edge_index[1, idx]
                src_id = oid[src_idx]
                trgt_id = oid[trgt_idx]
                #print(f"(src, tgt): {src_id}, {trgt_id}")
                # Prediction Daten
                # if idx == 0:
                #     print(f"edge_pred[0]: {edge_pred[0]}")
                predicate_pred = edge_pred[idx] # holen der Zeile von den 26 Klassenvorhersagen
                #print(f"shape predicate_pred: {predicate_pred.shape}")
                #print(f"predicate_pred: {predicate_pred}")
                indices_pred = np.where(predicate_pred == True)[0]
                # Zeichnen
                num_indices_pred = len(indices_pred)
                if len(indices_pred) > 0: 
                    #print(f"i am here")
                    predicate_names = rel_names[indices_pred]
                    # label_text = "(GT:"
                    label_text = ' '.join(predicate_names) 
                    # label_text += ")"
                    dot.edge(str(src_id), str(trgt_id), label_text)
                    counter += len(predicate_names)
        else: 
            for idx in range(num_edges):
                src_idx = edge_index[0, idx]
                trgt_idx = edge_index[1, idx]
                src_id = oid[src_idx]
                trgt_id = oid[trgt_idx]
                # ab hier gibt es Veraenderungen
                # GT Daten
                predicate_gt = edge_gt[idx] # holen der Zeile von den 26 Klassenvorhersagen
                indices_gt = np.where(predicate_gt == 1)[0]
                # Prediction Daten
                predicate_pred = edge_pred[idx] # holen der Zeile von den 26 Klassenvorhersagen
                indices_pred = np.where(predicate_pred == 1)[0]
                # Zeichnen
                num_indices_gt = len(indices_gt)
                num_indices_pred = len(indices_pred)
                # if num_indices_gt > 0 and num_indices_pred > 0:
                    
                #      pass # fall 1
                
                if num_indices_gt > 0 and num_indices_pred > 0: # Fall 1: Edge in GT und Pred vorhanden
                    predicate_names_gt = rel_names[indices_gt]
                    predicate_names_pred = rel_names[indices_pred]
                    label_text = "Pred: "
                    label_text += ' '.join(predicate_names_pred) 
                    label_text += " (GT: "
                    label_text += ' '.join(predicate_names_gt) 
                    label_text += ")"
                    if sorted(predicate_names_gt) == sorted(predicate_names_pred): # Fall 1.1: alles ist perfekt
                        dot.edge(str(src_id), str(trgt_id), label_text, color=color_edge_correct_prediction)
                    else: # Fall 1.2 es ist nicht unbedingt perfekt
                        dot.edge(str(src_id), str(trgt_id), label_text, color=color_edge_partly_correct_prediction)
                    pass # fall 1
                elif num_indices_gt > 0: # Fall 2: Edge in GT vorhanden, in Pred nicht
                    predicate_names_gt = rel_names[indices_gt]
                    label_text = "Pred: None (GT:"
                    label_text += ' '.join(predicate_names_gt) 
                    label_text += ")"
                    dot.edge(str(src_id), str(trgt_id), label_text, color=color_edge_missing_prediction)
                elif num_indices_pred > 0: # Fall 3: Edge nicht in GT vorhanden, aber in Pred
                    predicate_names_pred = rel_names[indices_pred]
                    label_text = "Pred: "
                    label_text += ' '.join(predicate_names_pred) 
                    label_text += "(GT: None)"
                dot.edge(str(src_id), str(trgt_id), label_text, color=color_edge_wrong_prediction)

            # if len(indices_gt) > 0: 
            #     predicate_names = rel_names[indices_gt]
            #     label_text = "(GT:"
            #     label_text += ' '.join(predicate_names) 
            #     label_text += ")"
            #     dot.edge(str(src_id), str(trgt_id), label_text)
        print(f"Anzahl der Edges: {counter}")
        print(dot)
        dir = "s3dis_manual_annotated" + scan_id + "_graphviz.pdf"
        dot.render(directory=dir, view=True)

def main():
    pth_experiment = "./experiments/custom_dataset/custom_3DSSG_full_l160"
    file_names = ["sample_gt.pt", "obj_logits_gt.pt", "rel_logits_gt.pt", "obj_cls_predictions_gt.pt", "rel_cls_predictions_gt.pt"]
    #file_names = ["sample.pt", "obj_logits.pt", "rel_logits.pt", "obj_cls_predictions.pt", "rel_cls_predictions.pt"]
    sample = torch.load(os.path.join(pth_experiment, file_names[0]))
    #print(f"sample: {sample}")
    obj_logits = torch.load(os.path.join(pth_experiment, file_names[1]))
    rel_logits = torch.load(os.path.join(pth_experiment, file_names[2]))
    rel_logits = rel_logits #/ 50

    logit_values = rel_logits.flatten().cpu().detach().numpy()
    # plt.hist(logit_values, bins=50)
    # plt.title('Logits Distribution')
    # file_path = "./experiments/custom_dataset/custom_3DSSG_full_l160/gt/s3dis_logit_values_gt_hist.png"
    # plt.savefig(file_path, dpi=300, bbox_inches="tight")
    # plt.show()
    min_value = torch.min(rel_logits)
    print("Kleinster Logit Wert Relationship:", min_value.item())
    max_value = torch.max(rel_logits)
    print("Groesster Logit Wert Relationship:", max_value.item())
    mean_value = torch.mean(rel_logits)
    print("Durchschnittswert Logit Wert Relationship:", mean_value.item())

    obj_cls_predictions = torch.load(os.path.join(pth_experiment, file_names[3]))
    rel_cls_predictions = torch.load(os.path.join(pth_experiment, file_names[4]))
    #print(f"rel_logits.shape: {rel_logits.shape}")
    pred_probab_edge_cls = torch.sigmoid(rel_logits) # .mean(0)
    rel_probs = torch.sigmoid(rel_logits)
    prob_values = rel_probs.flatten().cpu().detach().numpy()
    # np.savetxt("./experiments/custom_dataset/custom_3DSSG_full_l160/gt/sigmoid_predictions.txt", prob_values)
    # plt.hist(prob_values, bins=50)
    # plt.title('Sigmoid Probabilities Distribution')
    # file_path = "./experiments/custom_dataset/custom_3DSSG_full_l160/gt/s3dis_sigmoid_prediction_values_gt_hist.png"
    # plt.savefig(file_path, dpi=300, bbox_inches="tight")
    # plt.show()
    min_value = torch.min(rel_probs)
    print("Kleinster Sigmoid Wert Relationship:", min_value.item())
    max_value = torch.max(rel_probs)
    print("Groesster Sigmoid Wert Relationship:", max_value.item())
    mean_value = torch.mean(rel_probs)
    print("Durchschnittswert Sigmoid Wert Relationship:", mean_value.item())
 
    rel_cls_predictions  = pred_probab_edge_cls > 0.5
    num_true = torch.sum(rel_cls_predictions).item() 
    num_false = rel_cls_predictions.numel() - num_true
    print(f"num_true: {num_true}")
    print(f"num_false: {num_false}")

    pred_values_cls, pred_indices_cls = obj_cls_predictions.max(1)
    print(f"pred_indices_cls: {pred_indices_cls}")


    # load classes txt
    pth_data = "./data/custom_data"
    classes_file = "classes.txt"
    relationships_file = "relationships.txt"
    # obj_gt = sample['node']['y'].cpu().numpy()#torch.numpy(sample['node']['y'])
    obj_pred = pred_indices_cls.cpu().numpy() #torch.numpy(pred_indices_cls)
    class_names = []
    with open(os.path.join(pth_data, classes_file), "r") as file:
        for line in file:
            line = line.rstrip().lower()
            class_names.append(line)
    class_names = np.array(class_names)

    classes = class_names[pred_indices_cls.cpu().numpy()]
    print(f"predicted classes: {classes}")
    print(f"len classes: {len(classes)}")

    gt_labels = ['clutter', 'chair', 'ceiling', 'clutter', 'sofa', 'clutter', 'wall',
                  'sofa', 'sofa', 'table', 'clutter', 'wall', 'wall', 'clutter',
                    'clutter', 'clutter', 'table', 'clutter', 'sofa', 'chair',
                      'beam', 'clutter', 'sofa', 'clutter', 'clutter', 'clutter',
                        'ceiling', 'chair', 'wall', 'clutter', 'wall', 'table',
                          'sofa', 'clutter', 'clutter', 'sofa', 'wall', 'table',
                            'table', 'clutter', 'floor', 'table', 'clutter', 
                            'clutter', 'chair', 'chair', 'clutter', 'ceiling',
                              'chair', 'clutter'] # for gt data
    gt_labels_manual = ['lamp', 'chair', 'ceiling', 'lamp', 'sofa', 'lamp', 'wall',
                  'sofa', 'sofa', 'table', 'pillow', 'wall', 'wall', 'lamp',
                    'pillow', 'lamp', 'table', 'lamp', 'sofa', 'chair',
                      'beam', 'tv', 'sofa', 'lamp', 'lamp', 'box',
                        'ceiling', 'cushion', 'wall', 'clutter', 'wall', 'table',
                          'sofa', 'lamp', 'lamp', 'sofa', 'wall', 'table',
                            'stool', 'lamp', 'floor', 'table', 'lamp', 
                            'lamp', 'chair', 'chair', 'lamp', 'ceiling',
                              'cushion', 'lamp'] # for gt data

    relationship_names = []
    with open(os.path.join(pth_data, relationships_file), "r") as file:
        for line in file:
            line = line.rstrip().lower()
            relationship_names.append(line)

    for idx, pair in enumerate(zip(classes, gt_labels)):
        print(f"ID {idx} pair (PRED, GT): {pair}")

    oid = sample['node'].oid
    #node_gt = sample['node'].y # für Inst_Seg aus GT-Daten; gt_labels 
    #node_gt = node_gt.cpu().numpy()
    #class_names_np = np.array(class_names)
    #node_gt = class_names[node_gt]
    node_gt = gt_labels_manual
    print(f"node_gt: {node_gt}")

    edge_index = sample['node', 'to', 'node'].edge_index
    scan_id = sample.scan_id
    edge_gt = None

    # print(f"rel_cls_predictions: {rel_cls_predictions}")
    # contains_true = torch.any(rel_cls_predictions).item()
    # print(f"contains_true: {contains_true}")
    # true_count = torch.sum(rel_cls_predictions).item()
    # print(f"true_count.item(): {true_count}")
    graphVisualization_extended(oid, pred_indices_cls, rel_cls_predictions, node_gt, edge_gt, edge_index, scan_id[0], class_names, relationship_names, file_name = "graphviz.pdf")

    # obj_pred_str = np.array(class_names)[obj_pred]
    # obj_gt_str = np.array(class_names)[obj_gt]
    # compare_obj_pred_gt = zip(obj_gt_str, obj_pred_str)
    # oid = sample['node']['oid'].cpu().numpy()
    # edge_index = sample['node', 'to', 'node']['edge_index'].cpu().numpy()
    # rel_pred = rel_cls_predictions.cpu().numpy()

    #graph_vis_obj = GraphVis(oid, obj_pred, obj_gt, rel_pred, edge_index, class_names, relationship_names, True)

    # oid, obj_pred, obj_gt, rel_pred, edge_index, cls_names, rel_names, draw_gt = True
    #filtered_oid_list = [0, 4, 5, 37, 38, 41, 45] # sehr schlechte Ergebnisse

    #filtered_oid_list = [2, 4, 5, 21, 22, 23]
    #graph_vis_obj.draw_graph(filtered_oid_list)
if __name__ == '__main__':
    main()

