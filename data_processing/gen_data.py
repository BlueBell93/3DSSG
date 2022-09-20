if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
from collections import defaultdict
import os,json,trimesh, argparse
import open3d as o3d
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import util_ply, util_label, util, define
from utils.util_search import SAMPLE_METHODS,find_neighbors
import h5py,ast
import copy
import logging

def Parser(add_help=True):
    parser = argparse.ArgumentParser(description='Process some integers.', formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                     add_help=add_help)
    parser.add_argument('--type', type=str, default='train', choices=['train', 'test', 'validation'], help="allow multiple rel pred outputs per pair",required=False)
    parser.add_argument('--label_type', type=str,default='ScanNet20', 
                        choices=['3RScan', '3RScan160', 'NYU40', 'Eigen13', 'RIO27', 'RIO7','ScanNet20'], help='label',required=False)
    parser.add_argument('--pth_out', type=str,default='../data/tmp', help='pth to output directory',required=True)
    parser.add_argument('--relation', type=str,default='relationships', choices=['relationships_extended', 'relationships'])
    
    parser.add_argument('--target_scan', type=str, default='', help='path to a txt file that contains a list of scan ids that you want to use.')
    
    parser.add_argument('--scan_name', type=str, default='inseg.ply', 
                        help='what is the name of the output filename of the ply generated by your segmentation method.')
    
    # options
    parser.add_argument('--mapping',type=int,default=1,
                        help='map label from 3RScan to label_type. otherwise filter out labels outside label_type.')
    parser.add_argument('--v2', type=int,default=1,
                        help='v2 version')
    parser.add_argument('--inherit', type=int,default=1,help='inherit relationships from the 3RScan.')
    parser.add_argument('--verbose', type=bool, default=False, help='verbal',required=False)
    parser.add_argument('--debug', type=int, default=0, help='debug',required=False)
    parser.add_argument('--scale', type=float,default=1,help='scaling input point cloud.')
    
    # neighbor search parameters
    parser.add_argument('--search_method', type=str, choices=['BBOX','KNN'],default='BBOX',help='How to split the scene.')
    parser.add_argument('--radius_receptive', type=float,default=0.5,help='The receptive field of each seed.')
    
    # split parameters
    parser.add_argument('--split', type=int,default=0,help='Split scene into groups.')
    parser.add_argument('--radius_seed', type=float,default=1,help='The minimum distance between two seeds.')
    parser.add_argument('--min_segs', type=int,default=5,help='Minimum segments for each segGroup')
    parser.add_argument('--split_method', type=str, choices=['BBOX','KNN'],default='BBOX',help='How to split the scene.')
    
    # Correspondence Parameters
    parser.add_argument('--max_dist', type=float,default=0.1,help='maximum distance to find corresopndence.')
    parser.add_argument('--min_seg_size', type=int,default=512,help='Minimum number of points of a segment.')
    parser.add_argument('--corr_thres', type=float,default=0.5,help='How the percentage of the points to the same target segment must exceeds this value.')
    parser.add_argument('--occ_thres', type=float,default=0.75,help='2nd/1st must smaller than this.')
    
    # constant
    parser.add_argument('--segment_type', type=str,default='INSEG')
    return parser

debug = True
debug = False

def generate_groups(cloud:trimesh.points.PointCloud, distance:float=1, bbox_distance:float=0.75, 
                    min_seg_per_group = 5, segs_neighbors=None):
    # print('bounds:',cloud.bounds)
    # print('extents', cloud.extents)
    points = np.array(cloud.vertices.tolist())
    segments = cloud.metadata['ply_raw']['vertex']['data']['label'].flatten()
    seg_ids = np.unique(segments)
    selected_indices = list()
    
    index = np.random.choice(range(len(points)),1)
    selected_indices.append(index)
    should_continue = True
    while should_continue:
        distances_pre=None
        for index in selected_indices:
            point = points[index]
            distances = np.linalg.norm(points[:,0:2]-point[:,0:2],axis=1) # ignore z axis.
            if distances_pre is not None:
                distances = np.minimum(distances, distances_pre)
            distances_pre = distances
        selectable = np.where(distances > distance)[0]
        if len(selectable) < 1: 
            should_continue=False
            break
        index = np.random.choice(selectable,1)
        selected_indices.append(index)
        
    if args.verbose:print('num of selected point seeds:',len(selected_indices))

    
    if debug:
        seg_colors = dict()
        for index in seg_ids:
            seg_colors[index] = util.color_rgb(util.rand_24_bit())
        counter=0
    '''Get segment groups'''
    seg_group = list()
    
    ''' Building Box Method '''  
    from enum import Enum
    class SAMPLE_METHODS(Enum):
        BBOX=1
        RADIUS=2
    if args.split_method == 'BBOX':
        sample_method = SAMPLE_METHODS.BBOX
    elif args.split_method == 'KNN':
        sample_method = SAMPLE_METHODS.RADIUS
    
    if sample_method == SAMPLE_METHODS.BBOX:
        for index in selected_indices:
            point = points[index]
            min_box = (point-bbox_distance)[0]
            max_box = (point+bbox_distance)[0]
            
            filter_mask = (points[:,0] > min_box[0]) * (points[:,0] < max_box[0]) \
                            * (points[:,1] > min_box[1]) * (points[:,1] < max_box[1]) \
                            * (points[:,2] > min_box[2]) * (points[:,2] < max_box[2])
                            
            filtered_segments = segments[np.where(filter_mask > 0)[0]]
            segment_ids = np.unique(filtered_segments) 
            # print('segGroup {} has {} segments.'.format(index,len(segment_ids)))
            if len(segment_ids) < min_seg_per_group: continue
            seg_group.append(segment_ids.tolist())
            
            if debug:
                '''Visualize the segments involved'''
                cloud.visual.vertex_colors = [0,0,0,255]
                for segment_id in segment_ids:
                    segment_indices = np.where(segments == segment_id )[0]
                    for idx in segment_indices:
                        cloud.visual.vertex_colors[idx][:3] = seg_colors[segment_id]
                cloud.export('tmp'+str(counter)+'.ply')
                counter+=1
    elif sample_method == SAMPLE_METHODS.RADIUS:
        radknn = 0.1
        n_layers = 2
        trees = dict()
        segs  = dict()
        bboxes = dict()
        for idx in seg_ids:
            segs[idx] = points[np.where(segments==idx)]
            trees[idx] = o3d.geometry.KDTreeFlann(segs[idx].transpose())
            bboxes[idx] = [segs[idx].min(0)-radknn,segs[idx].max(0)+radknn]
        
        # search neighbor for each segments
        if segs_neighbors is None:
            segs_neighbors = find_neighbors(points, segments, search_method,receptive_field=args.radius_receptive)
        def cat_neighbors(idx:int, neighbor_list:dict):
            output = set()
            for n in neighbor_list[idx]:
                output.add(n)
            return output
        
        for idx in selected_indices:
            seg_id =segments[idx][0]
            neighbors = set()
            neighbors.add(seg_id)
            nn_layers = dict()
            for i in range(n_layers):
                nn_layers[i] = set()
                for j in neighbors:
                    new_nn = cat_neighbors(j, segs_neighbors)
                    nn_layers[i] = nn_layers[i].union(new_nn)
                neighbors = neighbors.union(nn_layers[i])
            
            # print(idx, nn_layers)
            for i in range(n_layers):
                for j in range(i+1, n_layers):
                    nn_layers[j] = nn_layers[j].difference(nn_layers[i])
            # print(idx, nn_layers)
            
            if len(neighbors) < min_seg_per_group: continue
            seg_group.append(neighbors)
            
            if debug:
                '''Visualize the segments involved'''
                cloud.visual.vertex_colors = [0,0,0,255]
                for segment_id in neighbors:
                    segment_indices = np.where(segments == segment_id )[0]
                    for idx in segment_indices:
                        cloud.visual.vertex_colors[idx][:3] = seg_colors[segment_id]
                cloud.export('tmp'+str(counter)+'.ply')
                counter+=1
    return seg_group

def process(pth_3RScan, scan_id,label_type,
            target_relationships:list,
            gt_relationships:dict=None, verbose=False,split_scene=True) -> list:
    pth_pd = os.path.join(pth_3RScan,scan_id,args.scan_name)
    if args.v2:
        pth_gt = os.path.join(pth_3RScan,scan_id,'labels.instances.align.annotated.v2.ply')
    else:
        pth_gt = os.path.join(pth_3RScan,scan_id,'labels.instances.align.annotated.ply')
    segseg_file_name = 'semseg.v2.json' if args.v2 else 'semseg.json'

    # some params
    max_distance = args.max_dist
    filter_segment_size = args.min_seg_size # if the num of points within a segment below this threshold, discard this
    filter_corr_thres = args.corr_thres # if percentage of the corresponding label must exceed this value to accept the correspondence
    filter_occ_ratio = args.occ_thres
    
    # load segments
    cloud_pd = trimesh.load(pth_pd, process=False)
    cloud_pd.apply_scale(args.scale)
    points_pd = np.array(cloud_pd.vertices.tolist())
    segments_pd = cloud_pd.metadata['ply_raw']['vertex']['data']['label'].flatten()
    # get num of segments
    segment_ids = np.unique(segments_pd) 
    segment_ids = segment_ids[segment_ids!=0]
    
    if args.verbose: print('filtering input segments.. (ori num of segments:',len(segment_ids),')')
    segments_pd_filtered=list()
    for seg_id in segment_ids:
        pts = points_pd[np.where(segments_pd==seg_id)]
        if len(pts) > filter_segment_size:
            segments_pd_filtered.append(seg_id)
    segment_ids = segments_pd_filtered
    if args.verbose: print('there are',len(segment_ids), 'segemnts:\n', segment_ids)
    
    # Find neighbors of each segment
    segs_neighbors = find_neighbors(points_pd, segments_pd, search_method,receptive_field=args.radius_receptive,selected_keys=segment_ids)
    if args.verbose:
        print('segs_neighbors:\n',segs_neighbors.keys())
    if split_scene:
        seg_groups = generate_groups(cloud_pd,args.radius_seed,args.radius_receptive,args.min_segs,
                                     segs_neighbors=segs_neighbors)
        if args.verbose:
            print('final segGroups:',len(seg_groups))
    else:    
        seg_groups = None

    # load gt
    cloud_gt = trimesh.load(pth_gt, process=False)
    points_gt = np.array(cloud_gt.vertices.tolist()).transpose()
    segments_gt = util_ply.get_label(cloud_gt, '3RScan', 'Segment').flatten()

    _, label_name_mapping, _ = util_label.getLabelMapping(args.label_type)
    pth_semseg_file = os.path.join(pth_3RScan, scan_id, segseg_file_name)
    instance2labelName = util.load_semseg(pth_semseg_file, label_name_mapping,args.mapping)
    
    '''extract object bounding box info'''
    # objs_obbinfo=dict()
    # with open(pth_semseg_file) as f: 
    #     data = json.load(f)
    # for group in data['segGroups']:
    #     obb = group['obb']
    #     obj_obbinfo = objs_obbinfo[group["id"]] = dict()
    #     obj_obbinfo['center'] = copy.deepcopy(obb['centroid'])
    #     obj_obbinfo['dimension'] = copy.deepcopy(obb['axesLengths'])
    #     obj_obbinfo['normAxes'] = copy.deepcopy( np.array(obb['normalizedAxes']).reshape(3,3).transpose().tolist() )
    # del data
    
    objs_obbinfo=dict()
    pth_obj_graph = os.path.join(pth_3RScan,scan_id,'graph.json')
    with open(pth_obj_graph) as f: 
        data = json.load(f)
    for nid, node in data[scan_id]['nodes'].items():
        
        obj_obbinfo = objs_obbinfo[int(nid)] = dict()
        obj_obbinfo['center'] = copy.deepcopy(node['center'])
        obj_obbinfo['dimension'] = copy.deepcopy(node['dimension'])
        obj_obbinfo['normAxes'] = copy.deepcopy( np.array(node['rotation']).reshape(3,3).transpose().tolist() )
    del data
        
    # count gt segment size
    size_segments_gt = dict()
    for segment_id in segments_gt:
        segment_indices = np.where(segments_gt == segment_id)[0]
        size_segments_gt[segment_id] = len(segment_indices)
    
    ''' Find and count all corresponding segments'''
    tree = o3d.geometry.KDTreeFlann(points_gt)
    count_seg_pd_2_corresponding_seg_gts = dict() # counts each segment_pd to its corresonding segment_gt
    
    size_segments_pd = dict()
    instance2labelName_filtered = dict()
    for segment_id in segment_ids:
        segment_indices = np.where(segments_pd == segment_id)[0]
        segment_points = points_pd[segment_indices]        

        size_segments_pd[segment_id] = len(segment_points)
        
        if filter_segment_size > 0:
            if size_segments_pd[segment_id] < filter_segment_size:
                # print('skip segment',segment_id,'with size',size_segments_pd[segment_id],'that smaller than',filter_segment_size)
                continue
            
        for i in range(len(segment_points)):
            point = segment_points[i]
            # [k, idx, distance] = tree.search_radius_vector_3d(point,0.001)
            k, idx, distance = tree.search_knn_vector_3d(point,1)
            if distance[0] > max_distance: continue
            # label_gt = labels_gt[idx][0]
            segment_gt = segments_gt[idx][0]
            
            if segment_gt not in instance2labelName: continue
            if instance2labelName[segment_gt] == 'none': continue
            instance2labelName_filtered[segment_gt] = instance2labelName[segment_gt]

            if segment_id not in count_seg_pd_2_corresponding_seg_gts: 
                count_seg_pd_2_corresponding_seg_gts[segment_id] = dict()            
            if segment_gt not in count_seg_pd_2_corresponding_seg_gts[segment_id]: 
                count_seg_pd_2_corresponding_seg_gts[segment_id][segment_gt] = 0
            count_seg_pd_2_corresponding_seg_gts[segment_id][segment_gt] += 1
    
    instance2labelName = instance2labelName_filtered
    
        # break
    if verbose or debug:
        print('There are {} segments have found their correponding GT segments.'.format(len(count_seg_pd_2_corresponding_seg_gts)))
        for k,i in count_seg_pd_2_corresponding_seg_gts.items():
            print('\t{}: {}'.format(k,len(i)))

    ''' Save as ply '''
    if debug:
        if args.label_type == 'NYU40':
            colors = util_label.get_NYU40_color_palette()
            cloud_gt.visual.vertex_colors = [0,0,0,255]
            for seg, label_name in instance2labelName.items():
                segment_indices = np.where(segments_gt == seg)[0]
                if label_name == 'none':continue
                label = util_label.NYU40_Label_Names.index(label_name)+1
                for index in segment_indices:
                    cloud_gt.visual.vertex_colors[index][:3] = colors[label]
            cloud_gt.export('tmp_gtcloud.ply')
        else:
            for seg, label_name in instance2labelName.items():
                segment_indices = np.where(segments_gt == seg)[0]
                if label_name != 'none':
                    continue
                for index in segment_indices:
                    cloud_gt.visual.vertex_colors[index][:3] = [0,0,0]
            cloud_gt.export('tmp_gtcloud.ply')

    ''' Find best corresponding segment '''
    map_segment_pd_2_gt = dict() # map segment_pd to segment_gt
    gt_segments_2_pd_segments = defaultdict(list) # how many segment_pd corresponding to this segment_gt
    for segment_id, cor_counter in count_seg_pd_2_corresponding_seg_gts.items():
        size_pd = size_segments_pd[segment_id]
        if verbose: print('segment_id', segment_id, size_pd)
        
        max_corr_ratio = -1
        max_corr_seg   = -1
        list_corr_ratio = list()
        for segment_gt, count in cor_counter.items():
            size_gt = size_segments_gt[segment_gt]
            corr_ratio = count/size_pd
            list_corr_ratio.append(corr_ratio)
            if corr_ratio > max_corr_ratio:
                max_corr_ratio = corr_ratio
                max_corr_seg   = segment_gt
            if verbose or debug: print('\t{0:s} {1:3d} {2:8d} {3:2.3f} {4:2.3f}'.\
                                       format(instance2labelName[segment_gt],segment_gt,count, count/size_gt, corr_ratio))
        if len(list_corr_ratio ) > 2:
            list_corr_ratio = sorted(list_corr_ratio,reverse=True)
            occ_ratio = list_corr_ratio[1]/list_corr_ratio[0]
        else:
            occ_ratio = 0

        if max_corr_ratio > filter_corr_thres and occ_ratio < filter_occ_ratio:
            '''
            This is to prevent a segment is almost equally occupied two or more gt segments. 
            '''
            if verbose or debug: print('add correspondence of segment {:s} {:4d} to label {:4d} with the ratio {:2.3f} {:1.3f}'.\
                  format(instance2labelName[segment_gt],segment_id,max_corr_seg,max_corr_ratio,occ_ratio))
            map_segment_pd_2_gt[segment_id] = max_corr_seg
            gt_segments_2_pd_segments[max_corr_seg].append(segment_id)
        else:
            if verbose or debug: print('filter correspondence segment {:s} {:4d} to label {:4d} with the ratio {:2.3f} {:1.3f}'.\
                  format(instance2labelName[segment_gt],segment_id,max_corr_seg,max_corr_ratio,occ_ratio))
                
    if verbose: 
        print('final correspondence:')
        print('  pd  gt')
        for segment, label in sorted(map_segment_pd_2_gt.items()):
            print("{:4d} {:4d}".format(segment,label))
        print('final pd segments within the same gt segment')
        for gt_segment, pd_segments in sorted(gt_segments_2_pd_segments.items()):
            print('{:4d}:'.format(gt_segment),end='')
            for pd_segment in pd_segments:
                print('{} '.format(pd_segment),end='')        
            print('')

    ''' Save as ply '''
    if debug:        
        if args.label_type == 'NYU40':
            colors = util_label.get_NYU40_color_palette()
            cloud_pd.visual.vertex_colors = [0,0,0,255]
            for segment_pd, segment_gt in map_segment_pd_2_gt.items():
                segment_indices = np.where(segments_pd == segment_pd)[0]
                label = util_label.NYU40_Label_Names.index(instance2labelName[segment_gt])+1
                color = colors[label]
                for index in segment_indices:
                    cloud_pd.visual.vertex_colors[index][:3] = color
            cloud_pd.export('tmp_corrcloud.ply')
        else:
            cloud_pd.visual.vertex_colors = [0,0,0,255]
            for segment_pd, segment_gt in map_segment_pd_2_gt.items():
                segment_indices = np.where(segments_pd == segment_pd)[0]
                for index in segment_indices:
                    cloud_pd.visual.vertex_colors[index] = [255,255,255,255]
            cloud_pd.export('tmp_corrcloud.ply')

    '''' Save as relationship_*.json '''
    list_relationships = list()
    if seg_groups is not None:
        for split_id in range(len(seg_groups)):
            seg_group = seg_groups[split_id]
            relationships = gen_relationship(scan_id,split_id,gt_relationships,map_segment_pd_2_gt, instance2labelName, 
                                             gt_segments_2_pd_segments,seg_group)
            if len(relationships["objects"]) == 0 or len(relationships['relationships']) == 0:
                continue
            list_relationships.append(relationships)
            
            ''' check '''
            for obj in relationships['objects']:
                assert(obj in seg_group)
            for rel in relationships['relationships']:
                assert(rel[0] in relationships['objects'])
                assert(rel[1] in relationships['objects'])
    else:
        relationships = gen_relationship(scan_id,0,gt_relationships, map_segment_pd_2_gt, instance2labelName, 
                                                   gt_segments_2_pd_segments)
        if len(relationships["objects"]) != 0 and len(relationships['relationships']) != 0:
                list_relationships.append(relationships)
                
    for relationships in list_relationships:
        for oid in relationships['objects'].keys():
            relationships['objects'][oid] = {**objs_obbinfo[oid], **relationships['objects'][oid]}
    
    return list_relationships, segs_neighbors


def gen_relationship(scan_id:str,split:int, gt_relationships:list,map_segment_pd_2_gt:dict,instance2labelName:dict,gt_segments_2_pd_segments:dict,
                     target_segments:list=None) -> dict:
    '''' Save as relationship_*.json '''
    relationships = dict() #relationships_new["scans"].append(s)
    relationships["scan"] = scan_id
    relationships["split"] = split
    
    objects = dict()
    for seg, segment_gt in map_segment_pd_2_gt.items():
        if target_segments is not None:
            if seg not in target_segments: continue
        name = instance2labelName[segment_gt]
        assert(name != '-' and name != 'none')
        objects[int(seg)] = dict()
        objects[int(seg)]['label'] = name
        objects[int(seg)]['instance_id'] = segment_gt
    relationships["objects"] = objects
    
    
    split_relationships = list()
    ''' Inherit relationships from ground truth segments '''
    if gt_relationships is not None:
        relationships_names = util.read_relationships(os.path.join(define.FILE_PATH, args.relation + ".txt"))

        for rel in gt_relationships:
            id_src = rel[0]
            id_tar = rel[1]
            num = rel[2]
            name = rel[3]
            idx_in_txt = relationships_names.index(name)
            assert(num==idx_in_txt)
            if name not in target_relationships: 
                continue
            idx_in_txt_new = target_relationships.index(name)
            
            if id_src == id_tar:
                continue # an edge canno self connect
                print('halloe',print(rel))
            
            if id_src in gt_segments_2_pd_segments and id_tar in gt_segments_2_pd_segments:
                segments_src = gt_segments_2_pd_segments[id_src]
                segments_tar = gt_segments_2_pd_segments[id_tar]                
                for segment_src in segments_src:
                    if segment_src not in objects:
                        if debug:print('filter',name,'segment_src', instance2labelName[id_src],' is not in objects')
                        continue
                    for segment_tar in segments_tar:        
                        if segment_tar not in objects:
                            if debug:print('filter',name,'segment_tar', instance2labelName[id_tar], ' is not in objects')
                            continue
                        if target_segments is not None:
                            ''' skip if they not in the target_segments'''
                            if segment_src not in target_segments: continue
                            if segment_tar not in target_segments: continue
                        
                        ''' check if they are neighbors '''
                        split_relationships.append([ int(segment_src), int(segment_tar), idx_in_txt_new, name ])
                        if debug:print('segment',segment_src, '(',id_src,')',segment_tar,'(',id_tar,')',
                                       'inherit',instance2labelName[id_src],name, instance2labelName[id_tar])
            # else:
            #     if debug:
            #         if id_src in gt_segments_2_pd_segments:
            #             print('filter', instance2labelName[id_src],name, instance2labelName[id_tar],'id_src', id_src, 'is not in the gt_segments_2_pd_segments')
            #         if id_tar in gt_segments_2_pd_segments:
            #             print('filter', instance2labelName[id_src],name, instance2labelName[id_tar],'id_tar', id_tar, 'is not in the gt_segments_2_pd_segments')
    
    ''' Build "same part" relationship '''
    idx_in_txt_new = target_relationships.index(define.NAME_SAME_PART)
    for _, groups in gt_segments_2_pd_segments.items():
        if target_segments is not None:
            filtered_groups = list()
            for g in groups:
                if g in target_segments:
                    filtered_groups.append(g)
            groups = filtered_groups
        if len(groups) <= 1: continue
                    
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                split_relationships.append([int(groups[i]),int(groups[j]), idx_in_txt_new, define.NAME_SAME_PART])
                split_relationships.append([int(groups[j]),int(groups[i]), idx_in_txt_new, define.NAME_SAME_PART])
                
    '''check if a pair has multiple relationsihps'''
    relatinoships_gt_dict= defaultdict(list)
    for r in split_relationships:
        r_src = int(r[0])
        r_tgt = int(r[1])
        r_lid = int(r[2])
        r_cls = r[3]
        relatinoships_gt_dict[(r_src,r_tgt)].append(r_cls)
    invalid_keys=list()
    for key,value in relatinoships_gt_dict.items():
        if len(value) != 1:
            invalid_keys.append(key)
    for key in invalid_keys:
        print('key:',key, 'has more than one predicates:',relatinoships_gt_dict[key])
        print(objects[key[0]]['label'],objects[key[1]]['label'])
    assert len(invalid_keys)==0
    
    relationships["relationships"] = split_relationships
    return relationships
    
if __name__ == '__main__':
    args = Parser().parse_args()
    Path(args.pth_out).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.pth_out,'gen_data_'+args.type+'.log'), level=logging.DEBUG)
    logger_py = logging.getLogger(__name__)
    
    debug |= args.debug>0
    args.verbose |= debug
    if args.search_method == 'BBOX':
        search_method = SAMPLE_METHODS.BBOX
    elif args.search_method == 'KNN':
        search_method = SAMPLE_METHODS.RADIUS
    
    util.set_random_seed(2020)
    label_names, _, _ = util_label.getLabelMapping(args.label_type)
    classes_json = list()
    for key,value in label_names.items():
        if value == '-':continue
        classes_json.append(value)
        
    ''' Read Scan and their type=['train', 'test', 'validation'] '''
    scan2type = {}
    with open(define.Scan3RJson_PATH, "r") as read_file:
        data = json.load(read_file)
        for scene in data:
            scan2type[scene["reference"]] = scene["type"]
            for scan in scene["scans"]:
                scan2type[scan["reference"]] = scene["type"]
    
    '''read relationships'''
    target_relationships=list()
    if args.inherit:
        # target_relationships += ['supported by', 'attached to','standing on', 'lying on','hanging on','connected to',
                                # 'leaning against','part of','build in','standing in','lying in','hanging in']
        target_relationships += ['supported by', 'attached to','standing on','hanging on','connected to','part of','build in']
    target_relationships.append(define.NAME_SAME_PART)
    
    target_scan=[]
    if args.target_scan != '':
        target_scan = util.read_txt_to_list(args.target_scan)
        
    valid_scans=list()
    relationships_new = dict()
    relationships_new["scans"] = list()
    relationships_new['neighbors'] = dict()
    counter= 0
    with open(os.path.join(define.FILE_PATH + args.relation + ".json"), "r") as read_file:
        data = json.load(read_file)
        filtered_data = list()
        
        for s in data["scans"]:
            scan_id = s["scan"]
            if len(target_scan) ==0:
                if scan2type[scan_id] != args.type: 
                    if args.verbose:
                        print('skip',scan_id,'not validation type')
                    continue
            else:
                if scan_id not in target_scan: continue
            
            filtered_data.append(s)
        
        for s in tqdm(filtered_data):
            scan_id = s["scan"]
            gt_relationships = s["relationships"]
            logger_py.info('processing scene {}'.format(scan_id))
            valid_scans.append(scan_id)
            relationships, segs_neighbors = process(define.DATA_PATH, scan_id, args.label_type, target_relationships,
                                    gt_relationships = gt_relationships,
                                    split_scene = args.split,
                                    verbose = args.verbose)
            if len(relationships) == 0:
                logger_py.info('skip {} due to not enough objs and relationships'.format(scan_id))
                continue
            else:
                if debug:  print('no skip', scan_id)
            
            relationships_new["scans"] += relationships
            relationships_new['neighbors'][scan_id] = segs_neighbors
            
            # if debug: break
            
    '''Save'''
    pth_args = os.path.join(args.pth_out,'args.json')
    with open(pth_args, 'w') as f:
            tmp = vars(args)
            json.dump(tmp, f, indent=2)
            
    pth_classes = os.path.join(args.pth_out, 'classes.txt')
    with open(pth_classes,'w') as f:
        for name in classes_json:
            if name == '-': continue
            f.write('{}\n'.format(name))
    pth_relation = os.path.join(args.pth_out, 'relationships.txt')
    with open(pth_relation,'w') as f:
        for name in target_relationships:
            f.write('{}\n'.format(name))
    pth_split = os.path.join(args.pth_out, args.type+'_scans.txt')
    with open(pth_split,'w') as f:
        for name in valid_scans:
            f.write('{}\n'.format(name))
    # '''Save to json'''
    # pth_relationships_json = os.path.join(args.pth_out, "relationships_" + args.type + ".json")
    # with open(pth_relationships_json, 'w') as f:
    #     json.dump(relationships_new, f)
        
    '''Save to h5'''
    pth_relationships_json = os.path.join(args.pth_out, "relationships_" + args.type + ".h5")
    h5f = h5py.File(pth_relationships_json, 'w')
    # reorganize scans from list to dict
    scans = dict()
    for s in relationships_new['scans']:
        scans[s['scan']] = s
    all_neighbors = relationships_new['neighbors']
    for scan_id in scans.keys():
        scan_data = scans[scan_id]
        neighbors = all_neighbors[scan_id]
        objects = scan_data['objects']
        
        d_scan = dict()
        d_nodes = d_scan['nodes'] = dict()
        
        ## Nodes
        for idx, data in enumerate(objects.items()):
            oid, obj_info = data
            ascii_nn = [str(n).encode("ascii", "ignore") for n in neighbors[oid]]
            d_nodes[oid] = dict()
            d_nodes[oid] = obj_info
            d_nodes[oid]['neighbors'] = ascii_nn
        
        ## Relationships
        str_relationships = list() 
        for rel in scan_data['relationships']:
            str_relationships.append([str(s) for s in rel])
        d_scan['relationships']= str_relationships
        
        s_scan = str(d_scan)
        h5_scan = h5f.create_dataset(scan_id,data=np.array([s_scan],dtype='S'),compression='gzip')
        # test decode 
        tmp = h5_scan[0].decode()
        assert isinstance(ast.literal_eval(tmp),dict)
        
        # ast.literal_eval(h5_scan)
    h5f.close()
    
