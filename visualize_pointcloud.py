import open3d as o3d
import os
import numpy as np
import json

path_3rscan = "./data/3RScan/data/3RScan"
#scan_id = "0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca"
#scan_id = "baf673a1-8e94-22f8-824b-1139a8fc8bda"
scan_id = "754e884c-ea24-2175-8b34-cead19d4198d"
scan_id = "c7895f2b-339c-2d13-8248-b0507e050314"
scan_id = "f2c76fe5-2239-29d0-8593-1a2555125595"
ply_file = "labels.instances.align.annotated.v2.ply"
#ply_file = "labels.instances.annotated.v2.align.ply"
mesh_file = "mesh.refined.v2.obj"
ply_file_no_align = "labels.instances.annotated.v2.ply"
semseg_v2_file = "semseg.v2.json" # obb 
#inseg_ply_file = "inseg.ply"
#inseg_ply_file = "2dssg_orbslam3.ply"
#inseg_ply_file = "color.align.ply"


scan_id_path = os.path.join(path_3rscan, scan_id)
mesh_path = os.path.join(scan_id_path, ply_file)
ply_file_path = os.path.join(scan_id_path, ply_file)
#ply_file_path = os.path.join(scan_id_path, ply_file_no_align)
semseg_v2_file_path = os.path.join(scan_id_path, semseg_v2_file)

# read point cloud
pcd = o3d.io.read_point_cloud(ply_file_path)
mesh = o3d.io.read_triangle_mesh(mesh_path)
#print(pcd)
#print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.5,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

# read semseg json 
with open(semseg_v2_file_path, 'r') as json_file:
    semseg = json.load(json_file)

# segGroups -> dann bekomme ich eine Liste (also Zugriff über Indizes)
# dann habe ich in der Liste wieder ein dict Objekt mit einem key namens "obb
# obb enthält als value widerum ein dict mit values (Arrays)

# keys
keys = list(semseg.keys())
objects = semseg[keys[0]] # keys[0] = segGroups; objects = List type
numb_obj = len(objects) # 30 Objects

render_array = []

for obj1 in objects:
    # obj1 = objects[i]
    label = obj1["label"]
    print(f"label {label}")
    obb_obj1 = obj1["obb"] 
    obb_obj1_keys = list(obb_obj1.keys())
    print(obb_obj1_keys)
    axesLengths = np.array(obb_obj1["axesLengths"]).reshape((3,1)) 
    centroid = np.array(obb_obj1["centroid"]).reshape((3,1)) 
    normalizedAxes = np.array(obb_obj1["normalizedAxes"]).reshape((3,3)) 
    normalizedAxes = np.rot90(np.fliplr(normalizedAxes))
#normalizedAxes = np.fliplr(normalizedAxes)

#normalizedAxes = np.array([normalizedAxes[0], normalizedAxes[2], normalizedAxes[1]])
# print(type(axesLengths))
# print(f"axesLength {axesLengths.shape}")
# print(f"centroid {centroid.shape}")
# print(f"normalizedAxes {normalizedAxes.shape}")
# print(obb_obj1)
# print(f"axesLength {axesLengths}")
# print(f"centroid {centroid}")
# print(f"normalizedAxes {normalizedAxes}")

    obb_obj1 = o3d.geometry.OrientedBoundingBox(center=centroid, R=normalizedAxes, extent=axesLengths)
    render_array.append(obb_obj1)

# o3d.visualization.draw_geometries([pcd, obb_obj1],
#                                   zoom=0.5,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])


render_array.append(mesh)
o3d.visualization.draw_geometries(render_array,
                                  zoom=0.5,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[0.0, 0.0, 0.0],
                                  up=[0.0, 1.0, 0.0])






# "segGroups": [
#         {
#             "objectId": 1,
#             "id": 1,
#             "partId": 1,
#             "index": 0,
#             "dominantNormal": [
#                 0,
#                 -2.220446049250313e-16,
#                 1
#             ],
#             "obb": {
#                 "axesLengths": [
#                     6.178954373532301,
#                     0.5182499173674748,
#                     3.3831135437636566
#                 ],
#                 "centroid": [
#                     0.358748431470707,
#                     0.3052196121981847,
#                     -1.1691250260369859
#                 ],
#                 "normalizedAxes": [
#                     0.7100464701652527,
#                     -0.7041548490524292,
#                     -1.563537852638917e-16,
#                     0,
#                     -2.220446049250313e-16,
#                     1,
#                     -0.7041548490524292,
#                     -0.7100464701652527,
#                     -1.5766198794625656e-16
#                 ]
#             },


# inseg_ply_file_path = os.path.join(path_3rscan, scan_id, inseg_ply_file)
# inseg_pcd = o3d.io.read_point_cloud(inseg_ply_file_path)
# print(inseg_pcd)
# print(np.asarray(inseg_pcd.points))
# o3d.visualization.draw_geometries([inseg_pcd],
#                                   zoom=0.5,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024])

