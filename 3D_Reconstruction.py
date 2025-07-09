import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2

import open3d as o3d

import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# load the image to convert
img = cv2.imread('images/img_8.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# load the DepthAnything model for monocular depth estimation
processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to("cuda")


# infer the model to get the depth map
depth_input = processor(images=img, return_tensors="pt").to("cuda")

with torch.no_grad():
    inference_outputs = model(**depth_input)
    output_depth = inference_outputs.predicted_depth

output_depth = output_depth.squeeze().cpu().numpy()


# visualize depth map
# plt.rcParams['figure.dpi'] = 100

# fig, axs = plt.subplots(2, 1)

# axs[0].imshow(img)
# axs[0].set_title('Depth Estimation')
# axs[1].imshow(output_depth)

# plt.show()

def depth_to_pointcloud_orthographic(depth_map, image, scale_factor=255):

    height, width = depth_map.shape

    # Create a grid of pixel coordinates
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Scale the depth values
    z = (depth_map / scale_factor) * height/2

    # Create 3D points (x and y are pixel coordinates, z is from the depth map)
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Filter out points with zero depth
    mask = points[:, 2] != 0
    points = points[mask]

    # Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd])
    
    # Add colors to the point cloud
    colors = image.reshape(-1, 3)[mask] / 255.0  # Normalize color values to [0, 1]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=1)
    inlier_cloud = pcd.select_by_index(ind)

    return inlier_cloud, z, height, width 

# convert dimensions of original image to match depth map
img = cv2.resize(img, (output_depth.shape[1], output_depth.shape[0]))


# Convert depth map and image to point cloud
point_cloud, z, height, width  = depth_to_pointcloud_orthographic(output_depth, img)

# draw the point cloud
# o3d.visualization.draw_geometries([point_cloud])

# calculate normals for the mesh
point_cloud.estimate_normals()
point_cloud.orient_normals_to_align_with_direction()

# create and draw a triangle mesh from the point cloud
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
o3d.visualization.draw_geometries([mesh])

o3d.io.write_triangle_mesh('./results/mesh_8.obj', mesh, write_triangle_uvs=True)