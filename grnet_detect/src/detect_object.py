#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import open3d as o3d

from GRNetDetector.GRNet_Detector import GRNet_Detector

red_white_color_map = np.array([
    [228, 177, 171],
    [227, 150, 149],
    [223, 115, 115],
    [218, 85, 82],
    [204, 68, 75],
    [204, 68, 75],
    [204, 68, 75],
    [204, 68, 75],
    [204, 68, 75],
    [204, 68, 75],
    [204, 68, 75],
    [204, 68, 75],
    [204, 68, 75],
], dtype=np.float)

red_blue_color_map = np.array([
    [179, 222, 226],
    [234, 242, 215],
    [239, 207, 227],
    [234, 154, 178],
    [226, 115, 150],
    [226, 115, 150],
    [226, 115, 150],
    [226, 115, 150],
    [226, 115, 150],
    [226, 115, 150],
    [226, 115, 150],
    [226, 115, 150],
    [226, 115, 150],
    [226, 115, 150],
    [226, 115, 150],
    [226, 115, 150],
    [226, 115, 150],
    [226, 115, 150],
    [226, 115, 150],
    [226, 115, 150],
    [226, 115, 150],
    [226, 115, 150],
    [226, 115, 150],
], dtype=np.float)

COLOR_MAP = red_blue_color_map

def demo():
    model_path = os.environ['HOME'] + "/.ros/GRNet-ShapeNet.pth"
    pcd_file_path = "/home/chli/chLi/2022_5_9_18-35-42/scan7.ply"
    o3d_pcd_file_path = "/home/chli/chLi/2022_5_9_18-35-42/scan9.ply"
    gt_pcd_file_path = "/home/chli/chLi/2022_5_9_18-35-42/target_gt_mesh_with_color.ply"
    output_file_path = "/home/chli/chLi/2022_5_9_18-35-42/scan7_grnet.ply"

    grnet_detector = GRNet_Detector()
    grnet_detector.load_model(model_path)
    pointcloud_result = grnet_detector.detect_pcd_file(pcd_file_path)

    #  complete_pointcloud = o3d.geometry.PointCloud()
    #  complete_pointcloud.points = o3d.utility.Vector3dVector(pointcloud_result)
    complete_mesh = o3d.io.read_triangle_mesh(gt_pcd_file_path)
    complete_pointcloud = o3d.geometry.PointCloud()
    complete_pointcloud.points = \
        o3d.utility.Vector3dVector(np.asarray(complete_mesh.vertices))

    partial_pointcloud = o3d.io.read_point_cloud(o3d_pcd_file_path)
    sigma = 0
    if sigma > 0:
        partial_points = np.array(partial_pointcloud.points)
        noise_x = np.random.normal(0, sigma, partial_points.shape[0])
        noise_y = np.random.normal(0, sigma, partial_points.shape[0])
        noise_z = np.random.normal(0, sigma, partial_points.shape[0])
        noise = []
        for i in range(partial_points.shape[0]):
            noise.append([noise_x[i], noise_y[i], noise_z[i]])
        noise = np.array(noise)
        partial_points += noise
        partial_pointcloud.points = o3d.utility.Vector3dVector(partial_points)
    partial_colors = np.array(
        [np.array([74, 24, 220])/255.0 for _ in
         range(np.array(partial_pointcloud.points).shape[0])])
    partial_pointcloud.colors = o3d.utility.Vector3dVector(partial_colors)
    partial_pointcloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))

    dist_to_partial = complete_pointcloud.compute_point_cloud_distance(
        partial_pointcloud)

    colors = []
    color_num = len(COLOR_MAP)
    min_dist = 0
    max_dist = np.max(dist_to_partial)
    dist_step = (max_dist - min_dist) / color_num

    for dist in dist_to_partial:
        dist_divide = dist / dist_step
        color_idx = int(dist_divide)
        if color_idx >= color_num:
            color_idx -= 1
        next_color_weight = dist_divide - color_idx

        color = (1.0 - next_color_weight) * COLOR_MAP[color_idx]
        if color_idx < color_num - 1:
            color += next_color_weight * COLOR_MAP[color_idx + 1]
        colors.append(color)

    colors = np.array(colors, dtype=np.float) / 255.0
    complete_pointcloud.colors = o3d.utility.Vector3dVector(colors)
    complete_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    #  complete_mesh.compute_vertex_normals()

    #  complete_pointcloud.estimate_normals(
        #  search_param=o3d.geometry.KDTreeSearchParamHybrid(
            #  radius=0.1, max_nn=30))

    o3d.visualization.draw_geometries([
        #  partial_pointcloud,
        complete_mesh
    ])
    #  o3d.io.write_point_cloud(output_file_path, complete_pointcloud)
    return True

if __name__ == "__main__":
    demo()

