#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import open3d as o3d

from GRNetDetector.GRNet_Detector import GRNet_Detector

red_white_color_map = np.array([
    [180, 180, 180],
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
], dtype=np.float)

red_blue_color_map = np.array([
    [179, 222, 226],
    [234, 242, 215],
    [239, 207, 227],
    [234, 154, 178],
    [226, 115, 150], [226, 115, 150],
    [226, 115, 150], [226, 115, 150],
    [226, 115, 150], [226, 115, 150],
    [226, 115, 150], [226, 115, 150],
    [226, 115, 150], [226, 115, 150],
], dtype=np.float)

COLOR_MAP = red_white_color_map

def demo():
    model_path = os.environ['HOME'] + "/.ros/GRNet-ShapeNet.pth"
    pcd_file_path = "/home/chli/chLi/2022_5_9_18-35-42/scan7.ply"
    o3d_pcd_file_path = "/home/chli/chLi/coscan_data/sofa-incomp.ply"
    gt_pcd_file_path = "/home/chli/chLi/coscan_data/sofa-complete.ply"
    output_file_path = "/home/chli/chLi/2022_5_9_18-35-42/scan7_grnet.ply"

    grnet_detector = GRNet_Detector()
    grnet_detector.load_model(model_path)
    pointcloud_result = grnet_detector.detect_pcd_file(pcd_file_path)

    #  complete_pointcloud = o3d.geometry.PointCloud()
    #  complete_pointcloud.points = o3d.utility.Vector3dVector(pointcloud_result)
    complete_mesh = o3d.io.read_triangle_mesh(gt_pcd_file_path)
    complete_pointcloud = o3d.io.read_point_cloud(gt_pcd_file_path)

    partial_mesh = o3d.io.read_triangle_mesh(o3d_pcd_file_path)
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
        partial_mesh.vertices = o3d.utility.Vector3dVector(partial_points)

    partial_colors = np.array([[101, 91, 82] for _ in np.array(partial_mesh.vertices)],
                              dtype=np.float)/255.0
    partial_pointcloud.colors = o3d.utility.Vector3dVector(partial_colors)
    partial_mesh.vertex_colors = o3d.utility.Vector3dVector(partial_colors)

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

    complete_pointcloud.normals = o3d.utility.Vector3dVector()

    partial_mesh.compute_vertex_normals()
    #  complete_mesh.compute_vertex_normals()

    sphere_complete_pointcloud = o3d.geometry.PointCloud()
    sphere_complete_points_list = []
    sphere_complete_colors_list = []
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=0.001,
        resolution=10)
    sphere_points = np.array(mesh_sphere.vertices)

    complete_points = np.array(complete_pointcloud.points)
    complete_colors = np.array(complete_pointcloud.colors)
    for i in range(len(complete_points)):
        new_points = sphere_points + complete_points[i]
        sphere_complete_points_list.append(new_points)
        for _ in sphere_points:
            sphere_complete_colors_list.append(complete_colors[i])
    points = np.concatenate(sphere_complete_points_list, axis=0)
    colors = np.array(sphere_complete_colors_list)
    sphere_complete_pointcloud.points = \
        o3d.utility.Vector3dVector(points)
    sphere_complete_pointcloud.colors = \
        o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([
        partial_mesh,
        sphere_complete_pointcloud
    ])

    #  partial_ply_path = "/home/chli/chLi/coscan_data/sofa_partial.ply"
    #  complete_pcd_path = "/home/chli/chLi/coscan_data/sofa_complete.ply"
    #  o3d.io.write_triangle_mesh(partial_ply_path, partial_mesh)
    #  o3d.io.write_point_cloud(complete_pcd_path, complete_pointcloud)

    return True

if __name__ == "__main__":
    demo()

