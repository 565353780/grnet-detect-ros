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
])

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
])

COLOR_MAP = red_white_color_map

def demo():
    model_path = os.environ['HOME'] + "/.ros/GRNet-ShapeNet.pth"
    pcd_file_path = "/home/chli/chLi/incomplete_chair.ply"
    o3d_pcd_file_path = "/home/chli/chLi/incomplete_chair_1000000.ply"
    output_file_path = "/home/chli/chLi/complete_chair.ply"

    grnet_detector = GRNet_Detector()
    grnet_detector.load_model(model_path)
    pointcloud_result = grnet_detector.detect_pcd_file(pcd_file_path)
    complete_pointcloud = o3d.geometry.PointCloud()
    complete_pointcloud.points = o3d.utility.Vector3dVector(pointcloud_result)

    partial_pointcloud = o3d.io.read_point_cloud(o3d_pcd_file_path)
    partial_colors = np.array(
        [np.array([197, 165, 59])/255.0 for _ in
         range(np.array(partial_pointcloud.points).shape[0])])
    partial_pointcloud.colors = o3d.utility.Vector3dVector(partial_colors)
    partial_pointcloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))

    dist_to_partial = complete_pointcloud.compute_point_cloud_distance(
        partial_pointcloud)

    colors = []
    color_num = len(COLOR_MAP)
    min_dist = np.min(dist_to_partial)
    max_dist = np.max(dist_to_partial)
    dist_step = (max_dist - min_dist) / color_num
    color_dist_range_list = \
        [min_dist + i * dist_step for i in range(color_num)]

    for dist in dist_to_partial:
        color_idx = 0
        for i in range(1, len(color_dist_range_list)):
            color_dist = color_dist_range_list[i]
            if dist <= color_dist:
                break
            color_idx += 1
        last_color_weight = \
            (color_dist_range_list[color_idx] - \
             color_dist_range_list[color_idx - 1]) / \
            dist_step
        current_color = COLOR_MAP[color_idx]
        last_color = COLOR_MAP[color_idx - 1]
        mix_color = \
            current_color + last_color_weight * (last_color - current_color)
        colors.append(mix_color)

    colors = np.array(colors, dtype=np.float) / 255.0
    complete_pointcloud.colors = o3d.utility.Vector3dVector(colors)

    #  complete_pointcloud.estimate_normals(
        #  search_param=o3d.geometry.KDTreeSearchParamHybrid(
            #  radius=0.1, max_nn=30))

    o3d.visualization.draw_geometries([
        partial_pointcloud,
        complete_pointcloud])
    #  o3d.io.write_point_cloud(output_file_path, complete_pointcloud)
    return True

if __name__ == "__main__":
    demo()

