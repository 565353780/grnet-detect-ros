#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import open3d as o3d

from GRNetDetector.GRNet_Detector import GRNet_Detector

COLOR_MAP = [
    (228, 177, 171),
    (227, 150, 149),
    (223, 115, 115),
    (218, 85, 82),
    (204, 68, 75)
]

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
    dist_to_partial = complete_pointcloud.compute_point_cloud_distance(
        partial_pointcloud)

    colors = []
    color_num = len(COLOR_MAP)
    min_dist = np.min(dist_to_partial)
    max_dist = np.max(dist_to_partial)
    dist_step = (max_dist - min_dist) / color_num
    color_dist_range_list = \
        [min_dist + i * dist_step for i in range(1, color_num)]

    point_used_list = np.array([0 for _ in range(pointcloud_result.shape[0])])

    max_dist_point_list = []
    max_dist_point_num = 3
    max_dist_point_circle = max_dist

    for _ in range(max_dist_point_num):
        unused_point_list = \
            pointcloud_result[np.where(point_used_list == 0)]
        unused_dist_list = \
            dist_to_partial[np.where(point_used_list == 0)]
        max_dist_point_idx = np.argmax(unused_dist_list)
        max_dist_point = unused_point_list[max_dist_point_idx]
        max_dist_point_list.append(max_dist_point)
        for i in range(pointcloud_result.shape[0]):
            if point_used_list[i] == 1:
                continue
            current_point = pointcloud_result[i]
            xdiff = current_point[0] - max_dist_point[0]
            ydiff = current_point[1] - max_dist_point[1]
            zdiff = current_point[2] - max_dist_point[2]
            current_dist = xdiff * xdiff + ydiff * ydiff + zdiff * zdiff
            if current_dist < max_dist_point_circle:
                point_used_list[i] = 1

    for i in range(pointcloud_result.shape[0]):
        if not point_used_list[i]:
            colors.append(COLOR_MAP[0])
        colors.append(COLOR_MAP[4])

    colors = np.array(colors, dtype=np.float) / 255.0
    complete_pointcloud.colors = o3d.utility.Vector3dVector(colors)

    complete_pointcloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))

    o3d.visualization.draw_geometries([
        partial_pointcloud,
        complete_pointcloud])
    #  o3d.io.write_point_cloud(output_file_path, complete_pointcloud)
    return True

if __name__ == "__main__":
    demo()

