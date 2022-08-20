# ---------------------------------------------------------------------------
# Created by: Chuanbo Wang
# Created on: 8/13/2022
# ---------------------------------------------------------------------------
# This script post process the predicted point cloud from seedformer. 
# ---------------------------------------------------------------------------
import os, open3d
import numpy as np


def remove_partial_from_pred(pred, partial):
    """
    A seedformer's prediction contains points from both its partial and gt point cloud.
    This function removes the partial point cloud form a prediction point clouds.
    So that the prediction point cloud after removal is just like the gt point cloud.

    Args:
        pred (open3d.geometry.PointCloud): prediciton point cloud
        partial (open3d.geometry.PointCloud): partial point cloud
    
    Return:
        processed (open3d.geometry.PointCloud): post processed prediciton point cloud
    """
    dist_vector = pred.compute_point_cloud_distance(partial)
    dist_vector = np.array(dist_vector)
    pred_after_removal_points = []

    pred_nearest_neighbor_distance = pred.compute_nearest_neighbor_distance()
    pred_nearest_neighbor_distance = np.array(pred_nearest_neighbor_distance)
    threshold = np.amax(pred_nearest_neighbor_distance) / 5
    for j, dist in enumerate(dist_vector):
        if dist > threshold: # points within 0.1 distance to the bracket point cloud are saved as bracket_points
            pred_after_removal_points.append(pred.points[j])
    pred_after_removal_points = np.array(pred_after_removal_points)
    pred_after_removal_pc = open3d.geometry.PointCloud()
    pred_after_removal_pc.points = open3d.utility.Vector3dVector(pred_after_removal_points)
    return pred_after_removal_pc


def main():    
    log_dir = "train_pcn_Log_2022_08_12_19_58_18"
    pred_path = "./test/" + log_dir + "/outputs/00000000/"
    pred_filelist = os.listdir(pred_path)
    pred_filelist = [f for f in pred_filelist if '_pred' in f]
    
    for pred_filename in pred_filelist:
        pred_filepath = pred_path + pred_filename
        partial_filepath = pred_path + pred_filename.replace('_pred','_partial')
        gt_filepath = pred_path + pred_filename.replace('_pred','_gt')
        pred = open3d.io.read_point_cloud(pred_filepath)
        partial = open3d.io.read_point_cloud(partial_filepath)
        gt = open3d.io.read_point_cloud(gt_filepath)
        pred_after_removal = remove_partial_from_pred(pred, partial)
        # paint  colors 
        pred_after_removal.paint_uniform_color([1,0,0])
        gt.paint_uniform_color([0,0,1])     
        partial.paint_uniform_color([0,1,0])
        
        # translate
        max_bound = pred_after_removal.get_axis_aligned_bounding_box().get_max_bound()
        x = 0
        y = 0
        z = max_bound[2] * 2
        translation = np.array([x,y,z]).astype('float64')
        partial.translate(translation)
        # open3d.visualization.draw_geometries([pred_after_removal, partial, gt])
        partial.translate(translation)
        gt.translate(translation)
        pred_after_removal.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        partial.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        gt.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        open3d.visualization.draw_geometries([pred_after_removal, partial, gt])


if __name__ == "__main__":
    main()