#ifndef PC_CLEAN
#define PC_CLEAN

#include <string>
#include <math.h>

#include <pcl/io/ply_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/registration/icp.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/normal_3d.h>

#include <Eigen/Dense>
#include <Eigen/Core>

using namespace std;

typedef pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal> PointToPlane;

// Remove the horizontal table plane captured by the moving camera
void horizontalPlaneRemoval(const pcl::PointCloud<pcl::PointXYZ>::Ptr &PC,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr &PC_filterred,
                            float *voxel_grid_TSDF,
                            const float &voxel_size,
                            const float &voxel_grid_origin_x,
                            const float &voxel_grid_origin_y,
                            const float &voxel_grid_origin_z,
                            const int &voxel_grid_dim_x,
                            const int &voxel_grid_dim_y,
                            const int &voxel_grid_dim_z)
{
    std::cout << "Initialize..." << std::endl;
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    std::cout << "Initialize..." << std::endl;

    Eigen::Vector3f normal_axis(0.0, 0.0, 1.0);
    pcl::SACSegmentation<pcl::PointNormal> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMaxIterations(500);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setAxis(normal_axis);
    seg.setEpsAngle(20.0f * (M_PI / 180.0f));
    seg.setDistanceThreshold(0.012);

    std::cout << "Finish setting up SAC" << std::endl;

    // value to describe how horizontal the plane is
    double horizontality_threshold = 0.9;
    // value to describe how close the plane is to the origin
    double dist_to_origin_threshold = 0.1;

    // Compute normals for the PC
    pcl::PointCloud<pcl::PointNormal>::Ptr PC_normals (new pcl::PointCloud<pcl::PointNormal>); 
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(PC);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ne.setSearchMethod (tree);
    ne.setRadiusSearch (0.03);
    ne.compute(*normals);
    pcl::concatenateFields(*PC, *normals, *PC_normals);

    std::cout << "Finish computing normal" << std::endl;

    seg.setInputCloud(PC_normals);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() == 0)
        PCL_ERROR("Could not estimate a planar model for the given dataset.");

    std::cout << "Plane Model Coefficients: " << coefficients->values[0] << ", "
              << coefficients->values[1] << ", "
              << coefficients->values[2] << ", "
              << coefficients->values[3] << std::endl;

    if (std::abs(coefficients->values[2]) < horizontality_threshold &&
        std::abs(coefficients->values[3]) > dist_to_origin_threshold)
        PCL_ERROR("The coefficients of the planar model estimated exceed the threshold!");

    for (const auto &idx : inliers->indices)
    {
        int pt_grid_x = roundf((PC->points[idx].x - voxel_grid_origin_x) / voxel_size);
        int pt_grid_y = roundf((PC->points[idx].y - voxel_grid_origin_y) / voxel_size);
        int pt_grid_z = roundf((PC->points[idx].z - voxel_grid_origin_z) / voxel_size);

        int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;

        voxel_grid_TSDF[volume_idx] = 1.0f;
    }

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(PC);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*PC_filterred);
}

// Euclidean clustering to cluster the point clouds captured by both camera
// Use Euclidean cluster at the end of all the processing step
void euclideanClusterExtraction(const pcl::PointCloud<pcl::PointXYZ>::Ptr &PC,
                                pcl::PointCloud<pcl::PointXYZ>::Ptr &PC_filterred,
                                const float *voxel_grid_TSDF,
                                float *filterred_voxel_grid_TSDF,
                                const float &voxel_size,
                                const float &voxel_grid_origin_x,
                                const float &voxel_grid_origin_y,
                                const float &voxel_grid_origin_z,
                                const int &voxel_grid_dim_x,
                                const int &voxel_grid_dim_y,
                                const int &voxel_grid_dim_z)
{
    // Set up K-d tree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(PC);

    // Euclidean Cluster
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.015);
    ec.setMinClusterSize(2000);
    ec.setMaxClusterSize(100000);

    ec.setSearchMethod(tree);
    ec.setInputCloud(PC);
    ec.extract(cluster_indices);

    // Find the cluster with the largest number of points
    int largest_cluster_idx = 0;
    int largest_cluster_point_num = 0;
    for (int i = 0; i < cluster_indices.size(); i++)
    {
        std::cout << "Cluster " << i << " size: " << cluster_indices[i].indices.size() << std::endl;
        if (cluster_indices[i].indices.size() > largest_cluster_point_num)
        {
            largest_cluster_idx = i;
            largest_cluster_point_num = cluster_indices[i].indices.size();
        }
    }

    std::cout << "Cluster size: " << cluster_indices.size() << std::endl;

    // Get the filterred PC
    pcl::PointXYZ p;
    for (const auto& it: cluster_indices[largest_cluster_idx].indices)
    {
        p = (*PC)[it];
        PC_filterred->push_back(p);
    }

    PC_filterred->width = PC_filterred->size();
    PC_filterred->height = 1;
    PC_filterred->is_dense = true;

    // Compute the AABB of the filtered pc
    pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
    feature_extractor.setInputCloud(PC_filterred);
    feature_extractor.compute();
    
    pcl::PointXYZ min_point_AABB;
    pcl::PointXYZ max_point_AABB;

    feature_extractor.getAABB(min_point_AABB, max_point_AABB);

    // Get the filterred voxel TSDF
    int lookup_range = 6;
    
    int min_x_idx = roundf((min_point_AABB.x - voxel_grid_origin_x) / voxel_size);
    int min_y_idx = roundf((min_point_AABB.y - voxel_grid_origin_y) / voxel_size);
    int min_z_idx = roundf((min_point_AABB.z - voxel_grid_origin_z) / voxel_size);
    // min_z_idx = min(min_z_idx - 1, 0);

    for (const auto& it: cluster_indices[largest_cluster_idx].indices)
    {
        p = (*PC)[it];
        int pt_grid_x = roundf((p.x - voxel_grid_origin_x) / voxel_size);
        int pt_grid_y = roundf((p.y - voxel_grid_origin_y) / voxel_size);
        int pt_grid_z = roundf((p.z - voxel_grid_origin_z) / voxel_size);

        // Also need to copy the TSDF in the surrounding area
        for (int i = -5; i < lookup_range; i++)
        {
            int x_idx = pt_grid_x + i;
            if (x_idx > voxel_grid_dim_x || x_idx < min_x_idx)
                continue;
            for (int j = -5; j < lookup_range; j++)
            {
                int y_idx = pt_grid_y + j;
                if (y_idx > voxel_grid_dim_y || y_idx < min_y_idx)
                    continue;   
                for (int k = -5; k < lookup_range; k++)
                {
                    int z_idx = pt_grid_z + k;
                    if (z_idx > voxel_grid_dim_z || z_idx < min_z_idx)
                        continue;
                    int volume_idx = z_idx * voxel_grid_dim_y * voxel_grid_dim_x + y_idx * voxel_grid_dim_x + x_idx;
                    filterred_voxel_grid_TSDF[volume_idx] = voxel_grid_TSDF[volume_idx];
                }
            }
        }
    }
}

#endif