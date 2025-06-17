#include "pgo_cuda_utils.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <iostream>

namespace pgo_cuda {
namespace cuda_utils {

bool FindCorrespondences(const PointCloudType::Ptr& source_cloud,
                        const PointCloudType::Ptr& target_cloud,
                        std::vector<std::pair<int, int>>& correspondences,
                        float max_distance) {
    if (!source_cloud || !target_cloud || source_cloud->empty() || target_cloud->empty()) {
        return false;
    }
    
    correspondences.clear();
    
    // Create KDTree for target cloud
    pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>());
    kdtree->setInputCloud(target_cloud);
    
    float max_dist_sq = max_distance * max_distance;
    
    for (size_t i = 0; i < source_cloud->size(); ++i) {
        std::vector<int> indices;
        std::vector<float> distances;
        
        if (kdtree->nearestKSearch(source_cloud->points[i], 1, indices, distances) > 0) {
            if (distances[0] <= max_dist_sq) {
                correspondences.push_back(std::make_pair(static_cast<int>(i), indices[0]));
            }
        }
    }
    
    return true;
}

bool TransformPointCloud(const PointCloudType::Ptr& input_cloud,
                        PointCloudType::Ptr& output_cloud,
                        const Eigen::Matrix4f& transform) {
    if (!input_cloud || input_cloud->empty()) {
        return false;
    }
    
    output_cloud->clear();
    output_cloud->resize(input_cloud->size());
    
    for (size_t i = 0; i < input_cloud->size(); ++i) {
        const auto& input_point = input_cloud->points[i];
        auto& output_point = output_cloud->points[i];
        
        Eigen::Vector4f point_homogeneous(input_point.x, input_point.y, input_point.z, 1.0f);
        Eigen::Vector4f transformed_point = transform * point_homogeneous;
        
        output_point.x = transformed_point.x();
        output_point.y = transformed_point.y();
        output_point.z = transformed_point.z();
        output_point.intensity = input_point.intensity;
    }
    
    return true;
}

bool DownsamplePointCloud(const PointCloudType::Ptr& input_cloud,
                         PointCloudType::Ptr& output_cloud,
                         float voxel_size) {
    if (!input_cloud || input_cloud->empty()) {
        return false;
    }
    
    pcl::VoxelGrid<PointType> voxel_filter;
    voxel_filter.setInputCloud(input_cloud);
    voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
    voxel_filter.filter(*output_cloud);
    
    return true;
}

bool ComputePointToPointDistances(const PointCloudType::Ptr& source_cloud,
                                 const PointCloudType::Ptr& target_cloud,
                                 std::vector<float>& distances) {
    if (!source_cloud || !target_cloud || source_cloud->empty() || target_cloud->empty()) {
        return false;
    }
    
    if (source_cloud->size() != target_cloud->size()) {
        return false;
    }
    
    distances.clear();
    distances.reserve(source_cloud->size());
    
    for (size_t i = 0; i < source_cloud->size(); ++i) {
        const auto& src_point = source_cloud->points[i];
        const auto& tgt_point = target_cloud->points[i];
        
        float dx = src_point.x - tgt_point.x;
        float dy = src_point.y - tgt_point.y;
        float dz = src_point.z - tgt_point.z;
        
        distances.push_back(std::sqrt(dx*dx + dy*dy + dz*dz));
    }
    
    return true;
}

bool VoxelGridFilter(const PointCloudType::Ptr& input_cloud,
                    PointCloudType::Ptr& output_cloud,
                    float leaf_size_x,
                    float leaf_size_y,
                    float leaf_size_z) {
    if (!input_cloud || input_cloud->empty()) {
        return false;
    }
    
    pcl::VoxelGrid<PointType> voxel_filter;
    voxel_filter.setInputCloud(input_cloud);
    voxel_filter.setLeafSize(leaf_size_x, leaf_size_y, leaf_size_z);
    voxel_filter.filter(*output_cloud);
    
    return true;
}

} // namespace cuda_utils
} // namespace pgo_cuda
