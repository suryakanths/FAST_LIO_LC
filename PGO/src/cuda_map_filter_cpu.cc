#include "cuda_map_filter.h"
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/transforms.h>

namespace cuda_map_filter {
namespace cpu_fallback {

bool ApplyFilters(const PointCloudType::Ptr& input_cloud,
                 PointCloudType::Ptr& output_cloud,
                 const FilterParams& params) {
    
    PointCloudType::Ptr filtered_cloud = input_cloud;
    
    // 1. Region of Interest filtering
    if (params.use_roi) {
        PointCloudType::Ptr roi_filtered(new PointCloudType());
        if (!ROIFilter(filtered_cloud, roi_filtered, params.roi_min, params.roi_max)) {
            return false;
        }
        filtered_cloud = roi_filtered;
    }
    
    // 2. Statistical outlier removal
    if (params.enable_outlier_removal) {
        PointCloudType::Ptr outlier_filtered(new PointCloudType());
        if (!StatisticalOutlierRemoval(filtered_cloud, outlier_filtered, 
                                      params.outlier_std_ratio, params.outlier_neighbors)) {
            return false;
        }
        filtered_cloud = outlier_filtered;
    }
    
    // 3. Radius outlier removal
    if (params.enable_radius_filtering) {
        PointCloudType::Ptr radius_filtered(new PointCloudType());
        if (!RadiusOutlierRemoval(filtered_cloud, radius_filtered, 
                                 params.radius_search, params.min_neighbors_in_radius)) {
            return false;
        }
        filtered_cloud = radius_filtered;
    }
    
    // 4. Voxel grid downsampling (should be last to preserve important features)
    if (params.enable_voxel_filtering) {
        PointCloudType::Ptr voxel_filtered(new PointCloudType());
        if (!VoxelGridFilter(filtered_cloud, voxel_filtered, params.voxel_size)) {
            return false;
        }
        filtered_cloud = voxel_filtered;
    }
    
    output_cloud = filtered_cloud;
    return true;
}

bool ROIFilter(const PointCloudType::Ptr& input_cloud,
               PointCloudType::Ptr& output_cloud,
               const geometry_msgs::Point& min_pt,
               const geometry_msgs::Point& max_pt) {
    
    if (!input_cloud || input_cloud->empty()) {
        output_cloud->clear();
        return true;
    }
    
    try {
        pcl::CropBox<PointType> crop_filter;
        crop_filter.setInputCloud(input_cloud);
        crop_filter.setMin(Eigen::Vector4f(min_pt.x, min_pt.y, min_pt.z, 1.0));
        crop_filter.setMax(Eigen::Vector4f(max_pt.x, max_pt.y, max_pt.z, 1.0));
        crop_filter.filter(*output_cloud);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "CPU ROI filter failed: " << e.what() << std::endl;
        return false;
    }
}

bool StatisticalOutlierRemoval(const PointCloudType::Ptr& input_cloud,
                              PointCloudType::Ptr& output_cloud,
                              float std_ratio,
                              int neighbors) {
    
    if (!input_cloud || input_cloud->empty()) {
        output_cloud->clear();
        return true;
    }
    
    if (input_cloud->size() < static_cast<size_t>(neighbors)) {
        std::cerr << "Not enough points for statistical outlier removal. Skipping." << std::endl;
        *output_cloud = *input_cloud;
        return true;
    }
    
    try {
        pcl::StatisticalOutlierRemoval<PointType> sor;
        sor.setInputCloud(input_cloud);
        sor.setMeanK(neighbors);
        sor.setStddevMulThresh(std_ratio);
        sor.filter(*output_cloud);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "CPU statistical outlier removal failed: " << e.what() << std::endl;
        return false;
    }
}

bool RadiusOutlierRemoval(const PointCloudType::Ptr& input_cloud,
                         PointCloudType::Ptr& output_cloud,
                         float radius,
                         int min_neighbors) {
    
    if (!input_cloud || input_cloud->empty()) {
        output_cloud->clear();
        return true;
    }
    
    try {
        pcl::RadiusOutlierRemoval<PointType> ror;
        ror.setInputCloud(input_cloud);
        ror.setRadiusSearch(radius);
        ror.setMinNeighborsInRadius(min_neighbors);
        ror.filter(*output_cloud);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "CPU radius outlier removal failed: " << e.what() << std::endl;
        return false;
    }
}

bool VoxelGridFilter(const PointCloudType::Ptr& input_cloud,
                    PointCloudType::Ptr& output_cloud,
                    float voxel_size) {
    
    if (!input_cloud || input_cloud->empty()) {
        output_cloud->clear();
        return true;
    }
    
    try {
        pcl::VoxelGrid<PointType> vg;
        vg.setInputCloud(input_cloud);
        vg.setLeafSize(voxel_size, voxel_size, voxel_size);
        vg.filter(*output_cloud);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "CPU voxel grid filter failed: " << e.what() << std::endl;
        return false;
    }
}

} // namespace cpu_fallback
} // namespace cuda_map_filter
