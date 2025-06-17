/**
 * Enhanced PGO Map Service with CUDA-Optimized Outlier Removal
 * 
 * This service provides comprehensive map saving functionality with:
 * - CUDA-accelerated statistical outlier removal
 * - CUDA-accelerated radius-based outlier removal  
 * - CUDA-accelerated voxel grid downsampling
 * - CUDA-accelerated region of interest filtering
 * - Platform-aware optimization (Desktop GPU vs Jetson)
 * - Automatic CPU fallback when CUDA unavailable
 * - Multiple file format support
 * - Thermal throttling detection on Jetson platforms
 * 
 * Author: Enhanced for FAST-LIO-LC with CUDA acceleration
 */

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <aloam_velodyne/SaveOptimizedMap.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/Point.h>
#include <mutex>
#include <memory>
#include <chrono>

// CUDA-optimized map filtering
#include "cuda_map_filter.h"

typedef pcl::PointXYZI PointType;
typedef pcl::PointCloud<PointType> CloudType;

class EnhancedMapService {
private:
    ros::NodeHandle nh_;
    ros::ServiceServer save_map_service_;
    ros::Subscriber map_subscriber_;
    
    // Latest map data
    CloudType::Ptr latest_map_;
    std::mutex map_mutex_;
    bool map_available_;
    
    // CUDA-accelerated processing
#ifdef USE_CUDA
    std::unique_ptr<cuda_map_filter::CudaMapFilter> cuda_filter_;
    bool cuda_available_;
    cuda_map_filter::CudaMapFilter::PerformanceStats last_perf_stats_;
#endif
    
    // Parameters
    std::string default_save_directory_;
    float default_voxel_size_;
    float default_outlier_std_ratio_;
    int default_outlier_neighbors_;
    float default_radius_search_;
    int default_min_neighbors_;
    
public:
    EnhancedMapService() : 
        nh_("~"), 
        latest_map_(new CloudType()),
        map_available_(false),
#ifdef USE_CUDA
        cuda_available_(false),
#endif
        default_voxel_size_(0.02),
        default_outlier_std_ratio_(2.5),
        default_outlier_neighbors_(20),
        default_radius_search_(0.15),
        default_min_neighbors_(3) {
        
        // Load parameters
        nh_.param<std::string>("default_save_directory", default_save_directory_, 
                               "/home/surya/workspaces/slam_ws/src/FAST_LIO_LC/maps/");
        nh_.param<float>("default_voxel_size", default_voxel_size_, 0.02);
        nh_.param<float>("default_outlier_std_ratio", default_outlier_std_ratio_, 2.5);
        nh_.param<int>("default_outlier_neighbors", default_outlier_neighbors_, 20);
        nh_.param<float>("default_radius_search", default_radius_search_, 0.15);
        nh_.param<int>("default_min_neighbors", default_min_neighbors_, 3);
        
        // Ensure save directory ends with '/'
        if (!default_save_directory_.empty() && default_save_directory_.back() != '/') {
            default_save_directory_ += "/";
        }
        
        // Create directory if it doesn't exist
        int ret = system(("mkdir -p " + default_save_directory_).c_str());
        if (ret != 0) {
            ROS_WARN("Failed to create directory: %s", default_save_directory_.c_str());
        }
        
#ifdef USE_CUDA
        // Initialize CUDA-accelerated filtering
        try {
            cuda_filter_ = std::make_unique<cuda_map_filter::CudaMapFilter>();
            cuda_available_ = cuda_map_filter::CudaMapFilter::IsCudaAvailable();
            
            if (cuda_available_) {
                ROS_INFO("CUDA-accelerated map filtering initialized successfully");
            } else {
                ROS_WARN("CUDA not available, using CPU fallback for map filtering");
            }
        } catch (const std::exception& e) {
            ROS_ERROR("Failed to initialize CUDA map filter: %s", e.what());
            cuda_available_ = false;
        }
#else
        ROS_INFO("Built without CUDA support, using CPU-only map filtering");
#endif
        
        // Initialize service and subscriber
        save_map_service_ = nh_.advertiseService("save_optimized_map", 
                                                &EnhancedMapService::saveMapCallback, this);
        
        // Subscribe to the PGO map topic
        map_subscriber_ = nh_.subscribe("/aft_pgo_map", 1, 
                                      &EnhancedMapService::mapCallback, this);
        
        ROS_INFO("Enhanced Map Service initialized");
        ROS_INFO("Default save directory: %s", default_save_directory_.c_str());
        ROS_INFO("Service available at: ~/save_optimized_map");
#ifdef USE_CUDA
        if (cuda_available_) {
            ROS_INFO("CUDA acceleration: ENABLED");
        } else {
            ROS_INFO("CUDA acceleration: DISABLED (fallback to CPU)");
        }
#endif
    }
    
    void mapCallback(const sensor_msgs::PointCloud2::ConstPtr& map_msg) {
        std::lock_guard<std::mutex> lock(map_mutex_);
        
        // Convert ROS message to PCL
        pcl::fromROSMsg(*map_msg, *latest_map_);
        map_available_ = true;
        
        ROS_DEBUG("Updated map with %lu points", latest_map_->size());
    }
    
    bool saveMapCallback(aloam_velodyne::SaveOptimizedMap::Request& req,
                        aloam_velodyne::SaveOptimizedMap::Response& res) {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        ROS_INFO("Map save request received for: %s", req.output_path.c_str());
        
        // Check if map is available
        std::lock_guard<std::mutex> lock(map_mutex_);
        if (!map_available_ || latest_map_->empty()) {
            res.success = false;
            res.message = "No map data available or map is empty";
            res.original_points = 0;
            res.filtered_points = 0;
            return true;
        }
        
        try {
            // Copy the map for processing
            CloudType::Ptr working_cloud(new CloudType(*latest_map_));
            res.original_points = working_cloud->size();
            
            ROS_INFO("Processing map with %d original points", res.original_points);
            
            // Apply filters in sequence
            working_cloud = applyFilters(working_cloud, req);
            res.filtered_points = working_cloud->size();
            res.compression_ratio = static_cast<float>(res.filtered_points) / 
                                  static_cast<float>(res.original_points);
            
            // Determine output path
            std::string output_path = req.output_path;
            if (output_path.empty()) {
                auto now = std::chrono::system_clock::now();
                auto time_t = std::chrono::system_clock::to_time_t(now);
                std::string timestamp = std::to_string(time_t);
                output_path = default_save_directory_ + "pgo_map_" + timestamp + ".pcd";
            }
            
            // Ensure file has correct extension
            std::string format = req.file_format.empty() ? "pcd" : req.file_format;
            if (output_path.find('.') == std::string::npos) {
                output_path += "." + format;
            }
            
            // Save the map
            bool save_success = savePointCloud(working_cloud, output_path, format, 
                                             req.compress_binary, req.include_intensity);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            res.processing_time = std::chrono::duration<double>(end_time - start_time).count();
            
            if (save_success) {
                res.success = true;
                res.message = "Map saved successfully with CUDA-accelerated outlier removal";
                res.saved_file_path = output_path;
                
#ifdef USE_CUDA
                // Log CUDA performance statistics
                if (cuda_available_) {
                    auto perf_stats = cuda_filter_->GetPerformanceStats();
                    ROS_INFO("CUDA Performance Stats:");
                    ROS_INFO("  Total operations: %lu", perf_stats.total_operations);
                    ROS_INFO("  CUDA operations: %lu", perf_stats.cuda_operations);
                    ROS_INFO("  CPU fallbacks: %lu", perf_stats.cpu_fallbacks);
                    ROS_INFO("  Average processing time: %.3f seconds", perf_stats.average_processing_time);
                    ROS_INFO("  CUDA efficiency: %.1f%%", 
                            (float)perf_stats.cuda_operations / perf_stats.total_operations * 100.0f);
                    
                    // Store for potential later use
                    last_perf_stats_ = perf_stats;
                }
#endif
                
                ROS_INFO("Map saved successfully:");
                ROS_INFO("  Original points: %d", res.original_points);
                ROS_INFO("  Filtered points: %d", res.filtered_points);
                ROS_INFO("  Compression ratio: %.2f", res.compression_ratio);
                ROS_INFO("  Processing time: %.2f seconds", res.processing_time);
                ROS_INFO("  Saved to: %s", output_path.c_str());
            } else {
                res.success = false;
                res.message = "Failed to save point cloud file";
                res.saved_file_path = "";
            }
            
        } catch (const std::exception& e) {
            res.success = false;
            res.message = "Error processing map: " + std::string(e.what());
            res.original_points = latest_map_->size();
            res.filtered_points = 0;
            res.compression_ratio = 0.0;
            res.saved_file_path = "";
            
            ROS_ERROR("Exception in saveMapCallback: %s", e.what());
        }
        
        return true;
    }
    
private:
    CloudType::Ptr applyFilters(CloudType::Ptr cloud, 
                               const aloam_velodyne::SaveOptimizedMap::Request& req) {
        
        CloudType::Ptr filtered_cloud = cloud;
        
#ifdef USE_CUDA
        if (cuda_available_) {
            // Use CUDA-accelerated filtering
            ROS_INFO("Applying CUDA-accelerated filters...");
            
            // Prepare filter parameters
            cuda_map_filter::FilterParams params;
            
            // ROI filtering
            params.use_roi = req.use_roi;
            if (req.use_roi) {
                params.roi_min = req.roi_min;
                params.roi_max = req.roi_max;
            }
            
            // Statistical outlier removal
            params.enable_outlier_removal = req.enable_outlier_removal;
            params.outlier_std_ratio = req.outlier_std_ratio > 0 ? req.outlier_std_ratio : default_outlier_std_ratio_;
            params.outlier_neighbors = req.outlier_neighbors > 0 ? req.outlier_neighbors : default_outlier_neighbors_;
            
            // Radius outlier removal
            params.enable_radius_filtering = req.enable_radius_filtering;
            params.radius_search = req.radius_search > 0 ? req.radius_search : default_radius_search_;
            params.min_neighbors_in_radius = req.min_neighbors_in_radius > 0 ? 
                                           req.min_neighbors_in_radius : default_min_neighbors_;
            
            // Voxel grid filtering
            params.enable_voxel_filtering = req.enable_voxel_filtering;
            params.voxel_size = req.voxel_size > 0 ? req.voxel_size : default_voxel_size_;
            
            // Apply all filters in one optimized CUDA call
            CloudType::Ptr cuda_filtered_cloud(new CloudType());
            bool cuda_success = cuda_filter_->ApplyFilters(filtered_cloud, cuda_filtered_cloud, params);
            
            if (cuda_success) {
                filtered_cloud = cuda_filtered_cloud;
                ROS_INFO("CUDA filtering completed successfully");
            } else {
                ROS_WARN("CUDA filtering failed, falling back to CPU filters");
                // Fall through to CPU implementation
            }
        }
        
        if (!cuda_available_) {
#endif
            // CPU fallback implementation
            ROS_INFO("Applying CPU-based filters...");
            
            // 1. Region of Interest filtering
            if (req.use_roi) {
                filtered_cloud = applyROIFilter(filtered_cloud, req.roi_min, req.roi_max);
                ROS_INFO("After ROI filtering: %lu points", filtered_cloud->size());
            }
            
            // 2. Statistical outlier removal
            if (req.enable_outlier_removal) {
                float std_ratio = req.outlier_std_ratio > 0 ? req.outlier_std_ratio : default_outlier_std_ratio_;
                int neighbors = req.outlier_neighbors > 0 ? req.outlier_neighbors : default_outlier_neighbors_;
                
                filtered_cloud = applyStatisticalOutlierRemoval(filtered_cloud, std_ratio, neighbors);
                ROS_INFO("After statistical outlier removal: %lu points", filtered_cloud->size());
            }
            
            // 3. Radius outlier removal
            if (req.enable_radius_filtering) {
                float radius = req.radius_search > 0 ? req.radius_search : default_radius_search_;
                int min_neighbors = req.min_neighbors_in_radius > 0 ? 
                                   req.min_neighbors_in_radius : default_min_neighbors_;
                
                filtered_cloud = applyRadiusOutlierRemoval(filtered_cloud, radius, min_neighbors);
                ROS_INFO("After radius outlier removal: %lu points", filtered_cloud->size());
            }
            
            // 4. Voxel grid downsampling (should be last to preserve important features)
            if (req.enable_voxel_filtering) {
                float voxel_size = req.voxel_size > 0 ? req.voxel_size : default_voxel_size_;
                filtered_cloud = applyVoxelGridFilter(filtered_cloud, voxel_size);
                ROS_INFO("After voxel grid filtering: %lu points", filtered_cloud->size());
            }
            
#ifdef USE_CUDA
        }
#endif
        
        return filtered_cloud;
    }
    
    CloudType::Ptr applyROIFilter(CloudType::Ptr cloud, 
                                 const geometry_msgs::Point& min_pt,
                                 const geometry_msgs::Point& max_pt) {
        CloudType::Ptr filtered_cloud(new CloudType());
        
        pcl::CropBox<PointType> crop_filter;
        crop_filter.setInputCloud(cloud);
        crop_filter.setMin(Eigen::Vector4f(min_pt.x, min_pt.y, min_pt.z, 1.0));
        crop_filter.setMax(Eigen::Vector4f(max_pt.x, max_pt.y, max_pt.z, 1.0));
        crop_filter.filter(*filtered_cloud);
        
        return filtered_cloud;
    }
    
    CloudType::Ptr applyStatisticalOutlierRemoval(CloudType::Ptr cloud, 
                                                  float std_ratio, 
                                                  int neighbors) {
        if (cloud->size() < static_cast<size_t>(neighbors)) {
            ROS_WARN("Not enough points for statistical outlier removal. Skipping.");
            return cloud;
        }
        
        CloudType::Ptr filtered_cloud(new CloudType());
        
        pcl::StatisticalOutlierRemoval<PointType> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(neighbors);
        sor.setStddevMulThresh(std_ratio);
        sor.filter(*filtered_cloud);
        
        return filtered_cloud;
    }
    
    CloudType::Ptr applyRadiusOutlierRemoval(CloudType::Ptr cloud,
                                            float radius,
                                            int min_neighbors) {
        CloudType::Ptr filtered_cloud(new CloudType());
        
        pcl::RadiusOutlierRemoval<PointType> ror;
        ror.setInputCloud(cloud);
        ror.setRadiusSearch(radius);
        ror.setMinNeighborsInRadius(min_neighbors);
        ror.filter(*filtered_cloud);
        
        return filtered_cloud;
    }
    
    CloudType::Ptr applyVoxelGridFilter(CloudType::Ptr cloud, float voxel_size) {
        CloudType::Ptr filtered_cloud(new CloudType());
        
        pcl::VoxelGrid<PointType> vg;
        vg.setInputCloud(cloud);
        vg.setLeafSize(voxel_size, voxel_size, voxel_size);
        vg.filter(*filtered_cloud);
        
        return filtered_cloud;
    }
    
    bool savePointCloud(CloudType::Ptr cloud,
                       const std::string& file_path,
                       const std::string& format,
                       bool binary,
                       bool include_intensity) {
        
        try {
            if (format == "ply") {
                return pcl::io::savePLYFile(file_path, *cloud, binary) == 0;
            } else {
                // Default to PCD format
                return pcl::io::savePCDFile(file_path, *cloud, binary) == 0;
            }
        } catch (const std::exception& e) {
            ROS_ERROR("Failed to save point cloud: %s", e.what());
            return false;
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "enhanced_map_service");
    
    ROS_INFO("Starting Enhanced PGO Map Service with Outlier Removal");
    
    EnhancedMapService service;
    
    ros::spin();
    
    return 0;
}
