#ifndef CUDA_MAP_FILTER_H
#define CUDA_MAP_FILTER_H

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <memory>
#endif

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <Eigen/Core>
#include <geometry_msgs/Point.h>
#include <vector>
#include <chrono>

namespace cuda_map_filter {

using PointType = pcl::PointXYZI;
using PointCloudType = pcl::PointCloud<PointType>;

// Platform-specific constants
namespace models {
    // Jetson Orin optimizations
    constexpr int ORIN_OPTIMAL_BLOCK_SIZE = 512;
    constexpr float ORIN_MEMORY_FRACTION = 0.85f;
    
    // Jetson Xavier optimizations  
    constexpr int XAVIER_OPTIMAL_BLOCK_SIZE = 512;
    constexpr float XAVIER_MEMORY_FRACTION = 0.8f;
    
    // Jetson TX2 optimizations
    constexpr int TX2_OPTIMAL_BLOCK_SIZE = 256;
    constexpr float TX2_MEMORY_FRACTION = 0.7f;
    
    // Jetson Nano optimizations
    constexpr int NANO_OPTIMAL_BLOCK_SIZE = 128;
    constexpr float NANO_MEMORY_FRACTION = 0.6f;
    
    // Desktop GPU optimizations
    constexpr int DESKTOP_OPTIMAL_BLOCK_SIZE = 256;
    constexpr float DESKTOP_MEMORY_FRACTION = 0.9f;
}

/**
 * Filter parameters structure
 */
struct FilterParams {
    // ROI filtering
    bool use_roi;
    geometry_msgs::Point roi_min;
    geometry_msgs::Point roi_max;
    
    // Statistical outlier removal
    bool enable_outlier_removal;
    float outlier_std_ratio;
    int outlier_neighbors;
    
    // Radius outlier removal
    bool enable_radius_filtering;
    float radius_search;
    int min_neighbors_in_radius;
    
    // Voxel grid filtering
    bool enable_voxel_filtering;
    float voxel_size;
    
    FilterParams() : use_roi(false), enable_outlier_removal(false),
                     outlier_std_ratio(1.0f), outlier_neighbors(50),
                     enable_radius_filtering(false), radius_search(0.5f),
                     min_neighbors_in_radius(5), enable_voxel_filtering(false),
                     voxel_size(0.1f) {}
};

/**
 * Platform detection structure for optimal CUDA configuration
 */
struct JetsonInfo {
    bool is_jetson;
    std::string model;                    // "NVIDIA Jetson Xavier NX"
    int compute_capability_major;         // 7 for Xavier, 8 for Orin
    int compute_capability_minor;         // 2 for Xavier, 7 for Orin
    size_t total_memory_mb;
    bool supports_unified_memory;
    
    JetsonInfo() : is_jetson(false), compute_capability_major(0), 
                   compute_capability_minor(0), total_memory_mb(0), 
                   supports_unified_memory(false) {}
};

/**
 * Platform-specific kernel configuration
 */
struct KernelConfig {
    int block_size;
    int grid_size_multiplier;
    bool use_unified_memory;
    bool enable_concurrent_streams;
    float memory_pool_fraction;
    
    KernelConfig() : block_size(256), grid_size_multiplier(1), 
                     use_unified_memory(false), enable_concurrent_streams(true),
                     memory_pool_fraction(0.8f) {}
};

/**
 * RAII CUDA memory management template
 */
template<typename T>
class CudaManagedMemory {
public:
    explicit CudaManagedMemory(size_t count, bool use_unified = false);
    ~CudaManagedMemory();
    
    // Non-copyable but movable
    CudaManagedMemory(const CudaManagedMemory&) = delete;
    CudaManagedMemory& operator=(const CudaManagedMemory&) = delete;
    CudaManagedMemory(CudaManagedMemory&& other) noexcept;
    CudaManagedMemory& operator=(CudaManagedMemory&& other) noexcept;
    
    T* get() const { return ptr_; }
    size_t size() const { return count_; }
    bool is_unified() const { return unified_; }
    
    bool CopyToDevice(const T* host_data, size_t count);
    bool CopyFromDevice(T* host_data, size_t count);
    
private:
    void Allocate();
    void Deallocate();
    
    T* ptr_;
    size_t count_;
    size_t byte_size_;
    bool unified_;
};

#ifdef USE_CUDA
/**
 * CUDA-accelerated map filtering processor with Jetson optimizations
 */
class CudaMapFilter {
public:
    CudaMapFilter();
    ~CudaMapFilter();

    // Static method to check CUDA availability
    static bool IsCudaAvailable();
    
    // Main filtering interface - applies all requested filters in optimal order
    bool ApplyFilters(const PointCloudType::Ptr& input_cloud,
                     PointCloudType::Ptr& output_cloud,
                     const FilterParams& params);
    
    // Individual filter methods (can be used separately)
    bool ROIFilter(const PointCloudType::Ptr& input_cloud,
                   PointCloudType::Ptr& output_cloud,
                   const geometry_msgs::Point& min_pt,
                   const geometry_msgs::Point& max_pt);
    
    bool StatisticalOutlierRemoval(const PointCloudType::Ptr& input_cloud,
                                  PointCloudType::Ptr& output_cloud,
                                  float std_ratio,
                                  int neighbors);
    
    bool RadiusOutlierRemoval(const PointCloudType::Ptr& input_cloud,
                             PointCloudType::Ptr& output_cloud,
                             float radius,
                             int min_neighbors);
    
    bool VoxelGridFilter(const PointCloudType::Ptr& input_cloud,
                        PointCloudType::Ptr& output_cloud,
                        float voxel_size);
    
    // Performance monitoring
    struct PerformanceStats {
        size_t total_operations;
        size_t cuda_operations;
        size_t cpu_fallbacks;
        double average_processing_time;
        size_t total_points_processed;
        
        PerformanceStats() : total_operations(0), cuda_operations(0), 
                           cpu_fallbacks(0), average_processing_time(0.0),
                           total_points_processed(0) {}
    };
    
    PerformanceStats GetPerformanceStats() const;
    void ResetPerformanceStats();
    
    // Memory management
    void ClearMemory();
    size_t GetAllocatedMemory() const;

private:
    struct CudaFilterData;
    std::unique_ptr<CudaFilterData> cuda_data_;
    
    JetsonInfo jetson_info_;
    KernelConfig kernel_config_;
    
    bool InitializeCuda();
    void CleanupCuda();
    
    JetsonInfo DetectJetsonPlatform();
    KernelConfig GetOptimalKernelConfig(const JetsonInfo& info);
    
    // GPU memory management
    bool AllocateGpuMemory(size_t num_points);
    void DeallocateGpuMemory();
    bool EnsureGpuMemory(size_t num_points);
    
    // Thermal monitoring for Jetson
    bool CheckThermalThrottling();
    float ReadThermalZoneTemp(const std::string& zone_name);
    
    // Batch processing for memory-constrained platforms
    size_t GetOptimalBatchSize(size_t input_points, size_t point_byte_size);
    bool ProcessInBatches(const PointCloudType::Ptr& input_cloud,
                         PointCloudType::Ptr& output_cloud,
                         size_t batch_size,
                         std::function<bool(const PointCloudType::Ptr&, PointCloudType::Ptr&)> filter_func);
    
    // Performance tracking
    mutable PerformanceStats perf_stats_;
    void UpdatePerformanceStats(bool cuda_used, double processing_time, size_t points_processed);
    
    size_t allocated_points_;
    bool cuda_initialized_;
    bool thermal_throttle_detected_;
};

#endif // USE_CUDA

// CPU fallback implementations
namespace cpu_fallback {
    bool ApplyFilters(const PointCloudType::Ptr& input_cloud,
                     PointCloudType::Ptr& output_cloud,
                     const FilterParams& params);
    
    bool ROIFilter(const PointCloudType::Ptr& input_cloud,
                   PointCloudType::Ptr& output_cloud,
                   const geometry_msgs::Point& min_pt,
                   const geometry_msgs::Point& max_pt);
    
    bool StatisticalOutlierRemoval(const PointCloudType::Ptr& input_cloud,
                                  PointCloudType::Ptr& output_cloud,
                                  float std_ratio,
                                  int neighbors);
    
    bool RadiusOutlierRemoval(const PointCloudType::Ptr& input_cloud,
                             PointCloudType::Ptr& output_cloud,
                             float radius,
                             int min_neighbors);
    
    bool VoxelGridFilter(const PointCloudType::Ptr& input_cloud,
                        PointCloudType::Ptr& output_cloud,
                        float voxel_size);
}

} // namespace cuda_map_filter

#endif // CUDA_MAP_FILTER_H
