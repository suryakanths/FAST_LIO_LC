#ifndef PGO_CUDA_UTILS_H
#define PGO_CUDA_UTILS_H

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <memory>
#endif

// Include full PCL headers - safe in header when not compiling CUDA
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <vector>

namespace pgo_cuda {

using PointType = pcl::PointXYZI;
using PointCloudType = pcl::PointCloud<PointType>;

#ifdef USE_CUDA
/**
 * CUDA-accelerated point cloud processing for PGO
 * Provides GPU acceleration for pose graph optimization operations
 */
class CudaPGOProcessor {
public:
    CudaPGOProcessor();
    ~CudaPGOProcessor();

    // Static method to check CUDA availability
    static bool IsCudaAvailable();

    // Point cloud correspondence search (for scan matching)
    bool FindCorrespondences(const PointCloudType::Ptr& source_cloud,
                            const PointCloudType::Ptr& target_cloud,
                            std::vector<std::pair<int, int>>& correspondences,
                            float max_distance);

    // Point cloud transformation for scan matching
    bool TransformPointCloud(const PointCloudType::Ptr& input_cloud,
                            PointCloudType::Ptr& output_cloud,
                            const Eigen::Matrix4f& transform);

    // Point cloud downsampling for keyframe extraction
    bool DownsamplePointCloud(const PointCloudType::Ptr& input_cloud,
                             PointCloudType::Ptr& output_cloud,
                             float voxel_size);

    // ICP-like point-to-point distance computation
    bool ComputePointToPointDistances(const PointCloudType::Ptr& source_cloud,
                                     const PointCloudType::Ptr& target_cloud,
                                     std::vector<float>& distances);

    // Voxel grid filtering for loop closure
    bool VoxelGridFilter(const PointCloudType::Ptr& input_cloud,
                        PointCloudType::Ptr& output_cloud,
                        float leaf_size_x,
                        float leaf_size_y,
                        float leaf_size_z);

    // Memory management
    void ClearMemory();

private:
    struct CudaData;
    std::unique_ptr<CudaData> cuda_data_;
    
    bool InitializeCuda();
    void CleanupCuda();
    
    // GPU memory management with pooling
    bool AllocateGpuMemory(size_t num_points);
    void DeallocateGpuMemory();
    bool EnsureGpuMemory(size_t num_points); // Only reallocate if needed
    
    size_t allocated_points_;
    bool cuda_initialized_;
    
    // Performance monitoring
    mutable size_t total_operations_;
    mutable size_t cuda_operations_;
    mutable size_t cpu_fallbacks_;
};

// CUDA utility functions (CPU fallbacks when CUDA not available)
namespace cuda_utils {
    bool FindCorrespondences(const PointCloudType::Ptr& source_cloud,
                            const PointCloudType::Ptr& target_cloud,
                            std::vector<std::pair<int, int>>& correspondences,
                            float max_distance);
    
    bool TransformPointCloud(const PointCloudType::Ptr& input_cloud,
                            PointCloudType::Ptr& output_cloud,
                            const Eigen::Matrix4f& transform);
    
    bool DownsamplePointCloud(const PointCloudType::Ptr& input_cloud,
                             PointCloudType::Ptr& output_cloud,
                             float voxel_size);

    bool ComputePointToPointDistances(const PointCloudType::Ptr& source_cloud,
                                     const PointCloudType::Ptr& target_cloud,
                                     std::vector<float>& distances);

    bool VoxelGridFilter(const PointCloudType::Ptr& input_cloud,
                        PointCloudType::Ptr& output_cloud,
                        float leaf_size_x,
                        float leaf_size_y,
                        float leaf_size_z);
}

#else // !USE_CUDA

// Stub class when CUDA is not available
class CudaPGOProcessor {
public:
    CudaPGOProcessor() = default;
    ~CudaPGOProcessor() = default;

    static bool IsCudaAvailable() { return false; }

    bool FindCorrespondences(const PointCloudType::Ptr& source_cloud,
                            const PointCloudType::Ptr& target_cloud,
                            std::vector<std::pair<int, int>>& correspondences,
                            float max_distance) { return false; }

    bool TransformPointCloud(const PointCloudType::Ptr& input_cloud,
                            PointCloudType::Ptr& output_cloud,
                            const Eigen::Matrix4f& transform) { return false; }

    bool DownsamplePointCloud(const PointCloudType::Ptr& input_cloud,
                             PointCloudType::Ptr& output_cloud,
                             float voxel_size) { return false; }

    bool ComputePointToPointDistances(const PointCloudType::Ptr& source_cloud,
                                     const PointCloudType::Ptr& target_cloud,
                                     std::vector<float>& distances) { return false; }

    bool VoxelGridFilter(const PointCloudType::Ptr& input_cloud,
                        PointCloudType::Ptr& output_cloud,
                        float leaf_size_x,
                        float leaf_size_y,
                        float leaf_size_z) { return false; }

    void ClearMemory() {}
};

// CPU fallback implementations
namespace cuda_utils {
    bool FindCorrespondences(const PointCloudType::Ptr& source_cloud,
                            const PointCloudType::Ptr& target_cloud,
                            std::vector<std::pair<int, int>>& correspondences,
                            float max_distance);
    
    bool TransformPointCloud(const PointCloudType::Ptr& input_cloud,
                            PointCloudType::Ptr& output_cloud,
                            const Eigen::Matrix4f& transform);
    
    bool DownsamplePointCloud(const PointCloudType::Ptr& input_cloud,
                             PointCloudType::Ptr& output_cloud,
                             float voxel_size);

    bool ComputePointToPointDistances(const PointCloudType::Ptr& source_cloud,
                                     const PointCloudType::Ptr& target_cloud,
                                     std::vector<float>& distances);

    bool VoxelGridFilter(const PointCloudType::Ptr& input_cloud,
                        PointCloudType::Ptr& output_cloud,
                        float leaf_size_x,
                        float leaf_size_y,
                        float leaf_size_z);
}

#endif // USE_CUDA

} // namespace pgo_cuda

#endif // PGO_CUDA_UTILS_H
