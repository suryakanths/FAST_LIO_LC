// PCL interface wrapper - compiled only with C++, not NVCC
#include "pgo_cuda_utils.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

#ifdef USE_CUDA

namespace pgo_cuda {
namespace cuda_impl {
// External C interface to pure CUDA kernels
extern "C" {
    cudaError_t cuda_find_correspondences(const float* d_source_x,
                                         const float* d_source_y,
                                         const float* d_source_z,
                                         const float* d_target_x,
                                         const float* d_target_y,
                                         const float* d_target_z,
                                         int* d_correspondence_indices,
                                         float* d_correspondence_distances,
                                         int num_source_points,
                                         int num_target_points,
                                         float max_distance);
    
    cudaError_t cuda_transform_point_cloud(const float* d_input_x,
                                          const float* d_input_y,
                                          const float* d_input_z,
                                          const float* d_input_intensity,
                                          float* d_output_x,
                                          float* d_output_y,
                                          float* d_output_z,
                                          float* d_output_intensity,
                                          const float* d_transform_matrix,
                                          int num_points);
    
    cudaError_t cuda_downsample_point_cloud(const float* d_input_x,
                                           const float* d_input_y,
                                           const float* d_input_z,
                                           const float* d_input_intensity,
                                           float* d_output_x,
                                           float* d_output_y,
                                           float* d_output_z,
                                           float* d_output_intensity,
                                           const int* d_voxel_indices,
                                           int num_input_points,
                                           int num_output_points,
                                           float voxel_size);

    cudaError_t cuda_compute_point_distances(const float* d_source_x,
                                            const float* d_source_y,
                                            const float* d_source_z,
                                            const float* d_target_x,
                                            const float* d_target_y,
                                            const float* d_target_z,
                                            float* d_distances,
                                            int num_points);

    cudaError_t cuda_voxel_grid_filter(const float* d_input_x,
                                      const float* d_input_y,
                                      const float* d_input_z,
                                      const float* d_input_intensity,
                                      float* d_output_x,
                                      float* d_output_y,
                                      float* d_output_z,
                                      float* d_output_intensity,
                                      int* d_voxel_indices,
                                      int num_input_points,
                                      float leaf_size_x,
                                      float leaf_size_y,
                                      float leaf_size_z);
}
} // namespace cuda_impl

// Implementation of CudaPGOProcessor using the external C interface
struct CudaPGOProcessor::CudaData {
    // GPU memory pointers
    float* d_points_x = nullptr;
    float* d_points_y = nullptr;
    float* d_points_z = nullptr;
    float* d_points_intensity = nullptr;
    
    float* d_target_x = nullptr;
    float* d_target_y = nullptr;
    float* d_target_z = nullptr;
    
    float* d_transform_matrix = nullptr;
    float* d_distances = nullptr;
    int* d_indices = nullptr;
    int* d_correspondence_indices = nullptr;
    float* d_correspondence_distances = nullptr;
    
    size_t allocated_size = 0;
    cudaStream_t stream;
};

CudaPGOProcessor::CudaPGOProcessor() 
    : cuda_data_(std::make_unique<CudaData>())
    , allocated_points_(0)
    , cuda_initialized_(false) {
    InitializeCuda();
}

CudaPGOProcessor::~CudaPGOProcessor() {
    CleanupCuda();
}

bool CudaPGOProcessor::IsCudaAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

bool CudaPGOProcessor::InitializeCuda() {
    if (cuda_initialized_) {
        return true;
    }
    
    if (!IsCudaAvailable()) {
        std::cout << "CUDA not available for PGO acceleration" << std::endl;
        return false;
    }
    
    cudaError_t error = cudaStreamCreate(&cuda_data_->stream);
    if (error != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    cuda_initialized_ = true;
    std::cout << "CUDA PGO processor initialized successfully" << std::endl;
    return true;
}

void CudaPGOProcessor::CleanupCuda() {
    if (!cuda_initialized_) {
        return;
    }
    
    DeallocateGpuMemory();
    
    if (cuda_data_->stream) {
        cudaStreamDestroy(cuda_data_->stream);
    }
    
    cuda_initialized_ = false;
}

bool CudaPGOProcessor::AllocateGpuMemory(size_t num_points) {
    if (num_points <= allocated_points_) {
        return true; // Already allocated enough memory
    }
    
    DeallocateGpuMemory();
    
    size_t size = num_points * sizeof(float);
    
    // Allocate memory for point coordinates and attributes
    cudaError_t error = cudaSuccess;
    
    cudaError_t err1 = cudaMalloc(&cuda_data_->d_points_x, size);
    cudaError_t err2 = cudaMalloc(&cuda_data_->d_points_y, size);
    cudaError_t err3 = cudaMalloc(&cuda_data_->d_points_z, size);
    cudaError_t err4 = cudaMalloc(&cuda_data_->d_points_intensity, size);
    
    cudaError_t err5 = cudaMalloc(&cuda_data_->d_target_x, size);
    cudaError_t err6 = cudaMalloc(&cuda_data_->d_target_y, size);
    cudaError_t err7 = cudaMalloc(&cuda_data_->d_target_z, size);
    
    cudaError_t err8 = cudaMalloc(&cuda_data_->d_transform_matrix, 16 * sizeof(float));
    cudaError_t err9 = cudaMalloc(&cuda_data_->d_distances, size);
    cudaError_t err10 = cudaMalloc(&cuda_data_->d_indices, num_points * sizeof(int));
    cudaError_t err11 = cudaMalloc(&cuda_data_->d_correspondence_indices, num_points * sizeof(int));
    cudaError_t err12 = cudaMalloc(&cuda_data_->d_correspondence_distances, size);
    
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess ||
        err5 != cudaSuccess || err6 != cudaSuccess || err7 != cudaSuccess || err8 != cudaSuccess ||
        err9 != cudaSuccess || err10 != cudaSuccess || err11 != cudaSuccess || err12 != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory" << std::endl;
        DeallocateGpuMemory();
        return false;
    }
    
    allocated_points_ = num_points;
    cuda_data_->allocated_size = size;
    return true;
}

void CudaPGOProcessor::DeallocateGpuMemory() {
    if (cuda_data_->d_points_x) cudaFree(cuda_data_->d_points_x);
    if (cuda_data_->d_points_y) cudaFree(cuda_data_->d_points_y);
    if (cuda_data_->d_points_z) cudaFree(cuda_data_->d_points_z);
    if (cuda_data_->d_points_intensity) cudaFree(cuda_data_->d_points_intensity);
    if (cuda_data_->d_target_x) cudaFree(cuda_data_->d_target_x);
    if (cuda_data_->d_target_y) cudaFree(cuda_data_->d_target_y);
    if (cuda_data_->d_target_z) cudaFree(cuda_data_->d_target_z);
    if (cuda_data_->d_transform_matrix) cudaFree(cuda_data_->d_transform_matrix);
    if (cuda_data_->d_distances) cudaFree(cuda_data_->d_distances);
    if (cuda_data_->d_indices) cudaFree(cuda_data_->d_indices);
    if (cuda_data_->d_correspondence_indices) cudaFree(cuda_data_->d_correspondence_indices);
    if (cuda_data_->d_correspondence_distances) cudaFree(cuda_data_->d_correspondence_distances);
    
    // Reset pointers to nullptr
    cuda_data_->d_points_x = nullptr;
    cuda_data_->d_points_y = nullptr;
    cuda_data_->d_points_z = nullptr;
    cuda_data_->d_points_intensity = nullptr;
    cuda_data_->d_target_x = nullptr;
    cuda_data_->d_target_y = nullptr;
    cuda_data_->d_target_z = nullptr;
    cuda_data_->d_transform_matrix = nullptr;
    cuda_data_->d_distances = nullptr;
    cuda_data_->d_indices = nullptr;
    cuda_data_->d_correspondence_indices = nullptr;
    cuda_data_->d_correspondence_distances = nullptr;
    
    allocated_points_ = 0;
}

void CudaPGOProcessor::ClearMemory() {
    DeallocateGpuMemory();
}

// Helper function to copy point cloud data to GPU
template<typename T>
bool copyPointCloudToGpu(const typename pcl::PointCloud<T>::Ptr& cloud,
                        float* d_x, float* d_y, float* d_z, float* d_intensity,
                        cudaStream_t stream) {
    if (!cloud || cloud->empty()) {
        return false;
    }
    
    std::vector<float> host_x, host_y, host_z, host_intensity;
    host_x.reserve(cloud->size());
    host_y.reserve(cloud->size());
    host_z.reserve(cloud->size());
    host_intensity.reserve(cloud->size());
    
    for (const auto& point : cloud->points) {
        host_x.push_back(point.x);
        host_y.push_back(point.y);
        host_z.push_back(point.z);
        host_intensity.push_back(point.intensity);
    }
    
    cudaError_t err1 = cudaMemcpyAsync(d_x, host_x.data(), cloud->size() * sizeof(float), 
                                     cudaMemcpyHostToDevice, stream);
    cudaError_t err2 = cudaMemcpyAsync(d_y, host_y.data(), cloud->size() * sizeof(float), 
                                     cudaMemcpyHostToDevice, stream);
    cudaError_t err3 = cudaMemcpyAsync(d_z, host_z.data(), cloud->size() * sizeof(float), 
                                     cudaMemcpyHostToDevice, stream);
    cudaError_t err4 = cudaMemcpyAsync(d_intensity, host_intensity.data(), cloud->size() * sizeof(float), 
                                     cudaMemcpyHostToDevice, stream);
    
    return (err1 == cudaSuccess && err2 == cudaSuccess && err3 == cudaSuccess && err4 == cudaSuccess);
}

bool CudaPGOProcessor::FindCorrespondences(const PointCloudType::Ptr& source_cloud,
                                          const PointCloudType::Ptr& target_cloud,
                                          std::vector<std::pair<int, int>>& correspondences,
                                          float max_distance) {
    if (!cuda_initialized_ || !source_cloud || !target_cloud || 
        source_cloud->empty() || target_cloud->empty()) {
        return false;
    }
    
    size_t max_points = std::max(source_cloud->size(), target_cloud->size());
    if (!AllocateGpuMemory(max_points)) {
        return false;
    }
    
    // Copy source cloud to GPU
    if (!copyPointCloudToGpu<PointType>(source_cloud, 
                                        cuda_data_->d_points_x,
                                        cuda_data_->d_points_y,
                                        cuda_data_->d_points_z,
                                        cuda_data_->d_points_intensity,
                                        cuda_data_->stream)) {
        return false;
    }
    
    // Copy target cloud to GPU
    if (!copyPointCloudToGpu<PointType>(target_cloud,
                                        cuda_data_->d_target_x,
                                        cuda_data_->d_target_y,
                                        cuda_data_->d_target_z,
                                        nullptr, // Don't need intensity for target
                                        cuda_data_->stream)) {
        return false;
    }
    
    // Call CUDA kernel
    cudaError_t error = cuda_impl::cuda_find_correspondences(
        cuda_data_->d_points_x, cuda_data_->d_points_y, cuda_data_->d_points_z,
        cuda_data_->d_target_x, cuda_data_->d_target_y, cuda_data_->d_target_z,
        cuda_data_->d_correspondence_indices,
        cuda_data_->d_correspondence_distances,
        static_cast<int>(source_cloud->size()),
        static_cast<int>(target_cloud->size()),
        max_distance
    );
    
    if (error != cudaSuccess) {
        std::cerr << "CUDA correspondence finding failed: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Copy results back to host
    std::vector<int> host_indices(source_cloud->size());
    std::vector<float> host_distances(source_cloud->size());
    
    cudaError_t err1 = cudaMemcpyAsync(host_indices.data(), cuda_data_->d_correspondence_indices,
                                     source_cloud->size() * sizeof(int), cudaMemcpyDeviceToHost,
                                     cuda_data_->stream);
    cudaError_t err2 = cudaMemcpyAsync(host_distances.data(), cuda_data_->d_correspondence_distances,
                                     source_cloud->size() * sizeof(float), cudaMemcpyDeviceToHost,
                                     cuda_data_->stream);
    
    cudaStreamSynchronize(cuda_data_->stream);
    
    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        return false;
    }
    
    // Process results
    correspondences.clear();
    for (size_t i = 0; i < source_cloud->size(); ++i) {
        if (host_indices[i] >= 0 && host_distances[i] <= max_distance) {
            correspondences.push_back(std::make_pair(static_cast<int>(i), host_indices[i]));
        }
    }
    
    return true;
}

bool CudaPGOProcessor::TransformPointCloud(const PointCloudType::Ptr& input_cloud,
                                          PointCloudType::Ptr& output_cloud,
                                          const Eigen::Matrix4f& transform) {
    if (!cuda_initialized_ || !input_cloud || input_cloud->empty()) {
        return false;
    }
    
    if (!AllocateGpuMemory(input_cloud->size())) {
        return false;
    }
    
    // Copy input cloud to GPU
    if (!copyPointCloudToGpu<PointType>(input_cloud,
                                        cuda_data_->d_points_x,
                                        cuda_data_->d_points_y,
                                        cuda_data_->d_points_z,
                                        cuda_data_->d_points_intensity,
                                        cuda_data_->stream)) {
        return false;
    }
    
    // Copy transformation matrix to GPU
    std::vector<float> transform_data;
    transform_data.reserve(16);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            transform_data.push_back(transform(i, j));
        }
    }
    
    cudaError_t error = cudaMemcpyAsync(cuda_data_->d_transform_matrix, transform_data.data(),
                                      16 * sizeof(float), cudaMemcpyHostToDevice,
                                      cuda_data_->stream);
    if (error != cudaSuccess) {
        return false;
    }
    
    // Call CUDA transformation kernel (using target arrays as output)
    error = cuda_impl::cuda_transform_point_cloud(
        cuda_data_->d_points_x, cuda_data_->d_points_y, cuda_data_->d_points_z,
        cuda_data_->d_points_intensity,
        cuda_data_->d_target_x, cuda_data_->d_target_y, cuda_data_->d_target_z,
        nullptr, // We'll copy intensity separately
        cuda_data_->d_transform_matrix,
        static_cast<int>(input_cloud->size())
    );
    
    if (error != cudaSuccess) {
        std::cerr << "CUDA transformation failed: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Copy results back to host
    std::vector<float> host_x(input_cloud->size());
    std::vector<float> host_y(input_cloud->size());
    std::vector<float> host_z(input_cloud->size());
    std::vector<float> host_intensity(input_cloud->size());
    
    cudaError_t err1 = cudaMemcpyAsync(host_x.data(), cuda_data_->d_target_x,
                                     input_cloud->size() * sizeof(float), cudaMemcpyDeviceToHost,
                                     cuda_data_->stream);
    cudaError_t err2 = cudaMemcpyAsync(host_y.data(), cuda_data_->d_target_y,
                                     input_cloud->size() * sizeof(float), cudaMemcpyDeviceToHost,
                                     cuda_data_->stream);
    cudaError_t err3 = cudaMemcpyAsync(host_z.data(), cuda_data_->d_target_z,
                                     input_cloud->size() * sizeof(float), cudaMemcpyDeviceToHost,
                                     cuda_data_->stream);
    cudaError_t err4 = cudaMemcpyAsync(host_intensity.data(), cuda_data_->d_points_intensity,
                                     input_cloud->size() * sizeof(float), cudaMemcpyDeviceToHost,
                                     cuda_data_->stream);
    
    cudaStreamSynchronize(cuda_data_->stream);
    
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess) {
        return false;
    }
    
    // Create output cloud
    output_cloud->clear();
    output_cloud->resize(input_cloud->size());
    
    for (size_t i = 0; i < input_cloud->size(); ++i) {
        output_cloud->points[i].x = host_x[i];
        output_cloud->points[i].y = host_y[i];
        output_cloud->points[i].z = host_z[i];
        output_cloud->points[i].intensity = host_intensity[i];
    }
    
    return true;
}

// Implement other methods similarly...
bool CudaPGOProcessor::DownsamplePointCloud(const PointCloudType::Ptr& input_cloud,
                                           PointCloudType::Ptr& output_cloud,
                                           float voxel_size) {
    // For now, fall back to CPU implementation for downsampling
    // This could be implemented with CUDA later
    return cuda_utils::DownsamplePointCloud(input_cloud, output_cloud, voxel_size);
}

bool CudaPGOProcessor::ComputePointToPointDistances(const PointCloudType::Ptr& source_cloud,
                                                   const PointCloudType::Ptr& target_cloud,
                                                   std::vector<float>& distances) {
    if (!cuda_initialized_ || !source_cloud || !target_cloud || 
        source_cloud->empty() || target_cloud->empty() ||
        source_cloud->size() != target_cloud->size()) {
        return false;
    }
    
    if (!AllocateGpuMemory(source_cloud->size())) {
        return false;
    }
    
    // Copy clouds to GPU
    if (!copyPointCloudToGpu<PointType>(source_cloud,
                                        cuda_data_->d_points_x,
                                        cuda_data_->d_points_y,
                                        cuda_data_->d_points_z,
                                        nullptr, cuda_data_->stream)) {
        return false;
    }
    
    if (!copyPointCloudToGpu<PointType>(target_cloud,
                                        cuda_data_->d_target_x,
                                        cuda_data_->d_target_y,
                                        cuda_data_->d_target_z,
                                        nullptr, cuda_data_->stream)) {
        return false;
    }
    
    // Call CUDA kernel
    cudaError_t error = cuda_impl::cuda_compute_point_distances(
        cuda_data_->d_points_x, cuda_data_->d_points_y, cuda_data_->d_points_z,
        cuda_data_->d_target_x, cuda_data_->d_target_y, cuda_data_->d_target_z,
        cuda_data_->d_distances,
        static_cast<int>(source_cloud->size())
    );
    
    if (error != cudaSuccess) {
        std::cerr << "CUDA distance computation failed: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Copy results back to host
    distances.resize(source_cloud->size());
    cudaError_t copy_error = cudaMemcpyAsync(distances.data(), cuda_data_->d_distances,
                                           source_cloud->size() * sizeof(float), cudaMemcpyDeviceToHost,
                                           cuda_data_->stream);
    
    cudaStreamSynchronize(cuda_data_->stream);
    
    return copy_error == cudaSuccess;
}

bool CudaPGOProcessor::VoxelGridFilter(const PointCloudType::Ptr& input_cloud,
                                      PointCloudType::Ptr& output_cloud,
                                      float leaf_size_x,
                                      float leaf_size_y,
                                      float leaf_size_z) {
    // For now, fall back to CPU implementation for voxel grid filtering
    // This could be implemented with CUDA later
    return cuda_utils::VoxelGridFilter(input_cloud, output_cloud, 
                                      leaf_size_x, leaf_size_y, leaf_size_z);
}

} // namespace pgo_cuda

#endif // USE_CUDA
