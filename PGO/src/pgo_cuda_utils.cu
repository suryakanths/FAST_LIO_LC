// CUDA utility functions - compiled with NVCC
#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cfloat>

namespace pgo_cuda {
namespace cuda_impl {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(error)); \
            return error; \
        } \
    } while(0)

// Device function to compute squared distance between two 3D points
__device__ inline float squared_distance(float x1, float y1, float z1,
                                        float x2, float y2, float z2) {
    float dx = x1 - x2;
    float dy = y1 - y2;
    float dz = z1 - z2;
    return dx*dx + dy*dy + dz*dz;
}

// CUDA kernel for finding correspondences between two point clouds
__global__ void find_correspondences_kernel(const float* __restrict__ source_x,
                                           const float* __restrict__ source_y,
                                           const float* __restrict__ source_z,
                                           const float* __restrict__ target_x,
                                           const float* __restrict__ target_y,
                                           const float* __restrict__ target_z,
                                           int* __restrict__ correspondence_indices,
                                           float* __restrict__ correspondence_distances,
                                           int num_source_points,
                                           int num_target_points,
                                           float max_distance_sq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_source_points) {
        return;
    }
    
    float src_x = source_x[idx];
    float src_y = source_y[idx];
    float src_z = source_z[idx];
    
    float min_dist_sq = FLT_MAX;
    int best_match = -1;
    
    // Simple linear search for nearest neighbor
    for (int i = 0; i < num_target_points; ++i) {
        float dist_sq = squared_distance(src_x, src_y, src_z,
                                       target_x[i], target_y[i], target_z[i]);
        
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            best_match = i;
        }
    }
    
    // Store result if within threshold
    if (min_dist_sq <= max_distance_sq) {
        correspondence_indices[idx] = best_match;
        correspondence_distances[idx] = sqrtf(min_dist_sq);
    } else {
        correspondence_indices[idx] = -1;
        correspondence_distances[idx] = FLT_MAX;
    }
}

// CUDA kernel for point cloud transformation
__global__ void transform_point_cloud_kernel(const float* __restrict__ input_x,
                                            const float* __restrict__ input_y,
                                            const float* __restrict__ input_z,
                                            const float* __restrict__ input_intensity,
                                            float* __restrict__ output_x,
                                            float* __restrict__ output_y,
                                            float* __restrict__ output_z,
                                            float* __restrict__ output_intensity,
                                            const float* __restrict__ transform_matrix,
                                            int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) {
        return;
    }
    
    float x = input_x[idx];
    float y = input_y[idx];
    float z = input_z[idx];
    
    // Apply 4x4 transformation matrix
    // Transform matrix is stored in row-major order
    output_x[idx] = transform_matrix[0] * x + transform_matrix[1] * y + 
                   transform_matrix[2] * z + transform_matrix[3];
    output_y[idx] = transform_matrix[4] * x + transform_matrix[5] * y + 
                   transform_matrix[6] * z + transform_matrix[7];
    output_z[idx] = transform_matrix[8] * x + transform_matrix[9] * y + 
                   transform_matrix[10] * z + transform_matrix[11];
    
    // Copy intensity if provided
    if (output_intensity && input_intensity) {
        output_intensity[idx] = input_intensity[idx];
    }
}

// CUDA kernel for computing point-to-point distances
__global__ void compute_point_distances_kernel(const float* __restrict__ source_x,
                                              const float* __restrict__ source_y,
                                              const float* __restrict__ source_z,
                                              const float* __restrict__ target_x,
                                              const float* __restrict__ target_y,
                                              const float* __restrict__ target_z,
                                              float* __restrict__ distances,
                                              int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) {
        return;
    }
    
    float dist_sq = squared_distance(source_x[idx], source_y[idx], source_z[idx],
                                   target_x[idx], target_y[idx], target_z[idx]);
    distances[idx] = sqrtf(dist_sq);
}

} // namespace cuda_impl
} // namespace pgo_cuda

// C interface functions
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
                                     float max_distance) {
    
    const int threads_per_block = 256;
    const int blocks = (num_source_points + threads_per_block - 1) / threads_per_block;
    
    float max_distance_sq = max_distance * max_distance;
    
    pgo_cuda::cuda_impl::find_correspondences_kernel<<<blocks, threads_per_block>>>(
        d_source_x, d_source_y, d_source_z,
        d_target_x, d_target_y, d_target_z,
        d_correspondence_indices, d_correspondence_distances,
        num_source_points, num_target_points, max_distance_sq
    );
    
    return cudaGetLastError();
}

cudaError_t cuda_transform_point_cloud(const float* d_input_x,
                                      const float* d_input_y,
                                      const float* d_input_z,
                                      const float* d_input_intensity,
                                      float* d_output_x,
                                      float* d_output_y,
                                      float* d_output_z,
                                      float* d_output_intensity,
                                      const float* d_transform_matrix,
                                      int num_points) {
    
    const int threads_per_block = 256;
    const int blocks = (num_points + threads_per_block - 1) / threads_per_block;
    
    pgo_cuda::cuda_impl::transform_point_cloud_kernel<<<blocks, threads_per_block>>>(
        d_input_x, d_input_y, d_input_z, d_input_intensity,
        d_output_x, d_output_y, d_output_z, d_output_intensity,
        d_transform_matrix, num_points
    );
    
    return cudaGetLastError();
}

cudaError_t cuda_compute_point_distances(const float* d_source_x,
                                        const float* d_source_y,
                                        const float* d_source_z,
                                        const float* d_target_x,
                                        const float* d_target_y,
                                        const float* d_target_z,
                                        float* d_distances,
                                        int num_points) {
    
    const int threads_per_block = 256;
    const int blocks = (num_points + threads_per_block - 1) / threads_per_block;
    
    pgo_cuda::cuda_impl::compute_point_distances_kernel<<<blocks, threads_per_block>>>(
        d_source_x, d_source_y, d_source_z,
        d_target_x, d_target_y, d_target_z,
        d_distances, num_points
    );
    
    return cudaGetLastError();
}

// Placeholder implementations for functions not yet implemented
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
                                       float voxel_size) {
    // TODO: Implement CUDA downsampling
    return cudaErrorNotSupported;
}

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
                                  float leaf_size_z) {
    // TODO: Implement CUDA voxel grid filtering
    return cudaErrorNotSupported;
}

} // extern "C"

#endif // USE_CUDA
