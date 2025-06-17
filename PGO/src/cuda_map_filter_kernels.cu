// CUDA-only headers to avoid PCL compatibility issues
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <iostream>
#include <fstream>
#include <algorithm>

#ifdef USE_CUDA

namespace cuda_map_filter {

/**
 * CUDA kernels for map filtering operations
 */

__global__ void roiFilterKernel(const float* input_x,
                               const float* input_y,
                               const float* input_z,
                               const float* input_intensity,
                               float* output_x,
                               float* output_y,
                               float* output_z,
                               float* output_intensity,
                               int* output_indices,
                               int num_input_points,
                               float min_x, float min_y, float min_z,
                               float max_x, float max_y, float max_z,
                               int* num_output_points) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_input_points) return;
    
    float x = input_x[idx];
    float y = input_y[idx];
    float z = input_z[idx];
    
    // Check if point is within ROI bounds
    if (x >= min_x && x <= max_x &&
        y >= min_y && y <= max_y &&
        z >= min_z && z <= max_z) {
        
        // Use atomic operation to get unique output index
        int output_idx = atomicAdd(num_output_points, 1);
        
        output_x[output_idx] = x;
        output_y[output_idx] = y;
        output_z[output_idx] = z;
        output_intensity[output_idx] = input_intensity[idx];
        output_indices[output_idx] = idx;
    }
}

__global__ void voxelGridFilterKernel(const float* input_x,
                                     const float* input_y,
                                     const float* input_z,
                                     const float* input_intensity,
                                     float* output_x,
                                     float* output_y,
                                     float* output_z,
                                     float* output_intensity,
                                     int* voxel_indices,
                                     int num_input_points,
                                     float voxel_size,
                                     float min_x, float min_y, float min_z,
                                     int grid_x, int grid_y, int grid_z,
                                     int* num_output_points) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_input_points) return;
    
    float x = input_x[idx];
    float y = input_y[idx];
    float z = input_z[idx];
    
    // Calculate voxel coordinates
    int vx = (int)floorf((x - min_x) / voxel_size);
    int vy = (int)floorf((y - min_y) / voxel_size);
    int vz = (int)floorf((z - min_z) / voxel_size);
    
    // Check bounds
    if (vx < 0 || vx >= grid_x || vy < 0 || vy >= grid_y || vz < 0 || vz >= grid_z) {
        return;
    }
    
    // Calculate linear voxel index
    int voxel_idx = vx + vy * grid_x + vz * grid_x * grid_y;
    
    // Use atomic compare-and-swap to ensure only one point per voxel
    int old_val = atomicCAS(&voxel_indices[voxel_idx], -1, idx);
    if (old_val == -1) {
        // This is the first point in this voxel
        int output_idx = atomicAdd(num_output_points, 1);
        
        output_x[output_idx] = x;
        output_y[output_idx] = y;
        output_z[output_idx] = z;
        output_intensity[output_idx] = input_intensity[idx];
    }
}

__global__ void statisticalOutlierRemovalKernel(const float* input_x,
                                               const float* input_y,
                                               const float* input_z,
                                               const float* input_intensity,
                                               float* output_x,
                                               float* output_y,
                                               float* output_z,
                                               float* output_intensity,
                                               const float* distances,
                                               int num_input_points,
                                               float mean_distance,
                                               float std_dev,
                                               float std_ratio,
                                               int* num_output_points) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_input_points) return;
    
    float threshold = mean_distance + std_ratio * std_dev;
    
    if (distances[idx] <= threshold) {
        int output_idx = atomicAdd(num_output_points, 1);
        
        output_x[output_idx] = input_x[idx];
        output_y[output_idx] = input_y[idx];
        output_z[output_idx] = input_z[idx];
        output_intensity[output_idx] = input_intensity[idx];
    }
}

__global__ void radiusOutlierRemovalKernel(const float* input_x,
                                          const float* input_y,
                                          const float* input_z,
                                          const float* input_intensity,
                                          float* output_x,
                                          float* output_y,
                                          float* output_z,
                                          float* output_intensity,
                                          int num_input_points,
                                          float radius,
                                          int min_neighbors,
                                          int* num_output_points) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_input_points) return;
    
    float px = input_x[idx];
    float py = input_y[idx];
    float pz = input_z[idx];
    float radius_sq = radius * radius;
    
    int neighbor_count = 0;
    
    // Count neighbors within radius
    for (int i = 0; i < num_input_points; i++) {
        if (i == idx) continue;
        
        float dx = input_x[i] - px;
        float dy = input_y[i] - py;
        float dz = input_z[i] - pz;
        float dist_sq = dx*dx + dy*dy + dz*dz;
        
        if (dist_sq <= radius_sq) {
            neighbor_count++;
        }
        
        // Early termination if we already have enough neighbors
        if (neighbor_count >= min_neighbors) {
            break;
        }
    }
    
    if (neighbor_count >= min_neighbors) {
        int output_idx = atomicAdd(num_output_points, 1);
        
        output_x[output_idx] = px;
        output_y[output_idx] = py;
        output_z[output_idx] = pz;
        output_intensity[output_idx] = input_intensity[idx];
    }
}

// C-style interface for CUDA kernels
extern "C" {
    cudaError_t cuda_roi_filter(const float* d_input_x,
                               const float* d_input_y,
                               const float* d_input_z,
                               const float* d_input_intensity,
                               float* d_output_x,
                               float* d_output_y,
                               float* d_output_z,
                               float* d_output_intensity,
                               int* d_output_indices,
                               int num_input_points,
                               float min_x, float min_y, float min_z,
                               float max_x, float max_y, float max_z,
                               int* d_num_output_points,
                               int block_size) {
        
        dim3 blockSize(block_size);
        dim3 gridSize((num_input_points + blockSize.x - 1) / blockSize.x);
        
        roiFilterKernel<<<gridSize, blockSize>>>(
            d_input_x, d_input_y, d_input_z, d_input_intensity,
            d_output_x, d_output_y, d_output_z, d_output_intensity,
            d_output_indices, num_input_points,
            min_x, min_y, min_z, max_x, max_y, max_z,
            d_num_output_points
        );
        
        return cudaGetLastError();
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
                                      float voxel_size,
                                      float min_x, float min_y, float min_z,
                                      int grid_x, int grid_y, int grid_z,
                                      int* d_num_output_points,
                                      int block_size) {
        
        dim3 blockSize(block_size);
        dim3 gridSize((num_input_points + blockSize.x - 1) / blockSize.x);
        
        voxelGridFilterKernel<<<gridSize, blockSize>>>(
            d_input_x, d_input_y, d_input_z, d_input_intensity,
            d_output_x, d_output_y, d_output_z, d_output_intensity,
            d_voxel_indices, num_input_points, voxel_size,
            min_x, min_y, min_z, grid_x, grid_y, grid_z,
            d_num_output_points
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t cuda_statistical_outlier_removal(const float* d_input_x,
                                                const float* d_input_y,
                                                const float* d_input_z,
                                                const float* d_input_intensity,
                                                float* d_output_x,
                                                float* d_output_y,
                                                float* d_output_z,
                                                float* d_output_intensity,
                                                const float* d_distances,
                                                int num_input_points,
                                                float mean_distance,
                                                float std_dev,
                                                float std_ratio,
                                                int* d_num_output_points,
                                                int block_size) {
        
        dim3 blockSize(block_size);
        dim3 gridSize((num_input_points + blockSize.x - 1) / blockSize.x);
        
        statisticalOutlierRemovalKernel<<<gridSize, blockSize>>>(
            d_input_x, d_input_y, d_input_z, d_input_intensity,
            d_output_x, d_output_y, d_output_z, d_output_intensity,
            d_distances, num_input_points, mean_distance, std_dev, std_ratio,
            d_num_output_points
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t cuda_radius_outlier_removal(const float* d_input_x,
                                           const float* d_input_y,
                                           const float* d_input_z,
                                           const float* d_input_intensity,
                                           float* d_output_x,
                                           float* d_output_y,
                                           float* d_output_z,
                                           float* d_output_intensity,
                                           int num_input_points,
                                           float radius,
                                           int min_neighbors,
                                           int* d_num_output_points,
                                           int block_size) {
        
        dim3 blockSize(block_size);
        dim3 gridSize((num_input_points + blockSize.x - 1) / blockSize.x);
        
        radiusOutlierRemovalKernel<<<gridSize, blockSize>>>(
            d_input_x, d_input_y, d_input_z, d_input_intensity,
            d_output_x, d_output_y, d_output_z, d_output_intensity,
            num_input_points, radius, min_neighbors,
            d_num_output_points
        );
        
        return cudaGetLastError();
    }
}

} // namespace cuda_map_filter

#endif // USE_CUDA
