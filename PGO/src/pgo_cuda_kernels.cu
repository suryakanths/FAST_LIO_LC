// Advanced CUDA kernels for PGO - compiled with NVCC
#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <vector>
#include <cfloat>

namespace pgo_cuda {
namespace cuda_kernels {

// Advanced correspondence finding with spatial hashing
__global__ void find_correspondences_spatial_hash_kernel(const float* __restrict__ source_x,
                                                        const float* __restrict__ source_y,
                                                        const float* __restrict__ source_z,
                                                        const float* __restrict__ target_x,
                                                        const float* __restrict__ target_y,
                                                        const float* __restrict__ target_z,
                                                        int* __restrict__ correspondence_indices,
                                                        float* __restrict__ correspondence_distances,
                                                        const int* __restrict__ hash_table,
                                                        const int* __restrict__ hash_counts,
                                                        int num_source_points,
                                                        int num_target_points,
                                                        float max_distance_sq,
                                                        float grid_size,
                                                        int hash_table_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_source_points) {
        return;
    }
    
    float src_x = source_x[idx];
    float src_y = source_y[idx];
    float src_z = source_z[idx];
    
    // Compute hash for this point
    int hash_x = (int)(src_x / grid_size);
    int hash_y = (int)(src_y / grid_size);
    int hash_z = (int)(src_z / grid_size);
    
    float min_dist_sq = FLT_MAX;
    int best_match = -1;
    
    // Search in neighboring grid cells
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                int cell_x = hash_x + dx;
                int cell_y = hash_y + dy;
                int cell_z = hash_z + dz;
                
                // Simple hash function
                int hash = ((cell_x * 73856093) ^ (cell_y * 19349663) ^ (cell_z * 83492791)) % hash_table_size;
                if (hash < 0) hash += hash_table_size;
                
                int start_idx = hash_table[hash];
                int count = hash_counts[hash];
                
                // Search points in this hash bucket
                for (int i = 0; i < count && start_idx + i < num_target_points; ++i) {
                    int target_idx = start_idx + i;
                    
                    float dx_dist = src_x - target_x[target_idx];
                    float dy_dist = src_y - target_y[target_idx];
                    float dz_dist = src_z - target_z[target_idx];
                    float dist_sq = dx_dist*dx_dist + dy_dist*dy_dist + dz_dist*dz_dist;
                    
                    if (dist_sq < min_dist_sq) {
                        min_dist_sq = dist_sq;
                        best_match = target_idx;
                    }
                }
            }
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

// ICP-style point-to-plane distance computation
__global__ void compute_point_to_plane_distances_kernel(const float* __restrict__ source_x,
                                                       const float* __restrict__ source_y,
                                                       const float* __restrict__ source_z,
                                                       const float* __restrict__ target_x,
                                                       const float* __restrict__ target_y,
                                                       const float* __restrict__ target_z,
                                                       const float* __restrict__ target_nx,
                                                       const float* __restrict__ target_ny,
                                                       const float* __restrict__ target_nz,
                                                       float* __restrict__ distances,
                                                       int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) {
        return;
    }
    
    // Vector from target to source point
    float dx = source_x[idx] - target_x[idx];
    float dy = source_y[idx] - target_y[idx];
    float dz = source_z[idx] - target_z[idx];
    
    // Dot product with target normal gives point-to-plane distance
    float distance = dx * target_nx[idx] + dy * target_ny[idx] + dz * target_nz[idx];
    distances[idx] = fabsf(distance);
}

// Robust M-estimator weights for outlier rejection
__global__ void compute_robust_weights_kernel(const float* __restrict__ residuals,
                                             float* __restrict__ weights,
                                             int num_points,
                                             float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points) {
        return;
    }
    
    float residual = residuals[idx];
    float abs_residual = fabsf(residual);
    
    // Huber weight function
    if (abs_residual <= threshold) {
        weights[idx] = 1.0f;
    } else {
        weights[idx] = threshold / abs_residual;
    }
}

// Parallel reduction for computing sum of squared residuals
__global__ void reduce_squared_residuals_kernel(const float* __restrict__ residuals,
                                               const float* __restrict__ weights,
                                               float* __restrict__ partial_sums,
                                               int num_points) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (idx < num_points) {
        float residual = residuals[idx];
        float weight = weights ? weights[idx] : 1.0f;
        sdata[tid] = weight * residual * residual;
    } else {
        sdata[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Voxel grid downsampling kernel
__global__ void voxel_downsample_kernel(const float* __restrict__ input_x,
                                       const float* __restrict__ input_y,
                                       const float* __restrict__ input_z,
                                       const float* __restrict__ input_intensity,
                                       float* __restrict__ output_x,
                                       float* __restrict__ output_y,
                                       float* __restrict__ output_z,
                                       float* __restrict__ output_intensity,
                                       const int* __restrict__ voxel_indices,
                                       const int* __restrict__ point_counts,
                                       int num_output_voxels,
                                       float inv_leaf_size_x,
                                       float inv_leaf_size_y,
                                       float inv_leaf_size_z) {
    int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (voxel_idx >= num_output_voxels) {
        return;
    }
    
    int start_idx = voxel_indices[voxel_idx];
    int count = point_counts[voxel_idx];
    
    if (count == 0) {
        return;
    }
    
    // Compute centroid of points in this voxel
    float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f, sum_intensity = 0.0f;
    
    for (int i = 0; i < count; ++i) {
        int point_idx = start_idx + i;
        sum_x += input_x[point_idx];
        sum_y += input_y[point_idx];
        sum_z += input_z[point_idx];
        sum_intensity += input_intensity[point_idx];
    }
    
    float inv_count = 1.0f / count;
    output_x[voxel_idx] = sum_x * inv_count;
    output_y[voxel_idx] = sum_y * inv_count;
    output_z[voxel_idx] = sum_z * inv_count;
    output_intensity[voxel_idx] = sum_intensity * inv_count;
}

// Kernel for computing pose graph optimization residuals
__global__ void compute_pgo_residuals_kernel(const float* __restrict__ poses,
                                            const float* __restrict__ measurements,
                                            const float* __restrict__ information_matrices,
                                            float* __restrict__ residuals,
                                            const int* __restrict__ edge_indices,
                                            int num_edges,
                                            int pose_dim) {
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (edge_idx >= num_edges) {
        return;
    }
    
    int pose_i_idx = edge_indices[edge_idx * 2];
    int pose_j_idx = edge_indices[edge_idx * 2 + 1];
    
    // Load poses (assuming SE(3) representation: x, y, z, qx, qy, qz, qw)
    const float* pose_i = &poses[pose_i_idx * pose_dim];
    const float* pose_j = &poses[pose_j_idx * pose_dim];
    const float* measurement = &measurements[edge_idx * pose_dim];
    
    // Compute relative pose error (simplified for demonstration)
    // In practice, this would involve proper SE(3) group operations
    for (int i = 0; i < pose_dim; ++i) {
        float predicted = pose_j[i] - pose_i[i];  // Simplified relative pose
        float error = predicted - measurement[i];
        residuals[edge_idx * pose_dim + i] = error;
    }
}

} // namespace cuda_kernels
} // namespace pgo_cuda

// Additional C interface functions for advanced kernels
extern "C" {

cudaError_t cuda_find_correspondences_spatial_hash(const float* d_source_x,
                                                   const float* d_source_y,
                                                   const float* d_source_z,
                                                   const float* d_target_x,
                                                   const float* d_target_y,
                                                   const float* d_target_z,
                                                   int* d_correspondence_indices,
                                                   float* d_correspondence_distances,
                                                   const int* d_hash_table,
                                                   const int* d_hash_counts,
                                                   int num_source_points,
                                                   int num_target_points,
                                                   float max_distance,
                                                   float grid_size,
                                                   int hash_table_size) {
    
    const int threads_per_block = 256;
    const int blocks = (num_source_points + threads_per_block - 1) / threads_per_block;
    
    float max_distance_sq = max_distance * max_distance;
    
    pgo_cuda::cuda_kernels::find_correspondences_spatial_hash_kernel<<<blocks, threads_per_block>>>(
        d_source_x, d_source_y, d_source_z,
        d_target_x, d_target_y, d_target_z,
        d_correspondence_indices, d_correspondence_distances,
        d_hash_table, d_hash_counts,
        num_source_points, num_target_points, max_distance_sq,
        grid_size, hash_table_size
    );
    
    return cudaGetLastError();
}

cudaError_t cuda_compute_point_to_plane_distances(const float* d_source_x,
                                                  const float* d_source_y,
                                                  const float* d_source_z,
                                                  const float* d_target_x,
                                                  const float* d_target_y,
                                                  const float* d_target_z,
                                                  const float* d_target_nx,
                                                  const float* d_target_ny,
                                                  const float* d_target_nz,
                                                  float* d_distances,
                                                  int num_points) {
    
    const int threads_per_block = 256;
    const int blocks = (num_points + threads_per_block - 1) / threads_per_block;
    
    pgo_cuda::cuda_kernels::compute_point_to_plane_distances_kernel<<<blocks, threads_per_block>>>(
        d_source_x, d_source_y, d_source_z,
        d_target_x, d_target_y, d_target_z,
        d_target_nx, d_target_ny, d_target_nz,
        d_distances, num_points
    );
    
    return cudaGetLastError();
}

cudaError_t cuda_compute_robust_weights(const float* d_residuals,
                                       float* d_weights,
                                       int num_points,
                                       float threshold) {
    
    const int threads_per_block = 256;
    const int blocks = (num_points + threads_per_block - 1) / threads_per_block;
    
    pgo_cuda::cuda_kernels::compute_robust_weights_kernel<<<blocks, threads_per_block>>>(
        d_residuals, d_weights, num_points, threshold
    );
    
    return cudaGetLastError();
}

cudaError_t cuda_reduce_squared_residuals(const float* d_residuals,
                                         const float* d_weights,
                                         float* d_result,
                                         int num_points) {
    
    const int threads_per_block = 256;
    const int blocks = (num_points + threads_per_block - 1) / threads_per_block;
    
    // Allocate temporary storage for partial sums
    float* d_partial_sums;
    cudaMalloc(&d_partial_sums, blocks * sizeof(float));
    
    // First reduction: blocks -> partial sums
    size_t shared_mem_size = threads_per_block * sizeof(float);
    pgo_cuda::cuda_kernels::reduce_squared_residuals_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        d_residuals, d_weights, d_partial_sums, num_points
    );
    
    // Final reduction on CPU (simpler and more compatible)
    std::vector<float> host_partial_sums(blocks);
    cudaMemcpy(host_partial_sums.data(), d_partial_sums, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_sum = 0.0f;
    for (int i = 0; i < blocks; ++i) {
        total_sum += host_partial_sums[i];
    }
    
    cudaMemcpy(d_result, &total_sum, sizeof(float), cudaMemcpyHostToDevice);
    cudaFree(d_partial_sums);
    
    return cudaGetLastError();
}

} // extern "C"

#endif // USE_CUDA
