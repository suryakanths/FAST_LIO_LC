#include "cuda_map_filter.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>

namespace cuda_map_filter {

// Forward declarations for CUDA kernels
extern "C" {
    cudaError_t cuda_roi_filter(const float* d_input_x, const float* d_input_y, const float* d_input_z,
                               const float* d_input_intensity, float* d_output_x, float* d_output_y,
                               float* d_output_z, float* d_output_intensity, int* d_output_indices,
                               int num_input_points, float min_x, float min_y, float min_z,
                               float max_x, float max_y, float max_z, int* d_num_output_points, int block_size);
    
    cudaError_t cuda_voxel_grid_filter(const float* d_input_x, const float* d_input_y, const float* d_input_z,
                                      const float* d_input_intensity, float* d_output_x, float* d_output_y,
                                      float* d_output_z, float* d_output_intensity, int* d_voxel_indices,
                                      int num_input_points, float voxel_size, float min_x, float min_y, float min_z,
                                      int grid_x, int grid_y, int grid_z, int* d_num_output_points, int block_size);
    
    cudaError_t cuda_statistical_outlier_removal(const float* d_input_x, const float* d_input_y, const float* d_input_z,
                                                const float* d_input_intensity, float* d_output_x, float* d_output_y,
                                                float* d_output_z, float* d_output_intensity, const float* d_distances,
                                                int num_input_points, float mean_distance, float std_dev, float std_ratio,
                                                int* d_num_output_points, int block_size);
    
    cudaError_t cuda_radius_outlier_removal(const float* d_input_x, const float* d_input_y, const float* d_input_z,
                                           const float* d_input_intensity, float* d_output_x, float* d_output_y,
                                           float* d_output_z, float* d_output_intensity, int num_input_points,
                                           float radius, int min_neighbors, int* d_num_output_points, int block_size);
}

/**
 * Internal CUDA data structure
 */
struct CudaMapFilter::CudaFilterData {
    // Device memory pointers
    float* d_input_x;
    float* d_input_y;
    float* d_input_z;
    float* d_input_intensity;
    
    float* d_output_x;
    float* d_output_y;
    float* d_output_z;
    float* d_output_intensity;
    
    int* d_output_indices;
    int* d_voxel_indices;
    int* d_num_output_points;
    float* d_distances;
    
    // CUDA streams for concurrent processing
    cudaStream_t stream1;
    cudaStream_t stream2;
    
    // Memory management
    size_t allocated_points;
    bool streams_created;
    
    CudaFilterData() : d_input_x(nullptr), d_input_y(nullptr), d_input_z(nullptr), d_input_intensity(nullptr),
                       d_output_x(nullptr), d_output_y(nullptr), d_output_z(nullptr), d_output_intensity(nullptr),
                       d_output_indices(nullptr), d_voxel_indices(nullptr), d_num_output_points(nullptr),
                       d_distances(nullptr), allocated_points(0), streams_created(false) {}
};

/**
 * Template implementation for CudaManagedMemory
 */
template<typename T>
CudaManagedMemory<T>::CudaManagedMemory(size_t count, bool use_unified) 
    : ptr_(nullptr), count_(count), byte_size_(count * sizeof(T)), unified_(use_unified) {
    Allocate();
}

template<typename T>
CudaManagedMemory<T>::~CudaManagedMemory() {
    Deallocate();
}

template<typename T>
CudaManagedMemory<T>::CudaManagedMemory(CudaManagedMemory&& other) noexcept
    : ptr_(other.ptr_), count_(other.count_), byte_size_(other.byte_size_), unified_(other.unified_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
    other.byte_size_ = 0;
}

template<typename T>
CudaManagedMemory<T>& CudaManagedMemory<T>::operator=(CudaManagedMemory&& other) noexcept {
    if (this != &other) {
        Deallocate();
        ptr_ = other.ptr_;
        count_ = other.count_;
        byte_size_ = other.byte_size_;
        unified_ = other.unified_;
        other.ptr_ = nullptr;
        other.count_ = 0;
        other.byte_size_ = 0;
    }
    return *this;
}

template<typename T>
void CudaManagedMemory<T>::Allocate() {
    if (count_ == 0) return;
    
    cudaError_t error;
    if (unified_) {
        error = cudaMallocManaged(&ptr_, byte_size_);
    } else {
        error = cudaMalloc(&ptr_, byte_size_);
    }
    
    if (error != cudaSuccess) {
        ptr_ = nullptr;
        throw std::runtime_error("CUDA memory allocation failed: " + std::string(cudaGetErrorString(error)));
    }
}

template<typename T>
void CudaManagedMemory<T>::Deallocate() {
    if (ptr_) {
        cudaFree(ptr_);
        ptr_ = nullptr;
    }
}

template<typename T>
bool CudaManagedMemory<T>::CopyToDevice(const T* host_data, size_t count) {
    if (!ptr_ || count > count_) return false;
    
    if (unified_) {
        std::memcpy(ptr_, host_data, count * sizeof(T));
        return true;
    } else {
        cudaError_t error = cudaMemcpy(ptr_, host_data, count * sizeof(T), cudaMemcpyHostToDevice);
        return error == cudaSuccess;
    }
}

template<typename T>
bool CudaManagedMemory<T>::CopyFromDevice(T* host_data, size_t count) {
    if (!ptr_ || count > count_) return false;
    
    if (unified_) {
        cudaDeviceSynchronize(); // Ensure GPU operations are complete
        std::memcpy(host_data, ptr_, count * sizeof(T));
        return true;
    } else {
        cudaError_t error = cudaMemcpy(host_data, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost);
        return error == cudaSuccess;
    }
}

// Explicit template instantiation
template class CudaManagedMemory<float>;
template class CudaManagedMemory<int>;

/**
 * CudaMapFilter implementation
 */
CudaMapFilter::CudaMapFilter() 
    : cuda_data_(std::make_unique<CudaFilterData>()),
      allocated_points_(0),
      cuda_initialized_(false),
      thermal_throttle_detected_(false) {
    
    // Detect platform and configure optimizations
    jetson_info_ = DetectJetsonPlatform();
    kernel_config_ = GetOptimalKernelConfig(jetson_info_);
    
    if (IsCudaAvailable()) {
        InitializeCuda();
    }
}

CudaMapFilter::~CudaMapFilter() {
    CleanupCuda();
}

bool CudaMapFilter::IsCudaAvailable() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return (error == cudaSuccess && device_count > 0);
}

JetsonInfo CudaMapFilter::DetectJetsonPlatform() {
    JetsonInfo info;
    
    // Check if running on Jetson by reading device tree model
    std::ifstream model_file("/proc/device-tree/model");
    if (model_file.is_open()) {
        std::getline(model_file, info.model);
        model_file.close();
        
        if (info.model.find("NVIDIA Jetson") != std::string::npos) {
            info.is_jetson = true;
            
            // Detect specific Jetson model and capabilities
            if (info.model.find("Orin") != std::string::npos) {
                info.compute_capability_major = 8;
                info.compute_capability_minor = 7;
                info.total_memory_mb = 8192; // Typical for Orin NX
                info.supports_unified_memory = true;
            } else if (info.model.find("Xavier") != std::string::npos) {
                info.compute_capability_major = 7;
                info.compute_capability_minor = 2;
                info.total_memory_mb = 8192; // Typical for Xavier NX
                info.supports_unified_memory = true;
            } else if (info.model.find("TX2") != std::string::npos) {
                info.compute_capability_major = 6;
                info.compute_capability_minor = 2;
                info.total_memory_mb = 8192;
                info.supports_unified_memory = true;
            } else if (info.model.find("Nano") != std::string::npos) {
                info.compute_capability_major = 5;
                info.compute_capability_minor = 3;
                info.total_memory_mb = 4096;
                info.supports_unified_memory = true;
            }
        }
    }
    
    return info;
}

KernelConfig CudaMapFilter::GetOptimalKernelConfig(const JetsonInfo& info) {
    KernelConfig config;
    
    if (info.is_jetson) {
        config.use_unified_memory = info.supports_unified_memory;
        
        if (info.model.find("Orin") != std::string::npos) {
            config.block_size = models::ORIN_OPTIMAL_BLOCK_SIZE;
            config.memory_pool_fraction = models::ORIN_MEMORY_FRACTION;
            config.enable_concurrent_streams = true;
        } else if (info.model.find("Xavier") != std::string::npos) {
            config.block_size = models::XAVIER_OPTIMAL_BLOCK_SIZE;
            config.memory_pool_fraction = models::XAVIER_MEMORY_FRACTION;
            config.enable_concurrent_streams = true;
        } else if (info.model.find("TX2") != std::string::npos) {
            config.block_size = models::TX2_OPTIMAL_BLOCK_SIZE;
            config.memory_pool_fraction = models::TX2_MEMORY_FRACTION;
            config.enable_concurrent_streams = false;
        } else if (info.model.find("Nano") != std::string::npos) {
            config.block_size = models::NANO_OPTIMAL_BLOCK_SIZE;
            config.memory_pool_fraction = models::NANO_MEMORY_FRACTION;
            config.enable_concurrent_streams = false;
        }
    } else {
        // Desktop GPU configuration
        config.block_size = models::DESKTOP_OPTIMAL_BLOCK_SIZE;
        config.memory_pool_fraction = models::DESKTOP_MEMORY_FRACTION;
        config.use_unified_memory = false;
        config.enable_concurrent_streams = true;
    }
    
    return config;
}

bool CudaMapFilter::InitializeCuda() {
    if (cuda_initialized_) return true;
    
    // Set device and get properties
    cudaError_t error = cudaSetDevice(0);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    cudaDeviceProp deviceProp;
    error = cudaGetDeviceProperties(&deviceProp, 0);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get device properties: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Create CUDA streams if concurrent execution is enabled
    if (kernel_config_.enable_concurrent_streams) {
        error = cudaStreamCreate(&cuda_data_->stream1);
        if (error == cudaSuccess) {
            error = cudaStreamCreate(&cuda_data_->stream2);
        }
        
        if (error == cudaSuccess) {
            cuda_data_->streams_created = true;
        } else {
            std::cerr << "Warning: Failed to create CUDA streams, falling back to default stream" << std::endl;
            cuda_data_->streams_created = false;
        }
    }
    
    cuda_initialized_ = true;
    std::cout << "[CUDA Map Filter] Initialized successfully on " 
              << (jetson_info_.is_jetson ? jetson_info_.model : "Desktop GPU") << std::endl;
    std::cout << "[CUDA Map Filter] Block size: " << kernel_config_.block_size 
              << ", Unified memory: " << (kernel_config_.use_unified_memory ? "enabled" : "disabled") << std::endl;
    
    return true;
}

void CudaMapFilter::CleanupCuda() {
    DeallocateGpuMemory();
    
    if (cuda_data_->streams_created) {
        cudaStreamDestroy(cuda_data_->stream1);
        cudaStreamDestroy(cuda_data_->stream2);
        cuda_data_->streams_created = false;
    }
    
    cuda_initialized_ = false;
}

bool CudaMapFilter::AllocateGpuMemory(size_t num_points) {
    if (num_points <= allocated_points_) {
        return true; // Already have enough memory
    }
    
    // Deallocate existing memory
    DeallocateGpuMemory();
    
    try {
        // Allocate input arrays
        cuda_data_->d_input_x = static_cast<float*>(
            kernel_config_.use_unified_memory ? 
            malloc(num_points * sizeof(float)) : 
            nullptr);
        
        if (!kernel_config_.use_unified_memory) {
            cudaError_t error = cudaMalloc(&cuda_data_->d_input_x, num_points * sizeof(float));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate d_input_x");
            
            error = cudaMalloc(&cuda_data_->d_input_y, num_points * sizeof(float));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate d_input_y");
            
            error = cudaMalloc(&cuda_data_->d_input_z, num_points * sizeof(float));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate d_input_z");
            
            error = cudaMalloc(&cuda_data_->d_input_intensity, num_points * sizeof(float));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate d_input_intensity");
            
            // Allocate output arrays
            error = cudaMalloc(&cuda_data_->d_output_x, num_points * sizeof(float));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate d_output_x");
            
            error = cudaMalloc(&cuda_data_->d_output_y, num_points * sizeof(float));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate d_output_y");
            
            error = cudaMalloc(&cuda_data_->d_output_z, num_points * sizeof(float));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate d_output_z");
            
            error = cudaMalloc(&cuda_data_->d_output_intensity, num_points * sizeof(float));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate d_output_intensity");
            
            // Allocate auxiliary arrays
            error = cudaMalloc(&cuda_data_->d_output_indices, num_points * sizeof(int));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate d_output_indices");
            
            error = cudaMalloc(&cuda_data_->d_voxel_indices, num_points * sizeof(int));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate d_voxel_indices");
            
            error = cudaMalloc(&cuda_data_->d_num_output_points, sizeof(int));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate d_num_output_points");
            
            error = cudaMalloc(&cuda_data_->d_distances, num_points * sizeof(float));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate d_distances");
        } else {
            // Use unified memory for Jetson platforms
            cudaError_t error = cudaMallocManaged(&cuda_data_->d_input_x, num_points * sizeof(float));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate unified d_input_x");
            
            error = cudaMallocManaged(&cuda_data_->d_input_y, num_points * sizeof(float));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate unified d_input_y");
            
            error = cudaMallocManaged(&cuda_data_->d_input_z, num_points * sizeof(float));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate unified d_input_z");
            
            error = cudaMallocManaged(&cuda_data_->d_input_intensity, num_points * sizeof(float));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate unified d_input_intensity");
            
            error = cudaMallocManaged(&cuda_data_->d_output_x, num_points * sizeof(float));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate unified d_output_x");
            
            error = cudaMallocManaged(&cuda_data_->d_output_y, num_points * sizeof(float));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate unified d_output_y");
            
            error = cudaMallocManaged(&cuda_data_->d_output_z, num_points * sizeof(float));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate unified d_output_z");
            
            error = cudaMallocManaged(&cuda_data_->d_output_intensity, num_points * sizeof(float));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate unified d_output_intensity");
            
            error = cudaMallocManaged(&cuda_data_->d_output_indices, num_points * sizeof(int));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate unified d_output_indices");
            
            error = cudaMallocManaged(&cuda_data_->d_voxel_indices, num_points * sizeof(int));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate unified d_voxel_indices");
            
            error = cudaMallocManaged(&cuda_data_->d_num_output_points, sizeof(int));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate unified d_num_output_points");
            
            error = cudaMallocManaged(&cuda_data_->d_distances, num_points * sizeof(float));
            if (error != cudaSuccess) throw std::runtime_error("Failed to allocate unified d_distances");
        }
        
        allocated_points_ = num_points;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "GPU memory allocation failed: " << e.what() << std::endl;
        DeallocateGpuMemory();
        return false;
    }
}

void CudaMapFilter::DeallocateGpuMemory() {
    if (cuda_data_->d_input_x) { cudaFree(cuda_data_->d_input_x); cuda_data_->d_input_x = nullptr; }
    if (cuda_data_->d_input_y) { cudaFree(cuda_data_->d_input_y); cuda_data_->d_input_y = nullptr; }
    if (cuda_data_->d_input_z) { cudaFree(cuda_data_->d_input_z); cuda_data_->d_input_z = nullptr; }
    if (cuda_data_->d_input_intensity) { cudaFree(cuda_data_->d_input_intensity); cuda_data_->d_input_intensity = nullptr; }
    if (cuda_data_->d_output_x) { cudaFree(cuda_data_->d_output_x); cuda_data_->d_output_x = nullptr; }
    if (cuda_data_->d_output_y) { cudaFree(cuda_data_->d_output_y); cuda_data_->d_output_y = nullptr; }
    if (cuda_data_->d_output_z) { cudaFree(cuda_data_->d_output_z); cuda_data_->d_output_z = nullptr; }
    if (cuda_data_->d_output_intensity) { cudaFree(cuda_data_->d_output_intensity); cuda_data_->d_output_intensity = nullptr; }
    if (cuda_data_->d_output_indices) { cudaFree(cuda_data_->d_output_indices); cuda_data_->d_output_indices = nullptr; }
    if (cuda_data_->d_voxel_indices) { cudaFree(cuda_data_->d_voxel_indices); cuda_data_->d_voxel_indices = nullptr; }
    if (cuda_data_->d_num_output_points) { cudaFree(cuda_data_->d_num_output_points); cuda_data_->d_num_output_points = nullptr; }
    if (cuda_data_->d_distances) { cudaFree(cuda_data_->d_distances); cuda_data_->d_distances = nullptr; }
    
    allocated_points_ = 0;
}

bool CudaMapFilter::EnsureGpuMemory(size_t num_points) {
    if (num_points > allocated_points_) {
        return AllocateGpuMemory(num_points);
    }
    return true;
}

float CudaMapFilter::ReadThermalZoneTemp(const std::string& zone_name) {
    std::string thermal_path = "/sys/class/thermal/thermal_zone0/temp";
    std::ifstream temp_file(thermal_path);
    
    if (temp_file.is_open()) {
        std::string temp_str;
        std::getline(temp_file, temp_str);
        temp_file.close();
        
        try {
            float temp_millidegrees = std::stof(temp_str);
            return temp_millidegrees / 1000.0f; // Convert to degrees Celsius
        } catch (const std::exception&) {
            return 0.0f;
        }
    }
    
    return 0.0f;
}

bool CudaMapFilter::CheckThermalThrottling() {
    if (!jetson_info_.is_jetson) return false;
    
    float gpu_temp = ReadThermalZoneTemp("GPU-therm");
    float cpu_temp = ReadThermalZoneTemp("CPU-therm");
    
    const float THERMAL_THROTTLE_THRESHOLD = 75.0f;
    
    bool throttling = (gpu_temp > THERMAL_THROTTLE_THRESHOLD || cpu_temp > THERMAL_THROTTLE_THRESHOLD);
    
    if (throttling && !thermal_throttle_detected_) {
        std::cout << "[CUDA Map Filter] Thermal throttling detected: GPU=" << gpu_temp 
                  << "°C, CPU=" << cpu_temp << "°C" << std::endl;
        thermal_throttle_detected_ = true;
    } else if (!throttling && thermal_throttle_detected_) {
        std::cout << "[CUDA Map Filter] Thermal throttling resolved" << std::endl;
        thermal_throttle_detected_ = false;
    }
    
    return throttling;
}

size_t CudaMapFilter::GetOptimalBatchSize(size_t input_points, size_t point_byte_size) {
    size_t available_memory = jetson_info_.total_memory_mb * 1024 * 1024;
    size_t usable_memory = available_memory * kernel_config_.memory_pool_fraction;
    
    // Account for input + output + auxiliary data
    size_t memory_per_point = point_byte_size * 3; // Rough estimate
    size_t max_points_by_memory = usable_memory / memory_per_point;
    
    // Platform-specific batch size limits
    size_t platform_limit;
    if (jetson_info_.is_jetson) {
        if (jetson_info_.model.find("Orin") != std::string::npos) {
            platform_limit = 500000;
        } else if (jetson_info_.model.find("Xavier") != std::string::npos) {
            platform_limit = 200000;
        } else if (jetson_info_.model.find("TX2") != std::string::npos) {
            platform_limit = 100000;
        } else { // Nano
            platform_limit = 50000;
        }
    } else {
        platform_limit = 1000000; // Desktop
    }
    
    return std::min({input_points, max_points_by_memory, platform_limit});
}

// Continue with filter implementations...
bool CudaMapFilter::ROIFilter(const PointCloudType::Ptr& input_cloud,
                             PointCloudType::Ptr& output_cloud,
                             const geometry_msgs::Point& min_pt,
                             const geometry_msgs::Point& max_pt) {
    
    if (!cuda_initialized_) {
        return cpu_fallback::ROIFilter(input_cloud, output_cloud, min_pt, max_pt);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    size_t num_points = input_cloud->size();
    if (num_points == 0) {
        output_cloud->clear();
        return true;
    }
    
    // Check for thermal throttling on Jetson
    if (jetson_info_.is_jetson && CheckThermalThrottling()) {
        return cpu_fallback::ROIFilter(input_cloud, output_cloud, min_pt, max_pt);
    }
    
    if (!EnsureGpuMemory(num_points)) {
        std::cerr << "Failed to allocate GPU memory for ROI filter" << std::endl;
        return cpu_fallback::ROIFilter(input_cloud, output_cloud, min_pt, max_pt);
    }
    
    try {
        // Copy point cloud data to GPU
        std::vector<float> host_x(num_points), host_y(num_points), host_z(num_points), host_intensity(num_points);
        
        for (size_t i = 0; i < num_points; ++i) {
            host_x[i] = input_cloud->points[i].x;
            host_y[i] = input_cloud->points[i].y;
            host_z[i] = input_cloud->points[i].z;
            host_intensity[i] = input_cloud->points[i].intensity;
        }
        
        if (kernel_config_.use_unified_memory) {
            std::memcpy(cuda_data_->d_input_x, host_x.data(), num_points * sizeof(float));
            std::memcpy(cuda_data_->d_input_y, host_y.data(), num_points * sizeof(float));
            std::memcpy(cuda_data_->d_input_z, host_z.data(), num_points * sizeof(float));
            std::memcpy(cuda_data_->d_input_intensity, host_intensity.data(), num_points * sizeof(float));
        } else {
            cudaMemcpy(cuda_data_->d_input_x, host_x.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(cuda_data_->d_input_y, host_y.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(cuda_data_->d_input_z, host_z.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(cuda_data_->d_input_intensity, host_intensity.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
        }
        
        // Initialize output point count to 0
        int zero = 0;
        cudaMemcpy(cuda_data_->d_num_output_points, &zero, sizeof(int), cudaMemcpyHostToDevice);
        
        // Launch ROI filter kernel
        cudaError_t error = cuda_roi_filter(
            cuda_data_->d_input_x, cuda_data_->d_input_y, cuda_data_->d_input_z, cuda_data_->d_input_intensity,
            cuda_data_->d_output_x, cuda_data_->d_output_y, cuda_data_->d_output_z, cuda_data_->d_output_intensity,
            cuda_data_->d_output_indices, num_points,
            min_pt.x, min_pt.y, min_pt.z, max_pt.x, max_pt.y, max_pt.z,
            cuda_data_->d_num_output_points, kernel_config_.block_size
        );
        
        if (error != cudaSuccess) {
            std::cerr << "CUDA ROI filter kernel failed: " << cudaGetErrorString(error) << std::endl;
            return cpu_fallback::ROIFilter(input_cloud, output_cloud, min_pt, max_pt);
        }
        
        // Get number of output points
        int num_output_points;
        cudaMemcpy(&num_output_points, cuda_data_->d_num_output_points, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Copy results back to host
        output_cloud->clear();
        output_cloud->resize(num_output_points);
        
        std::vector<float> out_x(num_output_points), out_y(num_output_points), 
                          out_z(num_output_points), out_intensity(num_output_points);
        
        if (kernel_config_.use_unified_memory) {
            cudaDeviceSynchronize();
            std::memcpy(out_x.data(), cuda_data_->d_output_x, num_output_points * sizeof(float));
            std::memcpy(out_y.data(), cuda_data_->d_output_y, num_output_points * sizeof(float));
            std::memcpy(out_z.data(), cuda_data_->d_output_z, num_output_points * sizeof(float));
            std::memcpy(out_intensity.data(), cuda_data_->d_output_intensity, num_output_points * sizeof(float));
        } else {
            cudaMemcpy(out_x.data(), cuda_data_->d_output_x, num_output_points * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(out_y.data(), cuda_data_->d_output_y, num_output_points * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(out_z.data(), cuda_data_->d_output_z, num_output_points * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(out_intensity.data(), cuda_data_->d_output_intensity, num_output_points * sizeof(float), cudaMemcpyDeviceToHost);
        }
        
        for (int i = 0; i < num_output_points; ++i) {
            output_cloud->points[i].x = out_x[i];
            output_cloud->points[i].y = out_y[i];
            output_cloud->points[i].z = out_z[i];
            output_cloud->points[i].intensity = out_intensity[i];
        }
        
        output_cloud->width = num_output_points;
        output_cloud->height = 1;
        output_cloud->is_dense = false;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double processing_time = std::chrono::duration<double>(end_time - start_time).count();
        UpdatePerformanceStats(true, processing_time, num_points);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in CUDA ROI filter: " << e.what() << std::endl;
        return cpu_fallback::ROIFilter(input_cloud, output_cloud, min_pt, max_pt);
    }
}

// Additional filter method implementations...

bool CudaMapFilter::ApplyFilters(const PointCloudType::Ptr& input_cloud,
                                PointCloudType::Ptr& output_cloud,
                                const FilterParams& params) {
    
    if (!cuda_initialized_) {
        return cuda_map_filter::cpu_fallback::ApplyFilters(input_cloud, output_cloud, params);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (!input_cloud || input_cloud->empty()) {
        output_cloud->clear();
        return true;
    }
    
    // Check for thermal throttling on Jetson
    if (jetson_info_.is_jetson && CheckThermalThrottling()) {
        return cuda_map_filter::cpu_fallback::ApplyFilters(input_cloud, output_cloud, params);
    }
    
    try {
        PointCloudType::Ptr working_cloud = input_cloud;
        
        // Apply filters in optimal order for CUDA processing
        
        // 1. ROI filtering (if enabled)
        if (params.use_roi) {
            PointCloudType::Ptr roi_filtered(new PointCloudType());
            if (!ROIFilter(working_cloud, roi_filtered, params.roi_min, params.roi_max)) {
                return cuda_map_filter::cpu_fallback::ApplyFilters(input_cloud, output_cloud, params);
            }
            working_cloud = roi_filtered;
        }
        
        // 2. Statistical outlier removal (if enabled)
        if (params.enable_outlier_removal) {
            PointCloudType::Ptr stat_filtered(new PointCloudType());
            if (!StatisticalOutlierRemoval(working_cloud, stat_filtered, 
                                          params.outlier_std_ratio, params.outlier_neighbors)) {
                return cuda_map_filter::cpu_fallback::ApplyFilters(input_cloud, output_cloud, params);
            }
            working_cloud = stat_filtered;
        }
        
        // 3. Radius outlier removal (if enabled)
        if (params.enable_radius_filtering) {
            PointCloudType::Ptr radius_filtered(new PointCloudType());
            if (!RadiusOutlierRemoval(working_cloud, radius_filtered, 
                                     params.radius_search, params.min_neighbors_in_radius)) {
                return cuda_map_filter::cpu_fallback::ApplyFilters(input_cloud, output_cloud, params);
            }
            working_cloud = radius_filtered;
        }
        
        // 4. Voxel grid filtering (if enabled) - should be last
        if (params.enable_voxel_filtering) {
            PointCloudType::Ptr voxel_filtered(new PointCloudType());
            if (!VoxelGridFilter(working_cloud, voxel_filtered, params.voxel_size)) {
                return cuda_map_filter::cpu_fallback::ApplyFilters(input_cloud, output_cloud, params);
            }
            working_cloud = voxel_filtered;
        }
        
        output_cloud = working_cloud;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double processing_time = std::chrono::duration<double>(end_time - start_time).count();
        UpdatePerformanceStats(true, processing_time, input_cloud->size());
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in CUDA ApplyFilters: " << e.what() << std::endl;
        return cuda_map_filter::cpu_fallback::ApplyFilters(input_cloud, output_cloud, params);
    }
}

bool CudaMapFilter::StatisticalOutlierRemoval(const PointCloudType::Ptr& input_cloud,
                                             PointCloudType::Ptr& output_cloud,
                                             float std_ratio,
                                             int neighbors) {
    
    if (!cuda_initialized_) {
        return cuda_map_filter::cpu_fallback::StatisticalOutlierRemoval(input_cloud, output_cloud, std_ratio, neighbors);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    size_t num_points = input_cloud->size();
    if (num_points == 0 || num_points < static_cast<size_t>(neighbors)) {
        *output_cloud = *input_cloud;
        return true;
    }
    
    // For now, fall back to CPU implementation for statistical outlier removal
    // as it requires complex neighbor search that would need additional CUDA implementation
    bool success = cuda_map_filter::cpu_fallback::StatisticalOutlierRemoval(input_cloud, output_cloud, std_ratio, neighbors);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double processing_time = std::chrono::duration<double>(end_time - start_time).count();
    UpdatePerformanceStats(false, processing_time, num_points); // CPU fallback used
    
    return success;
}

bool CudaMapFilter::RadiusOutlierRemoval(const PointCloudType::Ptr& input_cloud,
                                        PointCloudType::Ptr& output_cloud,
                                        float radius,
                                        int min_neighbors) {
    
    if (!cuda_initialized_) {
        return cuda_map_filter::cpu_fallback::RadiusOutlierRemoval(input_cloud, output_cloud, radius, min_neighbors);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    size_t num_points = input_cloud->size();
    if (num_points == 0) {
        output_cloud->clear();
        return true;
    }
    
    // Check for thermal throttling on Jetson
    if (jetson_info_.is_jetson && CheckThermalThrottling()) {
        return cuda_map_filter::cpu_fallback::RadiusOutlierRemoval(input_cloud, output_cloud, radius, min_neighbors);
    }
    
    // For large point clouds with radius outlier removal, the O(n²) complexity makes it expensive
    // Use CPU fallback for very large clouds or fall back based on platform capabilities
    size_t max_cuda_points = jetson_info_.is_jetson ? 50000 : 200000;
    
    if (num_points > max_cuda_points) {
        bool success = cuda_map_filter::cpu_fallback::RadiusOutlierRemoval(input_cloud, output_cloud, radius, min_neighbors);
        auto end_time = std::chrono::high_resolution_clock::now();
        double processing_time = std::chrono::duration<double>(end_time - start_time).count();
        UpdatePerformanceStats(false, processing_time, num_points);
        return success;
    }
    
    if (!EnsureGpuMemory(num_points)) {
        return cuda_map_filter::cpu_fallback::RadiusOutlierRemoval(input_cloud, output_cloud, radius, min_neighbors);
    }
    
    try {
        // Copy point cloud data to GPU
        std::vector<float> host_x(num_points), host_y(num_points), host_z(num_points), host_intensity(num_points);
        
        for (size_t i = 0; i < num_points; ++i) {
            host_x[i] = input_cloud->points[i].x;
            host_y[i] = input_cloud->points[i].y;
            host_z[i] = input_cloud->points[i].z;
            host_intensity[i] = input_cloud->points[i].intensity;
        }
        
        if (kernel_config_.use_unified_memory) {
            std::memcpy(cuda_data_->d_input_x, host_x.data(), num_points * sizeof(float));
            std::memcpy(cuda_data_->d_input_y, host_y.data(), num_points * sizeof(float));
            std::memcpy(cuda_data_->d_input_z, host_z.data(), num_points * sizeof(float));
            std::memcpy(cuda_data_->d_input_intensity, host_intensity.data(), num_points * sizeof(float));
        } else {
            cudaMemcpy(cuda_data_->d_input_x, host_x.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(cuda_data_->d_input_y, host_y.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(cuda_data_->d_input_z, host_z.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(cuda_data_->d_input_intensity, host_intensity.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);
        }
        
        // Initialize output point count to 0
        int zero = 0;
        cudaMemcpy(cuda_data_->d_num_output_points, &zero, sizeof(int), cudaMemcpyHostToDevice);
        
        // Launch radius outlier removal kernel
        cudaError_t error = cuda_radius_outlier_removal(
            cuda_data_->d_input_x, cuda_data_->d_input_y, cuda_data_->d_input_z, cuda_data_->d_input_intensity,
            cuda_data_->d_output_x, cuda_data_->d_output_y, cuda_data_->d_output_z, cuda_data_->d_output_intensity,
            num_points, radius, min_neighbors, cuda_data_->d_num_output_points, kernel_config_.block_size
        );
        
        if (error != cudaSuccess) {
            std::cerr << "CUDA radius outlier removal kernel failed: " << cudaGetErrorString(error) << std::endl;
            return cuda_map_filter::cpu_fallback::RadiusOutlierRemoval(input_cloud, output_cloud, radius, min_neighbors);
        }
        
        // Get number of output points
        int num_output_points;
        cudaMemcpy(&num_output_points, cuda_data_->d_num_output_points, sizeof(int), cudaMemcpyDeviceToHost);
        
        // Copy results back to host
        output_cloud->clear();
        output_cloud->resize(num_output_points);
        
        std::vector<float> out_x(num_output_points), out_y(num_output_points), 
                          out_z(num_output_points), out_intensity(num_output_points);
        
        if (kernel_config_.use_unified_memory) {
            cudaDeviceSynchronize();
            std::memcpy(out_x.data(), cuda_data_->d_output_x, num_output_points * sizeof(float));
            std::memcpy(out_y.data(), cuda_data_->d_output_y, num_output_points * sizeof(float));
            std::memcpy(out_z.data(), cuda_data_->d_output_z, num_output_points * sizeof(float));
            std::memcpy(out_intensity.data(), cuda_data_->d_output_intensity, num_output_points * sizeof(float));
        } else {
            cudaMemcpy(out_x.data(), cuda_data_->d_output_x, num_output_points * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(out_y.data(), cuda_data_->d_output_y, num_output_points * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(out_z.data(), cuda_data_->d_output_z, num_output_points * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(out_intensity.data(), cuda_data_->d_output_intensity, num_output_points * sizeof(float), cudaMemcpyDeviceToHost);
        }
        
        for (int i = 0; i < num_output_points; ++i) {
            output_cloud->points[i].x = out_x[i];
            output_cloud->points[i].y = out_y[i];
            output_cloud->points[i].z = out_z[i];
            output_cloud->points[i].intensity = out_intensity[i];
        }
        
        output_cloud->width = num_output_points;
        output_cloud->height = 1;
        output_cloud->is_dense = false;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double processing_time = std::chrono::duration<double>(end_time - start_time).count();
        UpdatePerformanceStats(true, processing_time, num_points);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in CUDA radius outlier removal: " << e.what() << std::endl;
        return cuda_map_filter::cpu_fallback::RadiusOutlierRemoval(input_cloud, output_cloud, radius, min_neighbors);
    }
}

bool CudaMapFilter::ProcessInBatches(const PointCloudType::Ptr& input_cloud,
                                    PointCloudType::Ptr& output_cloud,
                                    size_t batch_size,
                                    std::function<bool(const PointCloudType::Ptr&, PointCloudType::Ptr&)> filter_func) {
    
    output_cloud->clear();
    size_t total_points = input_cloud->size();
    
    for (size_t i = 0; i < total_points; i += batch_size) {
        size_t end_idx = std::min(i + batch_size, total_points);
        size_t current_batch_size = end_idx - i;
        
        // Create batch point cloud
        PointCloudType::Ptr batch_input(new PointCloudType());
        batch_input->resize(current_batch_size);
        
        for (size_t j = 0; j < current_batch_size; ++j) {
            batch_input->points[j] = input_cloud->points[i + j];
        }
        
        batch_input->width = current_batch_size;
        batch_input->height = 1;
        batch_input->is_dense = false;
        
        // Process batch
        PointCloudType::Ptr batch_output(new PointCloudType());
        if (!filter_func(batch_input, batch_output)) {
            return false;
        }
        
        // Merge results
        for (const auto& point : batch_output->points) {
            output_cloud->points.push_back(point);
        }
    }
    
    output_cloud->width = output_cloud->points.size();
    output_cloud->height = 1;
    output_cloud->is_dense = false;
    
    return true;
}

bool CudaMapFilter::VoxelGridFilter(const PointCloudType::Ptr& input_cloud,
                                   PointCloudType::Ptr& output_cloud,
                                   float voxel_size) {
    // Simple CPU implementation for now
    // TODO: Implement CUDA version
    pcl::VoxelGrid<PointType> vg;
    vg.setInputCloud(input_cloud);
    vg.setLeafSize(voxel_size, voxel_size, voxel_size);
    vg.filter(*output_cloud);
    
    return true;
}

void CudaMapFilter::UpdatePerformanceStats(bool cuda_used, double processing_time, size_t points_processed) {
    perf_stats_.total_operations++;
    if (cuda_used) {
        perf_stats_.cuda_operations++;
    } else {
        perf_stats_.cpu_fallbacks++;
    }
    
    // Update average processing time
    double total_time = perf_stats_.average_processing_time * (perf_stats_.total_operations - 1) + processing_time;
    perf_stats_.average_processing_time = total_time / perf_stats_.total_operations;
    
    perf_stats_.total_points_processed += points_processed;
}

CudaMapFilter::PerformanceStats CudaMapFilter::GetPerformanceStats() const {
    return perf_stats_;
}

void CudaMapFilter::ResetPerformanceStats() {
    perf_stats_ = PerformanceStats();
}

void CudaMapFilter::ClearMemory() {
    if (cuda_data_) {
        DeallocateGpuMemory();
    }
}

size_t CudaMapFilter::GetAllocatedMemory() const {
    return allocated_points_ * sizeof(PointType);
}

} // namespace cuda_map_filter

#endif // USE_CUDA
