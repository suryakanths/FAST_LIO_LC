#!/bin/bash

# CUDA-Enhanced Map Service Build and Test Script
# This script builds the enhanced map service with CUDA support and runs tests

set -e  # Exit on any error

echo "=========================================="
echo "CUDA-Enhanced Map Service Build & Test"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the correct workspace
if [[ ! -d "/home/surya/workspaces/slam_ws/src/FAST_LIO_LC" ]]; then
    print_error "Please run this script from the SLAM workspace directory"
    exit 1
fi

cd /home/surya/workspaces/slam_ws

# Check CUDA availability
print_status "Checking CUDA availability..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    print_success "CUDA found: version $CUDA_VERSION"
    CUDA_AVAILABLE=true
else
    print_warning "CUDA not found. Building with CPU-only support."
    CUDA_AVAILABLE=false
fi

# Check if running on Jetson
if [[ -f "/proc/device-tree/model" ]]; then
    JETSON_MODEL=$(cat /proc/device-tree/model)
    if [[ $JETSON_MODEL == *"NVIDIA Jetson"* ]]; then
        print_success "Jetson platform detected: $JETSON_MODEL"
        JETSON_PLATFORM=true
    else
        JETSON_PLATFORM=false
    fi
else
    JETSON_PLATFORM=false
fi

# Function to build the workspace
build_workspace() {
    print_status "Building workspace with CUDA support..."
    
    # Source ROS environment
    source /opt/ros/noetic/setup.bash
    
    # Clean build (optional)
    if [[ "$1" == "--clean" ]]; then
        print_status "Cleaning previous build..."
        rm -rf build/ devel/
    fi
    
    # Build with CUDA support
    if [[ $CUDA_AVAILABLE == true ]]; then
        print_status "Building with CUDA acceleration enabled..."
        catkin_make -DUSE_CUDA=ON
    else
        print_status "Building with CPU-only support..."
        catkin_make -DUSE_CUDA=OFF
    fi
    
    if [[ $? -eq 0 ]]; then
        print_success "Build completed successfully!"
        return 0
    else
        print_error "Build failed!"
        return 1
    fi
}

# Function to test the enhanced map service
test_map_service() {
    print_status "Testing enhanced map service..."
    
    # Source workspace
    source devel/setup.bash
    
    # Start roscore in background if not running
    if ! pgrep -x "roscore" > /dev/null; then
        print_status "Starting roscore..."
        roscore &
        ROSCORE_PID=$!
        sleep 3
    fi
    
    # Launch the enhanced map service
    print_status "Launching enhanced map service..."
    roslaunch aloam_velodyne enhanced_map_service_cuda.launch &
    SERVICE_PID=$!
    sleep 5
    
    # Check if service is running
    if ! ps -p $SERVICE_PID > /dev/null; then
        print_error "Map service failed to start"
        return 1
    fi
    
    # Run tests
    print_status "Running test suite..."
    cd src/FAST_LIO_LC/PGO/scripts
    chmod +x test_cuda_map_service.py
    
    if [[ $CUDA_AVAILABLE == true ]]; then
        python3 test_cuda_map_service.py --cuda --benchmark
    else
        python3 test_cuda_map_service.py --benchmark
    fi
    
    TEST_RESULT=$?
    
    # Cleanup
    print_status "Cleaning up test processes..."
    kill $SERVICE_PID 2>/dev/null || true
    if [[ ! -z "$ROSCORE_PID" ]]; then
        kill $ROSCORE_PID 2>/dev/null || true
    fi
    
    cd /home/surya/workspaces/slam_ws
    
    if [[ $TEST_RESULT -eq 0 ]]; then
        print_success "All tests passed!"
        return 0
    else
        print_error "Some tests failed!"
        return 1
    fi
}

# Function to show system information
show_system_info() {
    print_status "System Information:"
    echo "  OS: $(lsb_release -d | cut -f2)"
    echo "  Architecture: $(uname -m)"
    echo "  Kernel: $(uname -r)"
    
    if [[ $CUDA_AVAILABLE == true ]]; then
        echo "  CUDA: $CUDA_VERSION"
        if command -v nvidia-smi &> /dev/null; then
            echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
        fi
    else
        echo "  CUDA: Not available"
    fi
    
    if [[ $JETSON_PLATFORM == true ]]; then
        echo "  Platform: $JETSON_MODEL"
    else
        echo "  Platform: Desktop/Server"
    fi
    
    echo "  ROS: $(rosversion -d)"
    echo "  PCL: $(pkg-config --modversion pcl_common 2>/dev/null || echo 'Not found via pkg-config')"
}

# Function to display performance tips
show_performance_tips() {
    print_status "Performance Tips:"
    
    if [[ $JETSON_PLATFORM == true ]]; then
        echo "  🚀 Jetson-specific optimizations:"
        echo "     - Unified memory is automatically used"
        echo "     - Thermal monitoring is enabled"
        echo "     - Block sizes are optimized for your Jetson model"
        echo "     - Consider enabling max performance mode:"
        echo "       sudo nvpmodel -m 0 && sudo jetson_clocks"
    fi
    
    if [[ $CUDA_AVAILABLE == true ]]; then
        echo "  ⚡ CUDA optimizations:"
        echo "     - GPU memory pooling reduces allocation overhead"
        echo "     - Platform-specific block sizes are used"
        echo "     - Automatic CPU fallback for unsupported operations"
        echo "     - Monitor GPU usage with: nvidia-smi"
    fi
    
    echo "  🛠️  General tips:"
    echo "     - Use appropriate voxel sizes (0.05-0.2m typically)"
    echo "     - Enable outlier removal for cleaner maps"
    echo "     - ROI filtering reduces processing time"
    echo "     - Binary PCD format is faster than ASCII"
}

# Main execution
case "${1:-build}" in
    "build")
        show_system_info
        echo
        build_workspace $2
        ;;
    "test")
        test_map_service
        ;;
    "clean")
        print_status "Cleaning workspace..."
        rm -rf build/ devel/
        print_success "Workspace cleaned!"
        ;;
    "full")
        show_system_info
        echo
        build_workspace --clean
        if [[ $? -eq 0 ]]; then
            echo
            test_map_service
            echo
            show_performance_tips
        fi
        ;;
    "info")
        show_system_info
        echo
        show_performance_tips
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command] [options]"
        echo
        echo "Commands:"
        echo "  build [--clean]  Build the workspace (clean build if --clean)"
        echo "  test            Run tests for the enhanced map service"
        echo "  clean           Clean the workspace"
        echo "  full            Clean build + test + show tips"
        echo "  info            Show system information and tips"
        echo "  help            Show this help message"
        echo
        echo "Examples:"
        echo "  $0 build        # Regular build"
        echo "  $0 build --clean # Clean build"
        echo "  $0 full         # Complete build and test cycle"
        echo "  $0 test         # Run tests only"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

exit 0
