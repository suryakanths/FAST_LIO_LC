#!/usr/bin/env python3

"""
CUDA-Enhanced Map Service Test Script

This script demonstrates and tests the CUDA-accelerated map filtering capabilities
of the enhanced map service. It provides various test scenarios for different
filtering combinations and platform configurations.

Usage:
    python3 test_cuda_map_service.py [--cuda] [--jetson] [--benchmark]
    
Arguments:
    --cuda      Force CUDA testing (default: auto-detect)
    --jetson    Force Jetson optimizations (default: auto-detect)
    --benchmark Run performance benchmarks
"""

import rospy
import sys
import time
import argparse
from aloam_velodyne.srv import SaveOptimizedMap, SaveOptimizedMapRequest
from geometry_msgs.msg import Point

class CudaMapServiceTester:
    def __init__(self):
        rospy.init_node('cuda_map_service_tester', anonymous=True)
        
        # Wait for service to be available
        rospy.loginfo("Waiting for enhanced map service...")
        rospy.wait_for_service('/enhanced_map_service/save_optimized_map')
        
        self.service_proxy = rospy.ServiceProxy('/enhanced_map_service/save_optimized_map', SaveOptimizedMap)
        rospy.loginfo("Enhanced map service is available!")
        
    def test_basic_save(self):
        """Test basic map saving without filters"""
        rospy.loginfo("=== Testing Basic Map Save ===")
        
        request = SaveOptimizedMapRequest()
        request.output_path = "/tmp/test_basic_map.pcd"
        request.file_format = "pcd"
        request.compress_binary = True
        
        try:
            response = self.service_proxy(request)
            self._print_response("Basic Save", response)
            return response.success
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False
    
    def test_outlier_removal(self):
        """Test statistical and radius outlier removal"""
        rospy.loginfo("=== Testing Outlier Removal ===")
        
        request = SaveOptimizedMapRequest()
        request.output_path = "/tmp/test_outlier_filtered_map.pcd"
        request.file_format = "pcd"
        request.compress_binary = True
        
        # Enable statistical outlier removal
        request.enable_outlier_removal = True
        request.outlier_std_ratio = 1.0
        request.outlier_neighbors = 50
        
        # Enable radius outlier removal
        request.enable_radius_filtering = True
        request.radius_search = 0.5
        request.min_neighbors_in_radius = 5
        
        try:
            response = self.service_proxy(request)
            self._print_response("Outlier Removal", response)
            return response.success
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False
    
    def test_voxel_filtering(self):
        """Test voxel grid downsampling"""
        rospy.loginfo("=== Testing Voxel Grid Filtering ===")
        
        request = SaveOptimizedMapRequest()
        request.output_path = "/tmp/test_voxel_filtered_map.pcd"
        request.file_format = "pcd"
        request.compress_binary = True
        
        # Enable voxel grid filtering
        request.enable_voxel_filtering = True
        request.voxel_size = 0.1
        
        try:
            response = self.service_proxy(request)
            self._print_response("Voxel Filtering", response)
            return response.success
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False
    
    def test_roi_filtering(self):
        """Test Region of Interest filtering"""
        rospy.loginfo("=== Testing ROI Filtering ===")
        
        request = SaveOptimizedMapRequest()
        request.output_path = "/tmp/test_roi_filtered_map.pcd"
        request.file_format = "pcd"
        request.compress_binary = True
        
        # Enable ROI filtering
        request.use_roi = True
        request.roi_min = Point(x=-50.0, y=-50.0, z=-5.0)
        request.roi_max = Point(x=50.0, y=50.0, z=5.0)
        
        try:
            response = self.service_proxy(request)
            self._print_response("ROI Filtering", response)
            return response.success
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False
    
    def test_combined_filtering(self):
        """Test all filters combined"""
        rospy.loginfo("=== Testing Combined Filtering (All Filters) ===")
        
        request = SaveOptimizedMapRequest()
        request.output_path = "/tmp/test_combined_filtered_map.pcd"
        request.file_format = "pcd"
        request.compress_binary = True
        
        # Enable ROI filtering
        request.use_roi = True
        request.roi_min = Point(x=-100.0, y=-100.0, z=-10.0)
        request.roi_max = Point(x=100.0, y=100.0, z=10.0)
        
        # Enable statistical outlier removal
        request.enable_outlier_removal = True
        request.outlier_std_ratio = 1.0
        request.outlier_neighbors = 50
        
        # Enable radius outlier removal
        request.enable_radius_filtering = True
        request.radius_search = 0.5
        request.min_neighbors_in_radius = 5
        
        # Enable voxel grid filtering
        request.enable_voxel_filtering = True
        request.voxel_size = 0.1
        
        try:
            response = self.service_proxy(request)
            self._print_response("Combined Filtering", response)
            return response.success
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False
    
    def test_performance_benchmark(self):
        """Run performance benchmarks with different voxel sizes"""
        rospy.loginfo("=== Running Performance Benchmarks ===")
        
        voxel_sizes = [0.05, 0.1, 0.2, 0.5]
        results = []
        
        for voxel_size in voxel_sizes:
            rospy.loginfo(f"Benchmarking voxel size: {voxel_size}")
            
            request = SaveOptimizedMapRequest()
            request.output_path = f"/tmp/benchmark_voxel_{voxel_size}.pcd"
            request.file_format = "pcd"
            request.compress_binary = True
            request.enable_voxel_filtering = True
            request.voxel_size = voxel_size
            
            start_time = time.time()
            try:
                response = self.service_proxy(request)
                end_time = time.time()
                
                if response.success:
                    total_time = end_time - start_time
                    results.append({
                        'voxel_size': voxel_size,
                        'original_points': response.original_points,
                        'filtered_points': response.filtered_points,
                        'compression_ratio': response.compression_ratio,
                        'processing_time': response.processing_time,
                        'total_time': total_time
                    })
                    
                    rospy.loginfo(f"  Voxel {voxel_size}: {response.original_points} -> {response.filtered_points} points "
                                f"({response.compression_ratio:.3f} ratio) in {response.processing_time:.3f}s")
                
            except rospy.ServiceException as e:
                rospy.logerr(f"Benchmark failed for voxel size {voxel_size}: {e}")
        
        # Print benchmark summary
        rospy.loginfo("=== Benchmark Results ===")
        for result in results:
            rospy.loginfo(f"Voxel {result['voxel_size']}: "
                         f"{result['original_points']} -> {result['filtered_points']} points, "
                         f"ratio: {result['compression_ratio']:.3f}, "
                         f"processing: {result['processing_time']:.3f}s, "
                         f"total: {result['total_time']:.3f}s")
        
        return len(results) > 0
    
    def test_format_support(self):
        """Test different file format support"""
        rospy.loginfo("=== Testing File Format Support ===")
        
        formats = ["pcd", "ply"]
        success_count = 0
        
        for file_format in formats:
            rospy.loginfo(f"Testing {file_format.upper()} format...")
            
            request = SaveOptimizedMapRequest()
            request.output_path = f"/tmp/test_format.{file_format}"
            request.file_format = file_format
            request.compress_binary = True
            request.enable_voxel_filtering = True
            request.voxel_size = 0.1
            
            try:
                response = self.service_proxy(request)
                if response.success:
                    rospy.loginfo(f"  {file_format.upper()} format: SUCCESS")
                    success_count += 1
                else:
                    rospy.logwarn(f"  {file_format.upper()} format: FAILED - {response.message}")
            except rospy.ServiceException as e:
                rospy.logerr(f"  {file_format.upper()} format: ERROR - {e}")
        
        return success_count == len(formats)
    
    def _print_response(self, test_name, response):
        """Print formatted response information"""
        if response.success:
            rospy.loginfo(f"{test_name}: SUCCESS")
            rospy.loginfo(f"  Original points: {response.original_points}")
            rospy.loginfo(f"  Filtered points: {response.filtered_points}")
            rospy.loginfo(f"  Compression ratio: {response.compression_ratio:.3f}")
            rospy.loginfo(f"  Processing time: {response.processing_time:.3f} seconds")
            rospy.loginfo(f"  Saved to: {response.saved_file_path}")
            rospy.loginfo(f"  Message: {response.message}")
        else:
            rospy.logerr(f"{test_name}: FAILED")
            rospy.logerr(f"  Message: {response.message}")
    
    def run_all_tests(self, benchmark=False):
        """Run all test scenarios"""
        rospy.loginfo("Starting CUDA-Enhanced Map Service Tests...")
        
        tests = [
            ("Basic Save", self.test_basic_save),
            ("Outlier Removal", self.test_outlier_removal),
            ("Voxel Filtering", self.test_voxel_filtering),
            ("ROI Filtering", self.test_roi_filtering),
            ("Combined Filtering", self.test_combined_filtering),
            ("Format Support", self.test_format_support),
        ]
        
        if benchmark:
            tests.append(("Performance Benchmark", self.test_performance_benchmark))
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            rospy.loginfo(f"\n{'='*50}")
            try:
                if test_func():
                    passed += 1
                    rospy.loginfo(f"✓ {test_name} PASSED")
                else:
                    rospy.logwarn(f"✗ {test_name} FAILED")
            except Exception as e:
                rospy.logerr(f"✗ {test_name} ERROR: {e}")
            
            # Wait between tests
            time.sleep(1)
        
        rospy.loginfo(f"\n{'='*50}")
        rospy.loginfo(f"Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            rospy.loginfo("🎉 All tests completed successfully!")
            rospy.loginfo("CUDA-enhanced map filtering is working correctly.")
        else:
            rospy.logwarn(f"⚠️  {total - passed} test(s) failed. Check the logs for details.")
        
        return passed == total

def main():
    parser = argparse.ArgumentParser(description='Test CUDA-Enhanced Map Service')
    parser.add_argument('--cuda', action='store_true', help='Force CUDA testing')
    parser.add_argument('--jetson', action='store_true', help='Force Jetson optimizations')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--test', choices=['basic', 'outlier', 'voxel', 'roi', 'combined', 'format', 'all'], 
                       default='all', help='Specific test to run')
    
    args = parser.parse_args()
    
    try:
        tester = CudaMapServiceTester()
        
        if args.test == 'all':
            success = tester.run_all_tests(benchmark=args.benchmark)
        else:
            # Run specific test
            test_methods = {
                'basic': tester.test_basic_save,
                'outlier': tester.test_outlier_removal,
                'voxel': tester.test_voxel_filtering,
                'roi': tester.test_roi_filtering,
                'combined': tester.test_combined_filtering,
                'format': tester.test_format_support,
            }
            
            if args.test in test_methods:
                success = test_methods[args.test]()
            else:
                rospy.logerr(f"Unknown test: {args.test}")
                return 1
        
        return 0 if success else 1
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted by user")
        return 1
    except Exception as e:
        rospy.logerr(f"Test failed with exception: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
