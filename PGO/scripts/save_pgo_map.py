#!/usr/bin/env python3
"""
Enhanced PGO Map Service Client
 
This script provides an easy way to call the enhanced map service with outlier removal.

Usage examples:
  # Basic usage - save with default settings
  python3 save_pgo_map.py --output /path/to/map.pcd
  
  # With outlier removal
  python3 save_pgo_map.py --output /path/to/map.pcd --outlier-removal --voxel-filter --voxel-size 0.1
  
  # With all filters
  python3 save_pgo_map.py --output /path/to/map.pcd --outlier-removal --radius-filter --voxel-filter
"""

import rospy
import argparse
import sys
from aloam_velodyne.srv import SaveOptimizedMap, SaveOptimizedMapRequest
from geometry_msgs.msg import Point

def save_pgo_map(args):
    rospy.init_node('pgo_map_client', anonymous=True)
    
    # Wait for service
    service_name = '/enhanced_map_service/save_optimized_map'
    print(f"Waiting for service {service_name}...")
    rospy.wait_for_service(service_name)
    
    try:
        # Create service proxy
        save_map_service = rospy.ServiceProxy(service_name, SaveOptimizedMap)
        
        # Create request
        req = SaveOptimizedMapRequest()
        req.output_path = args.output
        req.file_format = args.format
        req.enable_outlier_removal = args.outlier_removal
        req.outlier_std_ratio = args.outlier_std_ratio
        req.outlier_neighbors = args.outlier_neighbors
        req.enable_voxel_filtering = args.voxel_filter
        req.voxel_size = args.voxel_size
        req.enable_radius_filtering = args.radius_filter
        req.radius_search = args.radius_search
        req.min_neighbors_in_radius = args.min_neighbors_radius
        req.compress_binary = args.binary
        req.include_intensity = args.intensity
        req.use_roi = args.use_roi
        
        # Set ROI if specified
        if args.use_roi:
            req.roi_min = Point(x=args.roi_min[0], y=args.roi_min[1], z=args.roi_min[2])
            req.roi_max = Point(x=args.roi_max[0], y=args.roi_max[1], z=args.roi_max[2])
        
        print("Calling map save service...")
        print(f"  Output path: {req.output_path}")
        print(f"  Outlier removal: {req.enable_outlier_removal}")
        print(f"  Voxel filtering: {req.enable_voxel_filtering}")
        print(f"  Radius filtering: {req.enable_radius_filtering}")
        
        # Call service
        response = save_map_service(req)
        
        # Print results
        if response.success:
            print("\n✓ Map saved successfully!")
            print(f"  File path: {response.saved_file_path}")
            print(f"  Original points: {response.original_points:,}")
            print(f"  Filtered points: {response.filtered_points:,}")
            print(f"  Compression ratio: {response.compression_ratio:.3f}")
            print(f"  Processing time: {response.processing_time:.2f} seconds")
            if response.compression_ratio < 0.8:
                print(f"  ⚠️  High compression ratio - significant noise removed!")
        else:
            print(f"\n✗ Failed to save map: {response.message}")
            sys.exit(1)
            
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Save PGO map with outlier removal')
    
    # Required arguments
    parser.add_argument('--output', '-o', default="/home/surya/workspaces/slam_ws/src/FAST_LIO_LC/maps/map.pcd",
                       help='Output file path (e.g., /path/to/map.pcd)')
    
    # File format options
    parser.add_argument('--format', '-f', default='pcd', choices=['pcd', 'ply'],
                       help='Output file format (default: pcd)')
    parser.add_argument('--binary', action='store_true',default=True,
                       help='Save as binary format (smaller file size)')
    parser.add_argument('--intensity', action='store_true', default=True,
                       help='Include intensity values (default: true)')
    
    # Outlier removal options - Conservative settings to preserve structure
    parser.add_argument('--outlier-removal', action='store_true',default=False,
                       help='Enable statistical outlier removal')
    parser.add_argument('--outlier-std-ratio', type=float, default=2.5,
                       help='Standard deviation ratio for outlier removal (default: 2.5 - conservative)')
    parser.add_argument('--outlier-neighbors', type=int, default=20,
                       help='Number of neighbors for outlier analysis (default: 20)')

    # Radius filter options - More permissive to keep dense areas
    parser.add_argument('--radius-filter', action='store_true',default=True,
                       help='Enable radius outlier removal')
    parser.add_argument('--radius-search', type=float, default=0.5,
                       help='Search radius for radius filtering (default: 0.5)')
    parser.add_argument('--min-neighbors-radius', type=int, default=10,
                       help='Minimum neighbors within radius (default: 10 - keeps more points)')

    # Voxel filter options - Smaller voxel size for denser map
    parser.add_argument('--voxel-filter', action='store_true',default=True,
                       help='Enable voxel grid downsampling')
    parser.add_argument('--voxel-size', type=float, default=0.05,
                       help='Voxel grid leaf size in meters (default: 0.02 - smaller for density)')

    # ROI options
    parser.add_argument('--use-roi', action='store_true',
                       help='Enable region of interest filtering')
    parser.add_argument('--roi-min', type=float, nargs=3, default=[-100, -100, -10],
                       metavar=('X', 'Y', 'Z'), help='ROI minimum bounds (default: -100 -100 -10)')
    parser.add_argument('--roi-max', type=float, nargs=3, default=[100, 100, 10],
                       metavar=('X', 'Y', 'Z'), help='ROI maximum bounds (default: 100 100 10)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.output:
        print("Error: Output path is required")
        sys.exit(1)
    
    # Add file extension if missing
    if '.' not in args.output:
        args.output += f'.{args.format}'
    
    save_pgo_map(args)

if __name__ == '__main__':
    main()
