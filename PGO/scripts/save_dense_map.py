#!/usr/bin/env python3
"""
Dense Map Saver - Optimized for preserving structures while removing noise

This script uses very conservative filtering settings to create dense maps
that preserve floors, walls, and other important structures while removing
sparse noise points.

Usage:
  # Ultra-dense map (minimal filtering)
  python3 save_dense_map.py --mode ultra-dense
  
  # Dense map (light noise removal)
  python3 save_dense_map.py --mode dense
  
  # Balanced dense map (moderate noise removal)
  python3 save_dense_map.py --mode balanced
  
  # Custom settings
  python3 save_dense_map.py --custom --outlier-std-ratio 3.0 --voxel-size 0.015
"""

import rospy
import argparse
import sys
from aloam_velodyne.srv import SaveOptimizedMap, SaveOptimizedMapRequest
from geometry_msgs.msg import Point

def get_preset_params(mode):
    """Get preset parameters for different density modes"""
    presets = {
        'ultra-dense': {
            'outlier_removal': False,  # Disable outlier removal
            'radius_filter': True,
            'voxel_filter': True,
            'outlier_std_ratio': 3.0,
            'outlier_neighbors': 30,
            'radius_search': 0.1,
            'min_neighbors_radius': 2,
            'voxel_size': 0.01
        },
        'dense': {
            'outlier_removal': True,
            'radius_filter': True,
            'voxel_filter': True,
            'outlier_std_ratio': 3.0,  # Very conservative
            'outlier_neighbors': 15,    # Fewer neighbors for faster processing
            'radius_search': 0.12,
            'min_neighbors_radius': 2,  # Very permissive
            'voxel_size': 0.015         # Small voxels for density
        },
        'balanced': {
            'outlier_removal': True,
            'radius_filter': True,
            'voxel_filter': True,
            'outlier_std_ratio': 2.5,  # Conservative
            'outlier_neighbors': 20,
            'radius_search': 0.15,
            'min_neighbors_radius': 3,
            'voxel_size': 0.02
        }
    }
    return presets.get(mode, presets['balanced'])

def save_dense_map(args):
    rospy.init_node('dense_map_client', anonymous=True)
    
    # Wait for service
    service_name = '/enhanced_map_service/save_optimized_map'
    print(f"Waiting for service {service_name}...")
    rospy.wait_for_service(service_name)
    
    try:
        # Create service proxy
        save_map_service = rospy.ServiceProxy(service_name, SaveOptimizedMap)
        
        # Get parameters based on mode or custom settings
        if args.mode and not args.custom:
            params = get_preset_params(args.mode)
            print(f"Using preset mode: {args.mode}")
        else:
            # Use custom parameters
            params = {
                'outlier_removal': args.outlier_removal,
                'radius_filter': args.radius_filter,
                'voxel_filter': args.voxel_filter,
                'outlier_std_ratio': args.outlier_std_ratio,
                'outlier_neighbors': args.outlier_neighbors,
                'radius_search': args.radius_search,
                'min_neighbors_radius': args.min_neighbors_radius,
                'voxel_size': args.voxel_size
            }
            print("Using custom parameters")
        
        # Create request
        req = SaveOptimizedMapRequest()
        req.output_path = args.output
        req.file_format = args.format
        req.enable_outlier_removal = params['outlier_removal']
        req.outlier_std_ratio = params['outlier_std_ratio']
        req.outlier_neighbors = params['outlier_neighbors']
        req.enable_voxel_filtering = params['voxel_filter']
        req.voxel_size = params['voxel_size']
        req.enable_radius_filtering = params['radius_filter']
        req.radius_search = params['radius_search']
        req.min_neighbors_in_radius = params['min_neighbors_radius']
        req.compress_binary = args.binary
        req.include_intensity = args.intensity
        req.use_roi = args.use_roi
        
        # Set ROI if specified
        if args.use_roi:
            req.roi_min = Point(x=args.roi_min[0], y=args.roi_min[1], z=args.roi_min[2])
            req.roi_max = Point(x=args.roi_max[0], y=args.roi_max[1], z=args.roi_max[2])
        
        print("\nDense Map Configuration:")
        print(f"  Output path: {req.output_path}")
        print(f"  Outlier removal: {req.enable_outlier_removal}")
        if req.enable_outlier_removal:
            print(f"    Std ratio: {req.outlier_std_ratio} (higher = more permissive)")
            print(f"    Neighbors: {req.outlier_neighbors}")
        print(f"  Radius filtering: {req.enable_radius_filtering}")
        if req.enable_radius_filtering:
            print(f"    Search radius: {req.radius_search}m")
            print(f"    Min neighbors: {req.min_neighbors_in_radius}")
        print(f"  Voxel filtering: {req.enable_voxel_filtering}")
        if req.enable_voxel_filtering:
            print(f"    Voxel size: {req.voxel_size}m (smaller = denser)")
        
        print("\nCalling dense map save service...")
        
        # Call service
        response = save_map_service(req)
        
        # Print results
        if response.success:
            print("\n✓ Dense map saved successfully!")
            print(f"  File path: {response.saved_file_path}")
            print(f"  Original points: {response.original_points:,}")
            print(f"  Filtered points: {response.filtered_points:,}")
            print(f"  Compression ratio: {response.compression_ratio:.3f}")
            print(f"  Processing time: {response.processing_time:.2f} seconds")
            
            # Density analysis
            density_retained = response.compression_ratio * 100
            print(f"\nDensity Analysis:")
            print(f"  {density_retained:.1f}% of original points retained")
            
            if density_retained > 90:
                print("  🟢 Ultra-dense map - minimal filtering applied")
            elif density_retained > 75:
                print("  🟡 Dense map - light filtering for noise removal")
            elif density_retained > 50:
                print("  🟠 Moderate filtering - balanced density/quality")
            else:
                print("  🔴 Heavy filtering - significant noise/outliers removed")
                print("     Consider using more conservative settings for denser maps")
                
        else:
            print(f"\n✗ Failed to save dense map: {response.message}")
            sys.exit(1)
            
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Save dense PGO map with minimal filtering')
    
    # Mode selection
    parser.add_argument('--mode', choices=['ultra-dense', 'dense', 'balanced'],
                       default='dense', help='Preset density mode (default: dense)')
    parser.add_argument('--custom', action='store_true',
                       help='Use custom parameters instead of preset mode')
    
    # Required arguments
    parser.add_argument('--output', '-o', 
                       default="/home/surya/workspaces/slam_ws/src/FAST_LIO_LC/maps/dense_map.pcd",
                       help='Output file path')
    
    # File format options
    parser.add_argument('--format', '-f', default='pcd', choices=['pcd', 'ply'],
                       help='Output file format (default: pcd)')
    parser.add_argument('--binary', action='store_true', default=True,
                       help='Save as binary format')
    parser.add_argument('--intensity', action='store_true', default=True,
                       help='Include intensity values')
    
    # Custom filter options (used when --custom is specified)
    parser.add_argument('--outlier-removal', action='store_true', default=True,
                       help='Enable statistical outlier removal')
    parser.add_argument('--outlier-std-ratio', type=float, default=2.5,
                       help='Standard deviation ratio (higher = more permissive)')
    parser.add_argument('--outlier-neighbors', type=int, default=20,
                       help='Number of neighbors for analysis')

    parser.add_argument('--radius-filter', action='store_true', default=True,
                       help='Enable radius outlier removal')
    parser.add_argument('--radius-search', type=float, default=0.15,
                       help='Search radius in meters')
    parser.add_argument('--min-neighbors-radius', type=int, default=3,
                       help='Minimum neighbors within radius')
    
    parser.add_argument('--voxel-filter', action='store_true', default=True,
                       help='Enable voxel grid downsampling')
    parser.add_argument('--voxel-size', type=float, default=0.02,
                       help='Voxel grid leaf size in meters (smaller = denser)')

    # ROI options
    parser.add_argument('--use-roi', action='store_true',
                       help='Enable region of interest filtering')
    parser.add_argument('--roi-min', type=float, nargs=3, default=[-100, -100, -10],
                       metavar=('X', 'Y', 'Z'), help='ROI minimum bounds')
    parser.add_argument('--roi-max', type=float, nargs=3, default=[100, 100, 10],
                       metavar=('X', 'Y', 'Z'), help='ROI maximum bounds')
    
    args = parser.parse_args()
    
    # Add file extension if missing
    if '.' not in args.output:
        args.output += f'.{args.format}'
    
    save_dense_map(args)

if __name__ == '__main__':
    main()
