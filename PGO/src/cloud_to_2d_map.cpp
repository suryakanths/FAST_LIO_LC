/**
 * @file cloud_to_2d_map.cpp
 * @author AI Assistant
 * @brief Node that subscribes to /cloud_registered and generates a 2D occupancy grid map
 * @version 1.0
 * @date 2025-06-12
 * 
 * This node:
 * 1. Subscribes to /cloud_registered point cloud topic
 * 2. Filters points above 1.5m height
 * 3. Projects remaining points to 2D
 * 4. Generates an occupancy grid map
 * 5. Publishes the map as nav_msgs/OccupancyGrid
 */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/MapMetaData.h>
#include <geometry_msgs/Pose.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <cmath>
#include <vector>
#include <mutex>
#include <set>

class CloudTo2DMap
{
private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    
    // Subscribers and Publishers
    ros::Subscriber cloud_sub_;
    ros::Publisher map_pub_;
    ros::Publisher map_metadata_pub_;
    
    // TF listener for coordinate transformations
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    
    // Map parameters
    double map_resolution_;     // meters per pixel
    double max_height_;         // maximum height to consider (1.5m)
    double min_height_;         // minimum height to consider
    std::string map_frame_;     // frame for the map
    std::string cloud_frame_;   // frame of the input cloud
    
    // Dynamic map parameters
    double origin_x_, origin_y_;    // Current map origin in world coordinates
    int map_width_cells_, map_height_cells_;  // Current map size in cells
    double expansion_margin_;       // Margin to expand map when needed
    
    // Map data
    nav_msgs::OccupancyGrid occupancy_grid_;
    std::vector<std::vector<int>> hit_count_;
    std::vector<std::vector<int>> miss_count_;
    
    // Mutex for thread safety
    std::mutex map_mutex_;
    
    // Parameters for occupancy calculation
    double occupied_threshold_;
    double free_threshold_;
    int min_hit_count_;
    
public:
    CloudTo2DMap() : private_nh_("~"), tf_listener_(tf_buffer_)
    {
        // Load parameters
        loadParameters();
        
        // Initialize map
        initializeMap();
        
        // Setup subscribers and publishers
        cloud_sub_ = nh_.subscribe("/aft_pgo_map", 1, &CloudTo2DMap::cloudCallback, this);
        map_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("/map", 1, true);
        map_metadata_pub_ = nh_.advertise<nav_msgs::MapMetaData>("/map_metadata", 1, true);
        
        ROS_INFO("CloudTo2DMap node initialized");
        ROS_INFO("Dynamic map parameters: resolution=%.2fm, height_filter=[%.1f, %.1f]m, expansion_margin=%.1fm", 
                 map_resolution_, min_height_, max_height_, expansion_margin_);
    }
    
private:
    void loadParameters()
    {
        private_nh_.param<double>("map_resolution", map_resolution_, 0.05);  // High resolution for detail
        private_nh_.param<double>("max_height", max_height_, 2.0);   // Include tables, desks, etc.
        private_nh_.param<double>("min_height", min_height_, 0.1);   // Exclude ground plane
        private_nh_.param<std::string>("map_frame", map_frame_, "map");
        private_nh_.param<std::string>("cloud_frame", cloud_frame_, "");
        private_nh_.param<double>("occupied_threshold", occupied_threshold_, 0.6);
        private_nh_.param<double>("free_threshold", free_threshold_, 0.25);
        private_nh_.param<int>("min_hit_count", min_hit_count_, 2);
        private_nh_.param<double>("expansion_margin", expansion_margin_, 10.0);  // Expand when within 10m of edge
    }
    
    void initializeMap()
    {
        // Start with a small initial map (50x50 meters)
        double initial_size = 50.0;  // meters
        map_width_cells_ = static_cast<int>(initial_size / map_resolution_);
        map_height_cells_ = static_cast<int>(initial_size / map_resolution_);
        
        // Set initial origin at center of map
        origin_x_ = -initial_size / 2.0;
        origin_y_ = -initial_size / 2.0;
        
        // Initialize occupancy grid
        occupancy_grid_.header.frame_id = map_frame_;
        occupancy_grid_.info.resolution = map_resolution_;
        occupancy_grid_.info.width = map_width_cells_;
        occupancy_grid_.info.height = map_height_cells_;
        
        occupancy_grid_.info.origin.position.x = origin_x_;
        occupancy_grid_.info.origin.position.y = origin_y_;
        occupancy_grid_.info.origin.position.z = 0.0;
        occupancy_grid_.info.origin.orientation.w = 1.0;
        
        // Initialize map data
        occupancy_grid_.data.resize(map_width_cells_ * map_height_cells_, -1); // unknown
        
        // Initialize hit and miss counters
        hit_count_.resize(map_height_cells_, std::vector<int>(map_width_cells_, 0));
        miss_count_.resize(map_height_cells_, std::vector<int>(map_width_cells_, 0));
        
        ROS_INFO("Initialized dynamic map: %dx%d cells (%.1fx%.1fm), resolution=%.2fm", 
                 map_width_cells_, map_height_cells_, initial_size, initial_size, map_resolution_);
    }
    
    void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg)
    {
        try {
            // Convert ROS message to PCL point cloud
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
            pcl::fromROSMsg(*cloud_msg, *cloud);
            
            if (cloud->empty()) {
                ROS_WARN("Received empty point cloud");
                return;
            }
            
            // Get transform from cloud frame to map frame
            geometry_msgs::TransformStamped transform;
            std::string source_frame = cloud_frame_.empty() ? cloud_msg->header.frame_id : cloud_frame_;
            
            try {
                transform = tf_buffer_.lookupTransform(map_frame_, source_frame, 
                                                     cloud_msg->header.stamp, ros::Duration(0.1));
            } catch (tf2::TransformException& ex) {
                ROS_WARN("Could not transform from %s to %s: %s", 
                         source_frame.c_str(), map_frame_.c_str(), ex.what());
                return;
            }
            
            // Process point cloud
            processPointCloud(cloud, transform);
            
            // Update and publish map
            updateOccupancyGrid();
            publishMap(cloud_msg->header.stamp);
            
        } catch (const std::exception& e) {
            ROS_ERROR("Error processing point cloud: %s", e.what());
        }
    }
    
    bool needsExpansion(double x_map, double y_map) 
    {
        // Check if point is within expansion margin of map boundaries
        double map_min_x = origin_x_;
        double map_max_x = origin_x_ + (map_width_cells_ * map_resolution_);
        double map_min_y = origin_y_;
        double map_max_y = origin_y_ + (map_height_cells_ * map_resolution_);
        
        return (x_map < map_min_x + expansion_margin_ || 
                x_map > map_max_x - expansion_margin_ ||
                y_map < map_min_y + expansion_margin_ || 
                y_map > map_max_y - expansion_margin_);
    }
    
    void expandMap(double target_x, double target_y)
    {
        // Calculate how much to expand in each direction
        double current_min_x = origin_x_;
        double current_max_x = origin_x_ + (map_width_cells_ * map_resolution_);
        double current_min_y = origin_y_;
        double current_max_y = origin_y_ + (map_height_cells_ * map_resolution_);
        
        // Add expansion margin around the target point
        double new_min_x = std::min(current_min_x, target_x - expansion_margin_);
        double new_max_x = std::max(current_max_x, target_x + expansion_margin_);
        double new_min_y = std::min(current_min_y, target_y - expansion_margin_);
        double new_max_y = std::max(current_max_y, target_y + expansion_margin_);
        
        // Calculate new dimensions
        int new_width_cells = static_cast<int>((new_max_x - new_min_x) / map_resolution_);
        int new_height_cells = static_cast<int>((new_max_y - new_min_y) / map_resolution_);
        
        // Calculate offset for copying old data
        int offset_x = static_cast<int>((current_min_x - new_min_x) / map_resolution_);
        int offset_y = static_cast<int>((current_min_y - new_min_y) / map_resolution_);
        
        // Create new data structures
        std::vector<int8_t> new_data(new_width_cells * new_height_cells, -1);
        std::vector<std::vector<int>> new_hit_count(new_height_cells, std::vector<int>(new_width_cells, 0));
        std::vector<std::vector<int>> new_miss_count(new_height_cells, std::vector<int>(new_width_cells, 0));
        
        // Copy old data to new structure
        for (int old_y = 0; old_y < map_height_cells_; ++old_y) {
            for (int old_x = 0; old_x < map_width_cells_; ++old_x) {
                int new_x = old_x + offset_x;
                int new_y = old_y + offset_y;
                
                if (new_x >= 0 && new_x < new_width_cells && new_y >= 0 && new_y < new_height_cells) {
                    // Copy occupancy data
                    int old_index = old_y * map_width_cells_ + old_x;
                    int new_index = new_y * new_width_cells + new_x;
                    new_data[new_index] = occupancy_grid_.data[old_index];
                    
                    // Copy hit and miss counts
                    new_hit_count[new_y][new_x] = hit_count_[old_y][old_x];
                    new_miss_count[new_y][new_x] = miss_count_[old_y][old_x];
                }
            }
        }
        
        // Update map parameters
        origin_x_ = new_min_x;
        origin_y_ = new_min_y;
        map_width_cells_ = new_width_cells;
        map_height_cells_ = new_height_cells;
        
        // Update occupancy grid
        occupancy_grid_.info.width = new_width_cells;
        occupancy_grid_.info.height = new_height_cells;
        occupancy_grid_.info.origin.position.x = origin_x_;
        occupancy_grid_.info.origin.position.y = origin_y_;
        occupancy_grid_.data = std::move(new_data);
        
        // Update counters
        hit_count_ = std::move(new_hit_count);
        miss_count_ = std::move(new_miss_count);
        
        ROS_INFO("Expanded map to %dx%d cells (%.1fx%.1fm), new origin: (%.1f, %.1f)", 
                 new_width_cells, new_height_cells, 
                 new_width_cells * map_resolution_, new_height_cells * map_resolution_,
                 origin_x_, origin_y_);
    }
    
    void processPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                          const geometry_msgs::TransformStamped& transform)
    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        
        // Extract translation and rotation from transform
        double tx = transform.transform.translation.x;
        double ty = transform.transform.translation.y;
        double tz = transform.transform.translation.z;
        
        tf2::Quaternion q;
        tf2::fromMsg(transform.transform.rotation, q);
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        
        // Check if we need to expand the map based on sensor position
        if (needsExpansion(tx, ty)) {
            expandMap(tx, ty);
        }
        
        // Use a set to store unique occupied cells (top-down projection)
        std::set<std::pair<int, int>> occupied_cells_set;
        bool map_expanded = false;
        
        for (const auto& point : cloud->points) {
            // Skip invalid points
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
                continue;
            }
            
            // Transform point to map frame
            double x_map = tx + point.x * cos(yaw) - point.y * sin(yaw);
            double y_map = ty + point.x * sin(yaw) + point.y * cos(yaw);
            double z_map = tz + point.z;
            
            // Filter by height - if ANY point in height range exists, mark cell as occupied
            if (z_map < min_height_ || z_map > max_height_) {
                continue;
            }
            
            // Check if this point needs map expansion
            if (!map_expanded && needsExpansion(x_map, y_map)) {
                expandMap(x_map, y_map);
                map_expanded = true;
                // Clear the set since grid coordinates may have changed after expansion
                occupied_cells_set.clear();
            }
            
            // Convert to grid coordinates (top-down projection)
            int grid_x = static_cast<int>((x_map - origin_x_) / map_resolution_);
            int grid_y = static_cast<int>((y_map - origin_y_) / map_resolution_);
            
            // Check bounds
            if (grid_x >= 0 && grid_x < map_width_cells_ &&
                grid_y >= 0 && grid_y < map_height_cells_) {
                
                // Add to set (automatically handles duplicates - same cell occupied by multiple points)
                occupied_cells_set.insert({grid_x, grid_y});
            }
        }
        
        // Convert set to vector for further processing
        std::vector<std::pair<int, int>> occupied_cells(occupied_cells_set.begin(), occupied_cells_set.end());
        
        // Update hit counts for occupied cells
        for (const auto& cell : occupied_cells) {
            hit_count_[cell.second][cell.first]++;
        }
        
        // Ray-casting for free space (simplified - mark cells between sensor and obstacles as free)
        // Sensor position in grid coordinates
        int sensor_x = static_cast<int>((tx - origin_x_) / map_resolution_);
        int sensor_y = static_cast<int>((ty - origin_y_) / map_resolution_);
        
        if (sensor_x >= 0 && sensor_x < map_width_cells_ &&
            sensor_y >= 0 && sensor_y < map_height_cells_) {
            
            for (const auto& cell : occupied_cells) {
                markFreeCells(sensor_x, sensor_y, cell.first, cell.second);
            }
        }
    }
    
    void markFreeCells(int x0, int y0, int x1, int y1)
    {
        // Bresenham's line algorithm to mark free cells between sensor and obstacle
        int dx = abs(x1 - x0);
        int dy = abs(y1 - y0);
        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;
        int err = dx - dy;
        
        int x = x0, y = y0;
        
        while (true) {
            // Don't mark the final cell (obstacle) as free
            if (x == x1 && y == y1) break;
            
            // Mark current cell as potentially free
            if (x >= 0 && x < map_width_cells_ &&
                y >= 0 && y < map_height_cells_) {
                miss_count_[y][x]++;
            }
            
            if (x == x1 && y == y1) break;
            
            int e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x += sx;
            }
            if (e2 < dx) {
                err += dx;
                y += sy;
            }
        }
    }
    
    void updateOccupancyGrid()
    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        
        for (int y = 0; y < map_height_cells_; ++y) {
            for (int x = 0; x < map_width_cells_; ++x) {
                int index = y * map_width_cells_ + x;
                
                int hits = hit_count_[y][x];
                int misses = miss_count_[y][x];
                int total = hits + misses;
                
                if (hits >= min_hit_count_) {
                    // If we have enough hits, mark as occupied
                    // This ensures that any obstacle in the height range marks the cell as occupied
                    occupancy_grid_.data[index] = 100;
                } else if (total == 0) {
                    // Unknown - no data
                    occupancy_grid_.data[index] = -1;
                } else if (misses > hits * 3) {
                    // Significantly more misses than hits - likely free space
                    occupancy_grid_.data[index] = 0;
                } else {
                    // Uncertain - not enough evidence either way
                    occupancy_grid_.data[index] = -1;
                }
            }
        }
    }
    
    void publishMap(const ros::Time& stamp)
    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        
        occupancy_grid_.header.stamp = stamp;
        occupancy_grid_.info.map_load_time = stamp;
        
        map_pub_.publish(occupancy_grid_);
        map_metadata_pub_.publish(occupancy_grid_.info);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "cloud_to_2d_map");
    
    try {
        CloudTo2DMap mapper;
        
        ROS_INFO("CloudTo2DMap node started. Waiting for point clouds on /aft_pgo_map...");
        
        ros::spin();
        
    } catch (const std::exception& e) {
        ROS_ERROR("CloudTo2DMap node failed: %s", e.what());
        return 1;
    }
    
    return 0;
}
