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
    double map_width_;          // meters
    double map_height_;         // meters
    double max_height_;         // maximum height to consider (1.5m)
    double min_height_;         // minimum height to consider
    std::string map_frame_;     // frame for the map
    std::string cloud_frame_;   // frame of the input cloud
    
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
        cloud_sub_ = nh_.subscribe("/cloud_registered", 1, &CloudTo2DMap::cloudCallback, this);
        map_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("/map", 1, true);
        map_metadata_pub_ = nh_.advertise<nav_msgs::MapMetaData>("/map_metadata", 1, true);
        
        ROS_INFO("CloudTo2DMap node initialized");
        ROS_INFO("Map parameters: resolution=%.2fm, size=%.1fx%.1fm, height_filter=[%.1f, %.1f]m", 
                 map_resolution_, map_width_, map_height_, min_height_, max_height_);
    }
    
private:
    void loadParameters()
    {
        private_nh_.param<double>("map_resolution", map_resolution_, 0.1);
        private_nh_.param<double>("map_width", map_width_, 100.0);
        private_nh_.param<double>("map_height", map_height_, 100.0);
        private_nh_.param<double>("max_height", max_height_, 1.5);
        private_nh_.param<double>("min_height", min_height_, -0.5);
        private_nh_.param<std::string>("map_frame", map_frame_, "map");
        private_nh_.param<std::string>("cloud_frame", cloud_frame_, "");
        private_nh_.param<double>("occupied_threshold", occupied_threshold_, 0.6);
        private_nh_.param<double>("free_threshold", free_threshold_, 0.25);
        private_nh_.param<int>("min_hit_count", min_hit_count_, 3);
    }
    
    void initializeMap()
    {
        // Calculate map dimensions in cells
        int width_cells = static_cast<int>(map_width_ / map_resolution_);
        int height_cells = static_cast<int>(map_height_ / map_resolution_);
        
        // Initialize occupancy grid
        occupancy_grid_.header.frame_id = map_frame_;
        occupancy_grid_.info.resolution = map_resolution_;
        occupancy_grid_.info.width = width_cells;
        occupancy_grid_.info.height = height_cells;
        
        // Set origin at center of map
        occupancy_grid_.info.origin.position.x = -map_width_ / 2.0;
        occupancy_grid_.info.origin.position.y = -map_height_ / 2.0;
        occupancy_grid_.info.origin.position.z = 0.0;
        occupancy_grid_.info.origin.orientation.w = 1.0;
        
        // Initialize map data
        occupancy_grid_.data.resize(width_cells * height_cells, -1); // unknown
        
        // Initialize hit and miss counters
        hit_count_.resize(height_cells, std::vector<int>(width_cells, 0));
        miss_count_.resize(height_cells, std::vector<int>(width_cells, 0));
        
        ROS_INFO("Initialized map: %dx%d cells (%.1fx%.1fm)", 
                 width_cells, height_cells, map_width_, map_height_);
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
        
        // Transform and filter points
        std::vector<std::pair<int, int>> occupied_cells;
        
        for (const auto& point : cloud->points) {
            // Skip invalid points
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
                continue;
            }
            
            // Transform point to map frame
            double x_map = tx + point.x * cos(yaw) - point.y * sin(yaw);
            double y_map = ty + point.x * sin(yaw) + point.y * cos(yaw);
            double z_map = tz + point.z;
            
            // Filter by height
            if (z_map < min_height_ || z_map > max_height_) {
                continue;
            }
            
            // Convert to grid coordinates
            int grid_x = static_cast<int>((x_map - occupancy_grid_.info.origin.position.x) / map_resolution_);
            int grid_y = static_cast<int>((y_map - occupancy_grid_.info.origin.position.y) / map_resolution_);
            
            // Check bounds
            if (grid_x >= 0 && grid_x < occupancy_grid_.info.width &&
                grid_y >= 0 && grid_y < occupancy_grid_.info.height) {
                
                occupied_cells.push_back({grid_x, grid_y});
            }
        }
        
        // Update hit counts for occupied cells
        for (const auto& cell : occupied_cells) {
            hit_count_[cell.second][cell.first]++;
        }
        
        // Ray-casting for free space (simplified - mark cells between sensor and obstacles as free)
        // Sensor position in grid coordinates
        int sensor_x = static_cast<int>((tx - occupancy_grid_.info.origin.position.x) / map_resolution_);
        int sensor_y = static_cast<int>((ty - occupancy_grid_.info.origin.position.y) / map_resolution_);
        
        if (sensor_x >= 0 && sensor_x < occupancy_grid_.info.width &&
            sensor_y >= 0 && sensor_y < occupancy_grid_.info.height) {
            
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
            if (x >= 0 && x < occupancy_grid_.info.width &&
                y >= 0 && y < occupancy_grid_.info.height) {
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
        
        for (int y = 0; y < occupancy_grid_.info.height; ++y) {
            for (int x = 0; x < occupancy_grid_.info.width; ++x) {
                int index = y * occupancy_grid_.info.width + x;
                
                int hits = hit_count_[y][x];
                int misses = miss_count_[y][x];
                int total = hits + misses;
                
                if (total == 0) {
                    // Unknown
                    occupancy_grid_.data[index] = -1;
                } else if (hits >= min_hit_count_) {
                    double occupancy_prob = static_cast<double>(hits) / total;
                    
                    if (occupancy_prob >= occupied_threshold_) {
                        // Occupied
                        occupancy_grid_.data[index] = 100;
                    } else if (occupancy_prob <= free_threshold_) {
                        // Free
                        occupancy_grid_.data[index] = 0;
                    } else {
                        // Unknown
                        occupancy_grid_.data[index] = -1;
                    }
                } else if (misses > hits * 2) {
                    // Likely free
                    occupancy_grid_.data[index] = 0;
                } else {
                    // Unknown
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
        
        ROS_INFO("CloudTo2DMap node started. Waiting for point clouds on /cloud_registered...");
        
        ros::spin();
        
    } catch (const std::exception& e) {
        ROS_ERROR("CloudTo2DMap node failed: %s", e.what());
        return 1;
    }
    
    return 0;
}
