/**
 * @file laserPosegraphOptimization.cpp
 * @author Yanliang Wang (wyl410922@qq.com)
 * @brief 
 * 1. Detect the keyframes
 * 2. Maintain the Gtsam-based pose graph
 * 3. Detect the radius-search-based loop closure, and add them to the pose graph
 * @version 0.1
 * @date 2022-03-11
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>
#include <cmath>
#include <chrono>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

#include <Eigen/Dense>

#include <ceres/ceres.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

#include "scancontext/Scancontext.h"

// CUDA acceleration for PGO
#include "pgo_cuda_utils.h"

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

using namespace gtsam;

using std::cout;
using std::endl;

double keyframeMeterGap;
double keyframeDegGap, keyframeRadGap;
double translationAccumulated = 1000000.0; // large value means must add the first given frame.
double rotaionAccumulated = 1000000.0; // large value means must add the first given frame.

bool isNowKeyFrame = false; 

Pose6D odom_pose_prev {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init 
Pose6D odom_pose_curr {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init pose is zero 

std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<sensor_msgs::NavSatFix::ConstPtr> gpsBuf;
std::queue<std::pair<int, int> > scLoopICPBuf;

std::mutex mBuf;
std::mutex mKF;

double timeLaserOdometry = 0.0;
double timeLaser = 0.0;

pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudMapAfterPGO(new pcl::PointCloud<PointType>());

std::vector<pcl::PointCloud<PointType>::Ptr> keyframeLaserClouds; 
std::vector<Pose6D> keyframePoses;
std::vector<Pose6D> keyframePosesUpdated;
std::vector<double> keyframeTimes;
int recentIdxUpdated = 0;
// for loop closure detection
std::map<int, int> loopIndexContainer; // 记录存在的回环对
pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtreeHistoryKeyPoses(new pcl::KdTreeFLANN<pcl::PointXYZ>());
ros::Publisher pubLoopConstraintEdge;


gtsam::NonlinearFactorGraph gtSAMgraph;
bool gtSAMgraphMade = false;
gtsam::Values initialEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;

noiseModel::Diagonal::shared_ptr priorNoise;
noiseModel::Diagonal::shared_ptr odomNoise;
noiseModel::Base::shared_ptr robustLoopNoise;
noiseModel::Base::shared_ptr robustGPSNoise;

pcl::VoxelGrid<PointType> downSizeFilterScancontext;
SCManager scManager;
double scDistThres, scMaximumRadius;

// CUDA acceleration
pgo_cuda::CudaPGOProcessor cuda_processor;
bool cuda_available = false;

// CPU optimization variables - AGGRESSIVE SETTINGS
static int optimization_skip_counter = 0;
static double last_optimization_time = 0.0;
static const double MIN_OPTIMIZATION_INTERVAL = 5.0; // AGGRESSIVE: Minimum 5 seconds between optimizations
static const int OPTIMIZATION_SKIP_FACTOR = 10; // AGGRESSIVE: Skip optimization more frequently  
static int loop_closure_count = 0;
static bool force_next_optimization = false;

// Emergency CPU throttling
static const double EMERGENCY_CPU_THRESHOLD = 600.0; // 600% CPU usage threshold
static bool emergency_throttle_mode = false;
static double last_cpu_check = 0.0;
static const double CPU_CHECK_INTERVAL = 10.0; // Check CPU every 10 seconds

// Performance monitoring
static auto last_performance_print = std::chrono::steady_clock::now();
static int total_keyframes_processed = 0;
static int optimizations_performed = 0;

pcl::VoxelGrid<PointType> downSizeFilterICP;
std::mutex mtxICP;
std::mutex mtxPosegraph;
std::mutex mtxRecentPose;

pcl::PointCloud<PointType>::Ptr laserCloudMapPGO(new pcl::PointCloud<PointType>());
pcl::VoxelGrid<PointType> downSizeFilterMapPGO;
bool laserCloudMapPGORedraw = true;

bool useGPS = true;
// bool useGPS = false;
sensor_msgs::NavSatFix::ConstPtr currGPS;
bool hasGPSforThisKF = false;
bool gpsOffsetInitialized = false; 
double gpsAltitudeInitOffset = 0.0;
double recentOptimizedX = 0.0;
double recentOptimizedY = 0.0;

ros::Publisher pubMapAftPGO, pubOdomAftPGO, pubPathAftPGO;
ros::Publisher pubLoopScanLocal, pubLoopSubmapLocal;
ros::Publisher pubOdomRepubVerifier;

std::string save_directory;
std::string pgKITTIformat, pgScansDirectory;
std::string odomKITTIformat;
std::fstream pgTimeSaveStream;

// for front_end
ros::Publisher pubKeyFramesId;

// for loop closure
double historyKeyframeSearchRadius;
double historyKeyframeSearchTimeDiff;
int historyKeyframeSearchNum;
double loopClosureFrequency;
int graphUpdateTimes;
double graphUpdateFrequency;
double loopNoiseScore;
double vizmapFrequency;
double vizPathFrequency;
double speedFactor;
ros::Publisher pubLoopScanLocalRegisted;
double loopFitnessScoreThreshold;
double mapVizFilterSize;

std::string padZeros(int val, int num_digits = 6) {
  std::ostringstream out;
  out << std::internal << std::setfill('0') << std::setw(num_digits) << val;
  return out.str();
}

gtsam::Pose3 Pose6DtoGTSAMPose3(const Pose6D& p)
{
    return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
} // Pose6DtoGTSAMPose3

void saveOdometryVerticesKITTIformat(std::string _filename)
{
    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(_filename.c_str(), std::fstream::out);
    for(const auto& _pose6d: keyframePoses) {
        gtsam::Pose3 pose = Pose6DtoGTSAMPose3(_pose6d);
        Point3 t = pose.translation();
        Rot3 R = pose.rotation();
        auto col1 = R.column(1); // Point3
        auto col2 = R.column(2); // Point3
        auto col3 = R.column(3); // Point3

        stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
    }
}

void saveOptimizedVerticesKITTIformat(gtsam::Values _estimates, std::string _filename)
{
    using namespace gtsam;

    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(_filename.c_str(), std::fstream::out);

    for(const auto& key_value: _estimates) {
        auto p = dynamic_cast<const GenericValue<Pose3>*>(&key_value.value);
        if (!p) continue;

        const Pose3& pose = p->value();

        Point3 t = pose.translation();
        Rot3 R = pose.rotation();
        auto col1 = R.column(1); // Point3
        auto col2 = R.column(2); // Point3
        auto col3 = R.column(3); // Point3

        stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
    }
}

void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &_laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(_laserOdometry);
	mBuf.unlock();
} // laserOdometryHandler

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &_laserCloudFullRes)
{
	mBuf.lock();
	fullResBuf.push(_laserCloudFullRes);
	mBuf.unlock();
} // laserCloudFullResHandler

void gpsHandler(const sensor_msgs::NavSatFix::ConstPtr &_gps)
{
    if(useGPS) {
        mBuf.lock();
        gpsBuf.push(_gps);
        mBuf.unlock();
    }
} // gpsHandler

void initNoises( void )
{
    gtsam::Vector priorNoiseVector6(6);
    priorNoiseVector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
    priorNoise = noiseModel::Diagonal::Variances(priorNoiseVector6);

    gtsam::Vector odomNoiseVector6(6);
    // odomNoiseVector6 << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
    odomNoiseVector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
    odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);

    // double loopNoiseScore = 0.5; // constant is ok...
    gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
    robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
    robustLoopNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6) );

    double bigNoiseTolerentToXY = 1000000000.0; // 1e9
    double gpsAltitudeNoiseScore = 250.0; // if height is misaligned after loop clsosing, use this value bigger
    gtsam::Vector robustNoiseVector3(3); // gps factor has 3 elements (xyz)
    robustNoiseVector3 << bigNoiseTolerentToXY, bigNoiseTolerentToXY, gpsAltitudeNoiseScore; // means only caring altitude here. (because LOAM-like-methods tends to be asymptotically flyging)
    robustGPSNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector3) );

} // initNoises

Pose6D getOdom(nav_msgs::Odometry::ConstPtr _odom)
{
    auto tx = _odom->pose.pose.position.x;
    auto ty = _odom->pose.pose.position.y;
    auto tz = _odom->pose.pose.position.z;

    double roll, pitch, yaw;
    geometry_msgs::Quaternion quat = _odom->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(quat.x, quat.y, quat.z, quat.w)).getRPY(roll, pitch, yaw);

    // Validate pose components
    if (!std::isfinite(tx) || !std::isfinite(ty) || !std::isfinite(tz) ||
        !std::isfinite(roll) || !std::isfinite(pitch) || !std::isfinite(yaw)) {
        std::cout << "[ERROR] Invalid odometry data detected: pos=(" 
                  << tx << ", " << ty << ", " << tz << "), rpy=(" 
                  << roll << ", " << pitch << ", " << yaw << ")" << std::endl;
        // Return a default/previous valid pose instead of invalid data
        // For now, return zero pose but in production you might want to use last valid pose
        return Pose6D{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, _odom->header.seq};
    }

    return Pose6D{tx, ty, tz, roll, pitch, yaw, _odom->header.seq}; 
} // getOdom

Pose6D diffTransformation(const Pose6D& _p1, const Pose6D& _p2)
{
    Eigen::Affine3f SE3_p1 = pcl::getTransformation(_p1.x, _p1.y, _p1.z, _p1.roll, _p1.pitch, _p1.yaw);
    Eigen::Affine3f SE3_p2 = pcl::getTransformation(_p2.x, _p2.y, _p2.z, _p2.roll, _p2.pitch, _p2.yaw);
    Eigen::Matrix4f SE3_delta0 = SE3_p1.matrix().inverse() * SE3_p2.matrix();
    Eigen::Affine3f SE3_delta; SE3_delta.matrix() = SE3_delta0;
    float dx, dy, dz, droll, dpitch, dyaw;
    pcl::getTranslationAndEulerAngles (SE3_delta, dx, dy, dz, droll, dpitch, dyaw);
    // std::cout << "delta : " << dx << ", " << dy << ", " << dz << ", " << droll << ", " << dpitch << ", " << dyaw << std::endl;

    return Pose6D{double(abs(dx)), double(abs(dy)), double(abs(dz)), double(abs(droll)), double(abs(dpitch)), double(abs(dyaw))};
} // SE3Diff

pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose6D& tf)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    // Validate transformation pose
    if (!std::isfinite(tf.x) || !std::isfinite(tf.y) || !std::isfinite(tf.z) ||
        !std::isfinite(tf.roll) || !std::isfinite(tf.pitch) || !std::isfinite(tf.yaw)) {
        std::cout << "[ERROR] Invalid transformation pose in local2global: pos=(" 
                  << tf.x << ", " << tf.y << ", " << tf.z << "), rpy=(" 
                  << tf.roll << ", " << tf.pitch << ", " << tf.yaw << ")" << std::endl;
        return cloudOut; // Return empty cloud
    }

    int cloudSize = cloudIn->size();
    cloudOut->reserve(cloudSize); // Reserve space but don't resize yet

    Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);
    
    int numberOfCores = 16;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        
        // Validate input point coordinates
        if (!std::isfinite(pointFrom.x) || !std::isfinite(pointFrom.y) || !std::isfinite(pointFrom.z)) {
            continue; // Skip invalid points
        }
        
        PointType pointTo;
        pointTo.x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        pointTo.y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        pointTo.z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        pointTo.intensity = pointFrom.intensity;
        
        // Validate transformed point coordinates
        if (std::isfinite(pointTo.x) && std::isfinite(pointTo.y) && std::isfinite(pointTo.z)) {
            #pragma omp critical
            {
                cloudOut->points.push_back(pointTo);
            }
        }
    }

    cloudOut->width = cloudOut->points.size();
    cloudOut->height = 1;
    cloudOut->is_dense = true;

    return cloudOut;
}

void pubPath( void )
{
    // pub odom and path 
    nav_msgs::Odometry odomAftPGO;
    nav_msgs::Path pathAftPGO;
    pathAftPGO.header.frame_id = "camera_init";
    
    // Initialize with a valid default pose
    odomAftPGO.header.frame_id = "camera_init";
    odomAftPGO.child_frame_id = "/aft_pgo";
    odomAftPGO.header.stamp = ros::Time::now();
    odomAftPGO.pose.pose.position.x = 0.0;
    odomAftPGO.pose.pose.position.y = 0.0;
    odomAftPGO.pose.pose.position.z = 0.0;
    odomAftPGO.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(0.0, 0.0, 0.0);
    
    mKF.lock(); 
    // for (int node_idx=0; node_idx < int(keyframePosesUpdated.size()) - 1; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    {
        const Pose6D& pose_est = keyframePosesUpdated.at(node_idx); // upodated poses
        // const gtsam::Pose3& pose_est = isamCurrentEstimate.at<gtsam::Pose3>(node_idx);

        // Validate pose before publishing
        if (!std::isfinite(pose_est.x) || !std::isfinite(pose_est.y) || !std::isfinite(pose_est.z) ||
            !std::isfinite(pose_est.roll) || !std::isfinite(pose_est.pitch) || !std::isfinite(pose_est.yaw)) {
            std::cout << "[WARNING] Invalid pose at node " << node_idx 
                      << ": pos=(" << pose_est.x << ", " << pose_est.y << ", " << pose_est.z 
                      << "), rpy=(" << pose_est.roll << ", " << pose_est.pitch << ", " << pose_est.yaw 
                      << "). Skipping this pose in path." << std::endl;
            continue;
        }

        nav_msgs::Odometry odomAftPGOthis;
        odomAftPGOthis.header.frame_id = "camera_init";
        odomAftPGOthis.child_frame_id = "/aft_pgo";
        odomAftPGOthis.header.stamp = ros::Time::now(); // Use current time to avoid TF_OLD_DATA warnings
        odomAftPGOthis.pose.pose.position.x = pose_est.x;
        odomAftPGOthis.pose.pose.position.y = pose_est.y;
        odomAftPGOthis.pose.pose.position.z = pose_est.z;
        odomAftPGOthis.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(pose_est.roll, pose_est.pitch, pose_est.yaw);
        odomAftPGO = odomAftPGOthis;

        geometry_msgs::PoseStamped poseStampAftPGO;
        poseStampAftPGO.header = odomAftPGOthis.header;
        poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

        pathAftPGO.header.stamp = odomAftPGOthis.header.stamp;
        pathAftPGO.header.frame_id = "camera_init";
        pathAftPGO.poses.push_back(poseStampAftPGO);
    }
    mKF.unlock(); 
    
    // Validate final pose before publishing transform
    bool validFinalPose = std::isfinite(odomAftPGO.pose.pose.position.x) && 
                         std::isfinite(odomAftPGO.pose.pose.position.y) && 
                         std::isfinite(odomAftPGO.pose.pose.position.z) &&
                         std::isfinite(odomAftPGO.pose.pose.orientation.x) &&
                         std::isfinite(odomAftPGO.pose.pose.orientation.y) &&
                         std::isfinite(odomAftPGO.pose.pose.orientation.z) &&
                         std::isfinite(odomAftPGO.pose.pose.orientation.w);
    
    if (validFinalPose) {
        pubOdomAftPGO.publish(odomAftPGO); // last pose 
        pubPathAftPGO.publish(pathAftPGO); // poses 

        static tf::TransformBroadcaster br;
        tf::Transform transform;
        tf::Quaternion q;
        transform.setOrigin(tf::Vector3(odomAftPGO.pose.pose.position.x, odomAftPGO.pose.pose.position.y, odomAftPGO.pose.pose.position.z));
        q.setW(odomAftPGO.pose.pose.orientation.w);
        q.setX(odomAftPGO.pose.pose.orientation.x);
        q.setY(odomAftPGO.pose.pose.orientation.y);
        q.setZ(odomAftPGO.pose.pose.orientation.z);
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, odomAftPGO.header.stamp, "camera_init", "/aft_pgo"));
    } else {
        std::cout << "[ERROR] Final pose contains invalid values. Skipping transform publication." << std::endl;
        std::cout << "  Position: (" << odomAftPGO.pose.pose.position.x 
                  << ", " << odomAftPGO.pose.pose.position.y 
                  << ", " << odomAftPGO.pose.pose.position.z << ")" << std::endl;
        std::cout << "  Orientation: (" << odomAftPGO.pose.pose.orientation.x 
                  << ", " << odomAftPGO.pose.pose.orientation.y 
                  << ", " << odomAftPGO.pose.pose.orientation.z 
                  << ", " << odomAftPGO.pose.pose.orientation.w << ")" << std::endl;
    }
} // pubPath

void updatePoses(void)
{
    mKF.lock(); 
    for (int node_idx=0; node_idx < int(isamCurrentEstimate.size()); node_idx++)
    {
        Pose6D& p =keyframePosesUpdated[node_idx];
        double new_x = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().x();
        double new_y = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().y();
        double new_z = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().z();
        double new_roll = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().roll();
        double new_pitch = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().pitch();
        double new_yaw = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().yaw();
        
        // Validate updated pose values
        if (std::isfinite(new_x) && std::isfinite(new_y) && std::isfinite(new_z) &&
            std::isfinite(new_roll) && std::isfinite(new_pitch) && std::isfinite(new_yaw)) {
            p.x = new_x;
            p.y = new_y;
            p.z = new_z;
            p.roll = new_roll;
            p.pitch = new_pitch;
            p.yaw = new_yaw;
        } else {
            std::cout << "[ERROR] Invalid optimized pose for node " << node_idx 
                      << ": pos=(" << new_x << ", " << new_y << ", " << new_z 
                      << "), rpy=(" << new_roll << ", " << new_pitch << ", " << new_yaw 
                      << "). Keeping previous values." << std::endl;
        }
    }
    mKF.unlock();

    mtxRecentPose.lock();
    const gtsam::Pose3& lastOptimizedPose = isamCurrentEstimate.at<gtsam::Pose3>(int(isamCurrentEstimate.size())-1);
    double opt_x = lastOptimizedPose.translation().x();
    double opt_y = lastOptimizedPose.translation().y();
    
    // Validate recent optimized pose
    if (std::isfinite(opt_x) && std::isfinite(opt_y)) {
        recentOptimizedX = opt_x;
        recentOptimizedY = opt_y;
    } else {
        std::cout << "[ERROR] Invalid recent optimized pose: (" << opt_x << ", " << opt_y 
                  << "). Keeping previous values." << std::endl;
    }

    recentIdxUpdated = int(keyframePosesUpdated.size()) - 1;

    mtxRecentPose.unlock();
} // updatePoses

void runISAM2opt(void)
{
    // CPU optimization: Check for emergency throttling
    double current_time = ros::Time::now().toSec();
    if (current_time - last_cpu_check > CPU_CHECK_INTERVAL) {
        last_cpu_check = current_time;
        
        // Simple CPU check - if we're in emergency mode, be extra conservative
        if (emergency_throttle_mode) {
            optimization_skip_counter++;
            if (optimization_skip_counter < OPTIMIZATION_SKIP_FACTOR * 2) { // Double the skip factor in emergency
                std::cout << "[EMERGENCY] CPU throttling - skipping optimization (" 
                          << optimization_skip_counter << ")" << std::endl;
                return;
            }
        }
    }
    
    // CPU optimization: Skip optimization if called too frequently
    optimization_skip_counter++;
    
    // Skip optimization if:
    // 1. Not enough time has passed since last optimization AND
    // 2. No new loop closures have been detected AND  
    // 3. Skip counter hasn't exceeded the threshold
    bool should_skip = (current_time - last_optimization_time < MIN_OPTIMIZATION_INTERVAL) && 
                      !force_next_optimization && 
                      (optimization_skip_counter < OPTIMIZATION_SKIP_FACTOR);
    
    if (should_skip) {
        std::cout << "[CPU OPT] Skipping optimization (counter: " << optimization_skip_counter 
                  << ", time since last: " << (current_time - last_optimization_time) << "s)" << std::endl;
        return;
    }
    
    // Reset counters and perform optimization
    optimization_skip_counter = 0;
    last_optimization_time = current_time;
    force_next_optimization = false;
    optimizations_performed++;
    
    std::cout << "[CPU OPT] Performing optimization #" << optimizations_performed 
              << " (keyframes: " << total_keyframes_processed 
              << ", loop closures: " << loop_closure_count << ")" << std::endl;
    
    // Adaptive optimization: Reduce number of iterations based on graph size
    int adaptive_update_times = graphUpdateTimes;
    if (keyframePoses.size() > 50) {
        adaptive_update_times = std::max(1, static_cast<int>(graphUpdateTimes) / 2); // Reduce iterations for medium graphs
    }
    if (keyframePoses.size() > 100) {
        adaptive_update_times = 1; // Minimal iterations for large graphs
    }
    if (emergency_throttle_mode) {
        adaptive_update_times = 1; // Always minimal in emergency mode
    }
    
    auto opt_start = std::chrono::high_resolution_clock::now();
    
    // called when a variable added 
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    
    // Adaptive optimization iterations
    for(int i = adaptive_update_times; i > 0; --i){
        isam->update();
    }
    
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    isamCurrentEstimate = isam->calculateEstimate();
    updatePoses();
    pubPath();  // 每优化一次就输出一次优化后的位姿
    
    auto opt_end = std::chrono::high_resolution_clock::now();
    auto opt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(opt_end - opt_start);
    
    std::cout << "[CPU OPT] Optimization completed in " << opt_duration.count() 
              << "ms with " << adaptive_update_times << " iterations" << std::endl;
    
    // Print performance statistics every 30 seconds
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_performance_print).count() >= 30) {
        std::cout << "[PERFORMANCE] Stats - Keyframes: " << total_keyframes_processed 
                  << ", Optimizations: " << optimizations_performed 
                  << ", Loop closures: " << loop_closure_count 
                  << ", CUDA ops: " << (cuda_available ? "available" : "not available") 
                  << ", Emergency mode: " << (emergency_throttle_mode ? "ON" : "OFF") << std::endl;
        last_performance_print = now;
    }
}

pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, gtsam::Pose3 transformIn)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    // Validate transformation
    auto translation = transformIn.translation();
    auto rotation = transformIn.rotation();
    if (!std::isfinite(translation.x()) || !std::isfinite(translation.y()) || !std::isfinite(translation.z()) ||
        !std::isfinite(rotation.roll()) || !std::isfinite(rotation.pitch()) || !std::isfinite(rotation.yaw())) {
        std::cout << "[ERROR] Invalid gtsam::Pose3 transformation in transformPointCloud" << std::endl;
        return cloudOut; // Return empty cloud
    }

    PointType *pointFrom;

    int cloudSize = cloudIn->size();
    cloudOut->reserve(cloudSize); // Reserve space but don't resize yet

    Eigen::Affine3f transCur = pcl::getTransformation(
                                    transformIn.translation().x(), transformIn.translation().y(), transformIn.translation().z(), 
                                    transformIn.rotation().roll(), transformIn.rotation().pitch(), transformIn.rotation().yaw() );
    
    int numberOfCores = 8; // TODO move to yaml 
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        pointFrom = &cloudIn->points[i];
        
        // Validate input point coordinates
        if (!std::isfinite(pointFrom->x) || !std::isfinite(pointFrom->y) || !std::isfinite(pointFrom->z)) {
            continue; // Skip invalid points
        }
        
        PointType pointTo;
        pointTo.x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
        pointTo.y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
        pointTo.z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
        pointTo.intensity = pointFrom->intensity;
        
        // Validate transformed point coordinates
        if (std::isfinite(pointTo.x) && std::isfinite(pointTo.y) && std::isfinite(pointTo.z)) {
            #pragma omp critical
            {
                cloudOut->points.push_back(pointTo);
            }
        }
    }
    
    cloudOut->width = cloudOut->points.size();
    cloudOut->height = 1;
    cloudOut->is_dense = true;
    
    return cloudOut;
} // transformPointCloud

void loopFindNearKeyframesCloud( pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& submap_size, const int& root_idx)
{
    // extract and stacking near keyframes (in global coord)
    nearKeyframes->clear();
    for (int i = -submap_size; i <= submap_size; ++i) {
        int keyNear = root_idx + i;
        if (keyNear < 0 || keyNear >= int(keyframeLaserClouds.size()) )
            continue;

        mKF.lock(); 
        *nearKeyframes += * local2global(keyframeLaserClouds[keyNear], keyframePosesUpdated[keyNear]);
        mKF.unlock(); 
    }

    if (nearKeyframes->empty())
        return;

    // downsample near keyframes with CUDA acceleration
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    
    bool cuda_success = false;
    if (cuda_available) {
        cuda_success = cuda_processor.DownsamplePointCloud(nearKeyframes, cloud_temp, 0.4f);
    }
    
    if (!cuda_success) {
        // Fallback to CPU implementation
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
    }
    
    *nearKeyframes = *cloud_temp;
} // loopFindNearKeyframesCloud

/**
 * 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合，降采样
*/
void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
{
    // 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合
    nearKeyframes->clear();
    int cloudSize = keyframeLaserClouds.size();
    for (int i = -searchNum; i <= searchNum; ++i)
    {
        int keyNear = key + i;
        if (keyNear < 0 || keyNear >= cloudSize )
            continue;
        // *nearKeyframes += *transformPointCloud(keyframeLaserClouds[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
        mKF.lock(); 
        *nearKeyframes += * local2global(keyframeLaserClouds[keyNear], keyframePosesUpdated[keyNear]);
        mKF.unlock();
    }

    if (nearKeyframes->empty())
        return;

    // 降采样 with CUDA acceleration
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    
    bool cuda_success = false;
    if (cuda_available) {
        cuda_success = cuda_processor.DownsamplePointCloud(nearKeyframes, cloud_temp, 0.4f);
    }
    
    if (!cuda_success) {
        // Fallback to CPU implementation
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
    }
    
    *nearKeyframes = *cloud_temp;
}

/**
 * Eigen格式的位姿变换
*/
Eigen::Affine3f Pose6dToAffine3f(Pose6D pose)
{ 
    // Validate pose values before transformation
    if (!std::isfinite(pose.x) || !std::isfinite(pose.y) || !std::isfinite(pose.z) ||
        !std::isfinite(pose.roll) || !std::isfinite(pose.pitch) || !std::isfinite(pose.yaw)) {
        std::cout << "[ERROR] Invalid Pose6D in Pose6dToAffine3f: pos=(" 
                  << pose.x << ", " << pose.y << ", " << pose.z << "), rpy=(" 
                  << pose.roll << ", " << pose.pitch << ", " << pose.yaw << ")" << std::endl;
        return Eigen::Affine3f::Identity(); // Return identity transformation
    }
    return pcl::getTransformation(pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw);
}

/**
 * 位姿格式变换
*/
gtsam::Pose3 Pose6dTogtsamPose3(Pose6D pose)
{
    // Validate pose values before transformation
    if (!std::isfinite(pose.x) || !std::isfinite(pose.y) || !std::isfinite(pose.z) ||
        !std::isfinite(pose.roll) || !std::isfinite(pose.pitch) || !std::isfinite(pose.yaw)) {
        std::cout << "[ERROR] Invalid Pose6D in Pose6dTogtsamPose3: pos=(" 
                  << pose.x << ", " << pose.y << ", " << pose.z << "), rpy=(" 
                  << pose.roll << ", " << pose.pitch << ", " << pose.yaw << ")" << std::endl;
        return gtsam::Pose3(); // Return identity pose
    }
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(pose.roll), double(pose.pitch), double(pose.yaw)),
                                gtsam::Point3(double(pose.x),    double(pose.y),     double(pose.z)));
}

gtsam::Pose3 doICPVirtualRelative( int _loop_kf_idx, int _curr_kf_idx )
{
    // parse pointclouds
    // int historyKeyframeSearchNum = 25; // enough. ex. [-25, 25] covers submap length of 50x1 = 50m if every kf gap is 1m
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());
    // loopFindNearKeyframesCloud(cureKeyframeCloud, _curr_kf_idx, 0, _loop_kf_idx); // use same root of loop kf idx 
    // loopFindNearKeyframesCloud(targetKeyframeCloud, _loop_kf_idx, historyKeyframeSearchNum, _loop_kf_idx); 
    // 提取当前关键帧特征点集合，降采样
    loopFindNearKeyframes(cureKeyframeCloud, _curr_kf_idx, 0);
    // 提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
    loopFindNearKeyframes(targetKeyframeCloud, _loop_kf_idx, historyKeyframeSearchNum);

    // Validate point clouds before ICP operations
    if (cureKeyframeCloud->empty() || targetKeyframeCloud->empty()) {
        std::cout << "[ERROR] Empty point clouds in doICPVirtualRelative: source=" 
                  << cureKeyframeCloud->size() << ", target=" << targetKeyframeCloud->size() << std::endl;
        return gtsam::Pose3();
    }

    // Filter out invalid points from source cloud
    pcl::PointCloud<PointType>::Ptr validCureCloud(new pcl::PointCloud<PointType>());
    validCureCloud->reserve(cureKeyframeCloud->size());
    for (const auto& point : cureKeyframeCloud->points) {
        if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
            validCureCloud->points.push_back(point);
        }
    }
    validCureCloud->width = validCureCloud->points.size();
    validCureCloud->height = 1;
    validCureCloud->is_dense = true;

    // Filter out invalid points from target cloud
    pcl::PointCloud<PointType>::Ptr validTargetCloud(new pcl::PointCloud<PointType>());
    validTargetCloud->reserve(targetKeyframeCloud->size());
    for (const auto& point : targetKeyframeCloud->points) {
        if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
            validTargetCloud->points.push_back(point);
        }
    }
    validTargetCloud->width = validTargetCloud->points.size();
    validTargetCloud->height = 1;
    validTargetCloud->is_dense = true;

    // Check if we have enough valid points for ICP
    if (validCureCloud->empty() || validTargetCloud->empty() ||
        validCureCloud->size() < 100 || validTargetCloud->size() < 100) {
        std::cout << "[ERROR] Insufficient valid points for ICP: source=" 
                  << validCureCloud->size() << ", target=" << validTargetCloud->size() 
                  << " (original: " << cureKeyframeCloud->size() << ", " << targetKeyframeCloud->size() << ")" << std::endl;
        return gtsam::Pose3();
    }

    std::cout << "[INFO] ICP with validated point clouds: source=" 
              << validCureCloud->size() << "/" << cureKeyframeCloud->size() 
              << ", target=" << validTargetCloud->size() << "/" << targetKeyframeCloud->size() << std::endl;

    // Use validated point clouds for visualization
    cureKeyframeCloud = validCureCloud;
    targetKeyframeCloud = validTargetCloud;

    // loop verification 
    sensor_msgs::PointCloud2 cureKeyframeCloudMsg;
    pcl::toROSMsg(*cureKeyframeCloud, cureKeyframeCloudMsg);
    cureKeyframeCloudMsg.header.frame_id = "camera_init";
    pubLoopScanLocal.publish(cureKeyframeCloudMsg);

    sensor_msgs::PointCloud2 targetKeyframeCloudMsg;
    pcl::toROSMsg(*targetKeyframeCloud, targetKeyframeCloudMsg);
    targetKeyframeCloudMsg.header.frame_id = "camera_init";
    pubLoopSubmapLocal.publish(targetKeyframeCloudMsg);

    // ICP Settings
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter 
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align pointclouds using validated clouds
    icp.setInputSource(cureKeyframeCloud);  // These are now the validated clouds
    icp.setInputTarget(targetKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    
    try {
        icp.align(*unused_result);
    } catch (const std::exception& e) {
        std::cout << "[ERROR] ICP alignment failed with exception: " << e.what() << std::endl;
        return gtsam::Pose3();
    }

    sensor_msgs::PointCloud2 cureKeyframeCloudRegMsg;
    pcl::toROSMsg(*unused_result, cureKeyframeCloudRegMsg);
    cureKeyframeCloudRegMsg.header.frame_id = "camera_init";
    pubLoopScanLocalRegisted.publish(cureKeyframeCloudRegMsg);
    
    // float loopFitnessScoreThreshold = 0.3; // user parameter but fixed low value is safe. 
    if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold) {
        std::cout << "[SC loop] ICP fitness test failed (" << icp.getFitnessScore() << " > " << loopFitnessScoreThreshold << "). Reject this SC loop." << std::endl;
        return gtsam::Pose3();
    } else {
        std::cout << "[SC loop] ICP fitness test passed (" << icp.getFitnessScore() << " < " << loopFitnessScoreThreshold << "). Add this SC loop." << std::endl;
    }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();

    Eigen::Affine3f tWrong = Pose6dToAffine3f(keyframePosesUpdated[_curr_kf_idx]);

    // 闭环优化后当前帧位姿
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
    pcl::getTranslationAndEulerAngles(tCorrect, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    // 闭环匹配帧的位姿
    gtsam::Pose3 poseTo =  Pose6dTogtsamPose3(keyframePosesUpdated[_loop_kf_idx]);

    return poseFrom.between(poseTo);
} // doICPVirtualRelative

void process_pg()
{
    while(1)
    {
		while ( !odometryBuf.empty() && !fullResBuf.empty() )
        {
            //
            // pop and check keyframe is or not  
            // 
			mBuf.lock();       
            while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < fullResBuf.front()->header.stamp.toSec())
                odometryBuf.pop();
            if (odometryBuf.empty())
            {
                mBuf.unlock();
                break;
            }

            // Time equal check
            timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
            timeLaser = fullResBuf.front()->header.stamp.toSec();

            laserCloudFullRes->clear();
            pcl::PointCloud<PointType>::Ptr thisKeyFrame(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*fullResBuf.front(), *thisKeyFrame);
            fullResBuf.pop();

            Pose6D pose_curr = getOdom(odometryBuf.front());
            odometryBuf.pop();

            // find nearest gps 
            double eps = 0.1; // find a gps topioc arrived within eps second 
            while (!gpsBuf.empty()) {
                auto thisGPS = gpsBuf.front();
                auto thisGPSTime = thisGPS->header.stamp.toSec();
                if( abs(thisGPSTime - timeLaserOdometry) < eps ) {
                    currGPS = thisGPS;
                    hasGPSforThisKF = true; 
                    break;
                } else {
                    hasGPSforThisKF = false;
                }
                gpsBuf.pop();
            }
            mBuf.unlock(); 

            //
            // Early reject by counting local delta movement (for equi-spereated kf drop)
            // 
            odom_pose_prev = odom_pose_curr;
            odom_pose_curr = pose_curr;
            Pose6D dtf = diffTransformation(odom_pose_prev, odom_pose_curr); // dtf means delta_transform

            double delta_translation = sqrt(dtf.x*dtf.x + dtf.y*dtf.y + dtf.z*dtf.z); // note: absolute value. 
            translationAccumulated += delta_translation;
            rotaionAccumulated += (dtf.roll + dtf.pitch + dtf.yaw); // sum just naive approach.  

            // 关键帧选择
            if( translationAccumulated > keyframeMeterGap || rotaionAccumulated > keyframeRadGap ) {
                isNowKeyFrame = true;
                translationAccumulated = 0.0; // reset 
                rotaionAccumulated = 0.0; // reset 
            } else {
                isNowKeyFrame = false;
            }

            if( ! isNowKeyFrame ) 
                continue; 

            if( !gpsOffsetInitialized ) {
                if(hasGPSforThisKF) { // if the very first frame 
                    gpsAltitudeInitOffset = currGPS->altitude;
                    gpsOffsetInitialized = true;
                } 
            }

            //
            // Save data and Add consecutive node 
            //
            pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
            
            // Try CUDA acceleration first, fallback to CPU if needed
            bool cuda_success = false;
            if (cuda_available) {
                cuda_success = cuda_processor.DownsamplePointCloud(thisKeyFrame, thisKeyFrameDS, 0.4f);
            }
            
            if (!cuda_success) {
                // Fallback to CPU implementation
                // Try CUDA acceleration first, fallback to CPU if needed
                bool cuda_success = false;
                if (cuda_available) {
                    cuda_success = cuda_processor.DownsamplePointCloud(thisKeyFrame, thisKeyFrameDS, 0.4f);
                }
                
                if (!cuda_success) {
                    // Fallback to CPU implementation
                    downSizeFilterScancontext.setInputCloud(thisKeyFrame);
                    downSizeFilterScancontext.filter(*thisKeyFrameDS);
                }
            }

            mKF.lock(); 
            keyframeLaserClouds.push_back(thisKeyFrameDS);
            keyframePoses.push_back(pose_curr);
            {
                // 发布关键帧id
                std_msgs::Header keyFrameHeader;
                keyFrameHeader.seq = pose_curr.seq;
                keyFrameHeader.stamp = ros::Time::now();
                pubKeyFramesId.publish(keyFrameHeader);
            }
            keyframePosesUpdated.push_back(pose_curr); // init
            keyframeTimes.push_back(timeLaserOdometry);

            // CPU optimization: Track keyframes processed
            total_keyframes_processed++;

            scManager.makeAndSaveScancontextAndKeys(*thisKeyFrameDS);

            laserCloudMapPGORedraw = true;
            mKF.unlock(); 

            const int prev_node_idx = keyframePoses.size() - 2; 
            const int curr_node_idx = keyframePoses.size() - 1; // becuase cpp starts with 0 (actually this index could be any number, but for simple implementation, we follow sequential indexing)
            if( ! gtSAMgraphMade /* prior node */) {
                const int init_node_idx = 0; 
                gtsam::Pose3 poseOrigin = Pose6DtoGTSAMPose3(keyframePoses.at(init_node_idx));
                // auto poseOrigin = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));

                mtxPosegraph.lock();
                {
                    // prior factor 
                    gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(init_node_idx, poseOrigin, priorNoise));
                    initialEstimate.insert(init_node_idx, poseOrigin);
                    // runISAM2opt();          
                }   
                mtxPosegraph.unlock();

                gtSAMgraphMade = true; 

                cout << "posegraph prior node " << init_node_idx << " added" << endl;
            } else /* consecutive node (and odom factor) after the prior added */ { // == keyframePoses.size() > 1 
                gtsam::Pose3 poseFrom = Pose6DtoGTSAMPose3(keyframePoses.at(prev_node_idx));
                gtsam::Pose3 poseTo = Pose6DtoGTSAMPose3(keyframePoses.at(curr_node_idx));

                mtxPosegraph.lock();
                {
                    // odom factor
                    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, poseFrom.between(poseTo), odomNoise));

                    // gps factor 
                    if(hasGPSforThisKF) {
                        double curr_altitude_offseted = currGPS->altitude - gpsAltitudeInitOffset;
                        mtxRecentPose.lock();
                        gtsam::Point3 gpsConstraint(recentOptimizedX, recentOptimizedY, curr_altitude_offseted); // in this example, only adjusting altitude (for x and y, very big noises are set) 
                        mtxRecentPose.unlock();
                        gtSAMgraph.add(gtsam::GPSFactor(curr_node_idx, gpsConstraint, robustGPSNoise));
                        cout << "GPS factor added at node " << curr_node_idx << endl;
                    }
                    initialEstimate.insert(curr_node_idx, poseTo);                
                    // runISAM2opt();
                }
                mtxPosegraph.unlock();

                if(curr_node_idx % 100 == 0)
                    cout << "posegraph odom node " << curr_node_idx << " added." << endl;
            }
            // if want to print the current graph, use gtSAMgraph.print("\nFactor Graph:\n");

            // save utility 
            std::string curr_node_idx_str = padZeros(curr_node_idx);
            pcl::io::savePCDFileBinary(pgScansDirectory + curr_node_idx_str + ".pcd", *thisKeyFrame); // scan 
            pgTimeSaveStream << timeLaser << std::endl; // path 
        }

        // ps. 
        // scan context detector is running in another thread (in constant Hz, e.g., 1 Hz)
        // pub path and point cloud in another thread

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_pg

void performSCLoopClosure(void)
{
    if( int(keyframePoses.size()) < scManager.NUM_EXCLUDE_RECENT) // do not try too early 
        return;

    auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
    int SCclosestHistoryFrameID = detectResult.first;
    if( SCclosestHistoryFrameID != -1 ) { 
        const int prev_node_idx = SCclosestHistoryFrameID;
        const int curr_node_idx = keyframePoses.size() - 1; // because cpp starts 0 and ends n-1
        cout << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx << "" << endl;

        mBuf.lock();
        scLoopICPBuf.push(std::pair<int, int>(prev_node_idx, curr_node_idx));
        // addding actual 6D constraints in the other thread, icp_calculation.
        mBuf.unlock();
    }
} // performSCLoopClosure

pcl::PointCloud<pcl::PointXYZ>::Ptr vector2pc(const std::vector<Pose6D> vectorPose6d){
    pcl::PointCloud<pcl::PointXYZ>::Ptr res( new pcl::PointCloud<pcl::PointXYZ> ) ;
    for( auto p : vectorPose6d){
        // Validate point coordinates to prevent KdTree assertion errors
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
            res->points.emplace_back(p.x, p.y, p.z);
        } else {
            std::cout << "[WARNING] Invalid pose coordinates detected: (" 
                      << p.x << ", " << p.y << ", " << p.z << "). Skipping this pose." << std::endl;
        }
    }
    return res;
}

/**
 * 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
*/
bool detectLoopClosureDistance(int *loopKeyCur, int *loopKeyPre)
{
    // 当前关键帧
    // int loopKeyCur = keyframePoses.size() - 1;
    // int loopKeyPre = -1;

    // 当前帧已经添加过闭环对应关系，不再继续添加
    auto it = loopIndexContainer.find(*loopKeyCur);
    if (it != loopIndexContainer.end())
        return false;

    // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合
    pcl::PointCloud<pcl::PointXYZ>::Ptr copy_cloudKeyPoses3D = vector2pc(keyframePoses);
    
    // Validate that we have sufficient valid points and that the query point is valid
    if (copy_cloudKeyPoses3D->points.empty()) {
        std::cout << "[WARNING] No valid keyframe poses available for loop closure detection." << std::endl;
        return false;
    }
    
    // Validate the query point (last pose)
    const auto& lastPose = copy_cloudKeyPoses3D->back();
    if (!std::isfinite(lastPose.x) || !std::isfinite(lastPose.y) || !std::isfinite(lastPose.z)) {
        std::cout << "[WARNING] Invalid query pose coordinates: (" 
                  << lastPose.x << ", " << lastPose.y << ", " << lastPose.z 
                  << "). Skipping loop closure detection." << std::endl;
        return false;
    }

    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
    kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
    ROS_INFO_STREAM("Found " << pointSearchIndLoop.size() << " candidates for loop closure.");
    // 在候选关键帧集合中，找到与当前帧时间相隔较远的帧，设为候选匹配帧
    for(int i = 0; i < pointSearchIndLoop.size(); ++i)
    {
        int id = pointSearchIndLoop[i];
        if ( abs( keyframeTimes[id] - keyframeTimes[*loopKeyCur] ) > historyKeyframeSearchTimeDiff )
        {
            *loopKeyPre = id;
            break;
        }
    }

    if (*loopKeyPre == -1 || *loopKeyCur == *loopKeyPre)
        return false;

    // *latestID = loopKeyCur;
    // *closestID = loopKeyPre;

    return true;
}

void performRSLoopClosure(void)
{
    if( keyframePoses.empty() ) // 如果历史关键帧为空
        return;

    // 当前关键帧索引，候选闭环匹配帧索引
    int loopKeyCur = keyframePoses.size() - 1;
    int loopKeyPre = -1;
    if ( detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) ){
        cout << "Loop detected! - between " << loopKeyPre << " and " << loopKeyCur << "" << endl;
        mBuf.lock();
        scLoopICPBuf.push(std::pair<int, int>(loopKeyPre, loopKeyCur));
        loopIndexContainer[loopKeyCur] = loopKeyPre ;
        // addding actual 6D constraints in the other thread, icp_calculation.
        mBuf.unlock();
    } else 
        return;
} // performRSLoopClosure

/**
 * rviz展示闭环边
*/
void visualizeLoopClosure()
{
    if (loopIndexContainer.empty())
        return;
    
    visualization_msgs::MarkerArray markerArray;
    // 闭环顶点
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = "camera_init"; // camera_init
    markerNode.header.stamp = ros::Time().fromSec( keyframeTimes.back() );
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
    markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
    markerNode.color.a = 1;
    // 闭环边
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = "camera_init";
    markerEdge.header.stamp = ros::Time().fromSec( keyframeTimes.back() );
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
    markerEdge.color.a = 1;

    // 遍历闭环
    for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
    {
        int key_cur = it->first;
        int key_pre = it->second;
        geometry_msgs::Point p;
        p.x = keyframePosesUpdated[key_cur].x;
        p.y = keyframePosesUpdated[key_cur].y;
        p.z = keyframePosesUpdated[key_cur].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        p.x = keyframePosesUpdated[key_pre].x;
        p.y = keyframePosesUpdated[key_pre].y;
        p.z = keyframePosesUpdated[key_pre].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    pubLoopConstraintEdge.publish(markerArray);
}

void process_lcd(void)
{
    // CPU optimization: Adaptive loop closure detection frequency
    float optimized_freq = std::min(static_cast<float>(loopClosureFrequency), 1.0f); // Cap at 1 Hz to reduce CPU usage
    ros::Rate rate(optimized_freq);
    
    std::cout << "[CPU OPT] Loop closure detection frequency reduced to " << optimized_freq << " Hz" << std::endl;
    
    while (ros::ok())
    {
        rate.sleep();
        // performSCLoopClosure();
        performRSLoopClosure(); // TODO
        visualizeLoopClosure();
    }
} // process_lcd

void process_icp(void)
{
    while(1)
    {
		while ( !scLoopICPBuf.empty() )
        {
            if( scLoopICPBuf.size() > 30 ) {
                ROS_WARN("Too many loop clousre candidates to be ICPed is waiting ... Do process_lcd less frequently (adjust loopClosureFrequency)");
            }

            mBuf.lock(); 
            std::pair<int, int> loop_idx_pair = scLoopICPBuf.front();
            scLoopICPBuf.pop();
            mBuf.unlock(); 

            const int prev_node_idx = loop_idx_pair.first;
            const int curr_node_idx = loop_idx_pair.second;
            auto relative_pose = doICPVirtualRelative(prev_node_idx, curr_node_idx);
            // if( !gtsam::Pose3::equals(relative_pose, gtsam::Pose3::identity()) ) {
            if( !relative_pose.equals( gtsam::Pose3() )) {
                // gtsam::Pose3 relative_pose = relative_pose_optional.value();
                mtxPosegraph.lock();
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(curr_node_idx, prev_node_idx, relative_pose, robustLoopNoise));
                
                // CPU optimization: Track loop closures and force next optimization
                loop_closure_count++;
                force_next_optimization = true;
                std::cout << "[CPU OPT] Loop closure #" << loop_closure_count 
                          << " detected between nodes " << curr_node_idx << " and " << prev_node_idx << std::endl;
                
                // runISAM2opt();
                mtxPosegraph.unlock();
            } 
        }

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_icp

void process_viz_path(void)
{
    // CPU optimization: Reduce path publishing frequency
    float optimized_hz = std::min(static_cast<float>(vizPathFrequency), 2.0f); // Cap at 2 Hz maximum
    ros::Rate rate(optimized_hz);
    
    std::cout << "[CPU OPT] Path visualization frequency reduced to " << optimized_hz << " Hz" << std::endl;
    
    while (ros::ok()) {
        rate.sleep();
        if(recentIdxUpdated > 1) {
            pubPath();
        }
    }
}

void process_isam(void)
{
    // CPU optimization: Adaptive optimization frequency
    // Start with lower frequency and increase only when needed
    float base_hz = std::max(0.2f, static_cast<float>(graphUpdateFrequency) * 0.5f); // Reduce base frequency by 50%
    float current_hz = base_hz;
    
    ros::Rate rate(current_hz);
    
    while (ros::ok()) {
        rate.sleep();
        
        if( gtSAMgraphMade ) {
            // Adaptive frequency based on system state
            bool needs_frequent_optimization = false;
            
            // Increase frequency temporarily if:
            // 1. Recent loop closures detected
            // 2. Large number of new keyframes since last optimization
            static int last_keyframe_count = 0;
            int new_keyframes = total_keyframes_processed - last_keyframe_count;
            
            if (force_next_optimization || new_keyframes > 10) {
                needs_frequent_optimization = true;
                current_hz = graphUpdateFrequency; // Use full frequency
            } else {
                current_hz = base_hz; // Use reduced frequency
            }
            
            // Update rate if changed
            static float last_hz = current_hz;
            if (std::abs(current_hz - last_hz) > 0.1f) {
                rate = ros::Rate(current_hz);
                last_hz = current_hz;
                std::cout << "[CPU OPT] Adjusted optimization frequency to " << current_hz << " Hz" << std::endl;
            }
            
            mtxPosegraph.lock();
            runISAM2opt();
            // cout << "running isam2 optimization ..." << endl;
            mtxPosegraph.unlock();
            
            last_keyframe_count = total_keyframes_processed;

            saveOptimizedVerticesKITTIformat(isamCurrentEstimate, pgKITTIformat); // pose
            saveOdometryVerticesKITTIformat(odomKITTIformat); // pose
        }
    }
}

void pubMap(void)
{
    int SKIP_FRAMES = 1; // sparse map visulalization to save computations 
    int counter = 0;

    laserCloudMapPGO->clear();

    mKF.lock(); 
    // for (int node_idx=0; node_idx < int(keyframePosesUpdated.size()); node_idx++) {
    for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) {
        if(counter % SKIP_FRAMES == 0) {
            *laserCloudMapPGO += *local2global(keyframeLaserClouds[node_idx], keyframePosesUpdated[node_idx]);
        }
        counter++;
    }
    mKF.unlock(); 

    // Downsample map with CUDA acceleration
    pcl::PointCloud<PointType>::Ptr laserCloudMapPGODS(new pcl::PointCloud<PointType>());
    
    bool cuda_success = false;
    if (cuda_available) {
        cuda_success = cuda_processor.DownsamplePointCloud(laserCloudMapPGO, laserCloudMapPGODS, mapVizFilterSize);
        if (cuda_success) {
            *laserCloudMapPGO = *laserCloudMapPGODS;
        }
    }
    
    if (!cuda_success) {
        // Fallback to CPU implementation
        downSizeFilterMapPGO.setInputCloud(laserCloudMapPGO);
        downSizeFilterMapPGO.filter(*laserCloudMapPGO);
    }

    sensor_msgs::PointCloud2 laserCloudMapPGOMsg;
    pcl::toROSMsg(*laserCloudMapPGO, laserCloudMapPGOMsg);
    laserCloudMapPGOMsg.header.frame_id = "camera_init";
    pubMapAftPGO.publish(laserCloudMapPGOMsg);
}

void process_viz_map(void)
{
    // CPU optimization: Significantly reduce map visualization frequency
    // Map visualization is very expensive, so reduce it more aggressively
    float optimized_freq = std::min(static_cast<float>(vizmapFrequency), 0.2f); // Cap at 0.2 Hz (once every 5 seconds)
    ros::Rate rate(optimized_freq);
    
    std::cout << "[CPU OPT] Map visualization frequency reduced to " << optimized_freq << " Hz" << std::endl;
    
    while (ros::ok()) {
        rate.sleep();
        if(recentIdxUpdated > 1) {
            pubMap();
        }
    }
} // pointcloud_viz


int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserPGO");
	ros::NodeHandle nh;

    nh.param<std::string>("save_directory", save_directory, "/home/surya/workspaces/slam_ws/src/FAST_LIO_LC/"); // pose assignment every k m move 
    
    // Ensure save_directory ends with '/'
    if (!save_directory.empty() && save_directory.back() != '/') {
        save_directory += "/";
    }
    
    pgKITTIformat = save_directory + "optimized_poses.txt";
    odomKITTIformat = save_directory + "odom_poses.txt";
    pgTimeSaveStream = std::fstream(save_directory + "times.txt", std::fstream::out); 
    pgTimeSaveStream.precision(std::numeric_limits<double>::max_digits10);
    pgScansDirectory = save_directory + "Scans/";
    
    // Check if directory creation was successful
    int result1 = system((std::string("rm -rf ") + pgScansDirectory).c_str());
    int result2 = system((std::string("mkdir -p ") + pgScansDirectory).c_str());
    
    if (result2 != 0) {
        ROS_ERROR("Failed to create directory: %s", pgScansDirectory.c_str());
        ROS_ERROR("Please check directory permissions or use a different save_directory parameter");
    } else {
        ROS_INFO("Created scan directory: %s", pgScansDirectory.c_str());
    }

	nh.param<double>("keyframe_meter_gap", keyframeMeterGap, 2.0); // pose assignment every k m move 
	nh.param<double>("keyframe_deg_gap", keyframeDegGap, 10.0); // pose assignment every k deg rot 
    keyframeRadGap = deg2rad(keyframeDegGap);

	nh.param<double>("sc_dist_thres", scDistThres, 0.2);  
	nh.param<double>("sc_max_radius", scMaximumRadius, 80.0); // 80 is recommended for outdoor, and lower (ex, 20, 40) values are recommended for indoor 

    // for loop closure detection
	nh.param<double>("historyKeyframeSearchRadius", historyKeyframeSearchRadius, 10.0);  
	nh.param<double>("historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0);  
	nh.param<int>("historyKeyframeSearchNum", historyKeyframeSearchNum, 25);  
	nh.param<double>("loopNoiseScore", loopNoiseScore, 0.5);  
	nh.param<int>("graphUpdateTimes", graphUpdateTimes, 2);  
	nh.param<double>("loopFitnessScoreThreshold", loopFitnessScoreThreshold, 0.3);  

	nh.param<double>("speedFactor", speedFactor, 1);  
    {
        nh.param<double>("loopClosureFrequency", loopClosureFrequency, 2);  
        loopClosureFrequency *= speedFactor;
        nh.param<double>("graphUpdateFrequency", graphUpdateFrequency, 1.0);  
        graphUpdateFrequency *= speedFactor;
        nh.param<double>("vizmapFrequency", vizmapFrequency, 0.1);  
        vizmapFrequency *= speedFactor;
        nh.param<double>("vizPathFrequency", vizPathFrequency, 10);  
        vizPathFrequency *= speedFactor;
        
    }

    

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);
    initNoises();

    scManager.setSCdistThres(scDistThres);
    scManager.setMaximumRadius(scMaximumRadius);

    float filter_size = 0.4; 
    downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);
    downSizeFilterICP.setLeafSize(filter_size, filter_size, filter_size);

	nh.param<double>("mapviz_filter_size", mapVizFilterSize, 0.4); // pose assignment every k frames 
    downSizeFilterMapPGO.setLeafSize(mapVizFilterSize, mapVizFilterSize, mapVizFilterSize);

    // Initialize CUDA processor for acceleration
    cuda_available = pgo_cuda::CudaPGOProcessor::IsCudaAvailable();
    if (cuda_available) {
        ROS_INFO("CUDA acceleration available for PGO");
    } else {
        ROS_INFO("CUDA not available, using CPU fallbacks for PGO");
    }

	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_registered_local", 100, laserCloudFullResHandler);
	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 100, laserOdometryHandler);
	ros::Subscriber subGPS = nh.subscribe<sensor_msgs::NavSatFix>("/gps/fix", 100, gpsHandler);

	pubOdomAftPGO = nh.advertise<nav_msgs::Odometry>("/aft_pgo_odom", 100);
	pubOdomRepubVerifier = nh.advertise<nav_msgs::Odometry>("/repub_odom", 100);

    // for front-end
    pubKeyFramesId = nh.advertise<std_msgs::Header>("/key_frames_ids", 10);

    // for loop closure
    pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/loop_closure_constraints", 1);
	pubLoopScanLocalRegisted = nh.advertise<sensor_msgs::PointCloud2>("/loop_scan_local_registed", 100);

	pubPathAftPGO = nh.advertise<nav_msgs::Path>("/aft_pgo_path", 100);
	pubMapAftPGO = nh.advertise<sensor_msgs::PointCloud2>("/aft_pgo_map", 100);

	pubLoopScanLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_scan_local", 100);
	pubLoopSubmapLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_submap_local", 100);

	std::thread posegraph_slam {process_pg}; // pose graph construction
	std::thread lc_detection {process_lcd}; // loop closure detection 
	std::thread icp_calculation {process_icp}; // loop constraint calculation via icp 
	std::thread isam_update {process_isam}; // if you want to call less isam2 run (for saving redundant computations and no real-time visulization is required), uncommment this and comment all the above runisam2opt when node is added. 

	std::thread viz_map {process_viz_map}; // visualization - map (low frequency because it is heavy)
	//std::thread viz_path {process_viz_path}; // visualization - path (high frequency)

 	ros::spin();

	return 0;
}