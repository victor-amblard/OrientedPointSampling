#include "Ops.h"

int main(int argc, char * argv[])
{
    if (argc < 2)
    {
        std::cerr << "Invalid number of arguments! Exiting...";
        return 0;
    }
    std::string filename = argv[1];
    std::cerr << "Loading " << filename << std::endl;
    pcl::PointCloudXYZ::Ptr cloud(new pcl::PointCloudXYZ);
    int success = pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud);
    if (success == -1)
    {
        std::cerr << "Failed to load point cloud! Exiting...";
        return 0;
    }
    std::cerr << "Point cloud has " << cloud->points.size() << " points " << std::endl;
    process(cloud);    
}