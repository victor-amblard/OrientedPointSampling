/**
 * Implementation of the following paper:
 * Oriented Point Sampling for Plane Detection in Unorganized Point Clouds, Bo Sun and Philippos Mordohai (2019) 
 * 
 * by Victor AMBLARD
*/

#include "Ops.h"

#define VERTICAL 1
#define HORIZONTAL 2
#define OTHER 3

double CLIP_ANGLE(double angle){
    if (angle > M_PI)
        angle -= M_PI;
    if (angle < M_PI && angle > M_PI/2)
        angle = M_PI - angle;
    
    return angle;
}

std::vector<int> getNearestNeighbors(const int idx,
                                    const pcl::PointCloudXYZ::Ptr cloud,
                                    const int K,
                                    const pcl::KdTreeFLANN<pcl::PointXYZ>& kdTree)
{
    pcl::PointXYZ searchPoint = cloud->points[idx];
    std::vector<int> idxNeighbors(K);
    std::vector<float> pointsNNSquaredDist(K);
    kdTree.nearestKSearch(searchPoint, K, idxNeighbors, pointsNNSquaredDist);

    return idxNeighbors;
}

float getDistanceToPlane(const int id,
                        const int oId,
                        const pcl::PointCloudXYZ::Ptr cloud,
                        const Eigen::Vector3f& normal)
{
    Eigen::Vector3f diff = (cloud->points[id].getVector3fMap() - cloud->points[oId].getVector3fMap());
    float distance = std::fabs(diff.dot(normal)); //normal vector is already of norm 1  

    return distance;
}

Eigen::Vector3f computeSVDNormal(const std::vector<int>& nnIdx,
                                const int piIdx,
                                const pcl::PointCloudXYZ::Ptr cloud,
                                const double sigma)
{
    Eigen::Vector3f pI = cloud->points[piIdx].getVector3fMap();
    Eigen::Matrix3f Mi = Eigen::Matrix3f::Zero();

    for (auto it = nnIdx.begin(); it != nnIdx.end(); ++it)
    {
        if (*it != piIdx) //All neighbors that are not the reference point
        {
            Eigen::Vector3f qij = cloud->points[*it].getVector3fMap();
            double sqNorm = (qij-pI).squaredNorm();
            double weight = std::exp(-sqNorm / (2 * pow(sigma, 2))); //weight factor 
            Eigen::Matrix3f curMat = weight * 1/sqNorm * (qij - pI) * (qij - pI).transpose();
            Mi += curMat; 
        }
    }

    Eigen::Vector3f normal;
    Eigen::EigenSolver<Eigen::Matrix3f> es;
    es.compute(Mi, true);
    
    auto eVals = es.eigenvalues(); 
    float minEig = std::numeric_limits<float>::max();
    auto eVec = es.eigenvectors();

    for (unsigned int i = 0 ; i < eVals.size();++i)
    {
        if (eVals(i).real() < minEig) //eigenvalues are real
        {   
            minEig = eVals(i).real();
            auto complexNormal = eVec.block(0,i,3,1);
            normal = Eigen::Vector3f(complexNormal(0,0).real(), complexNormal(1,0).real(), complexNormal(2,0).real());
        }
    }


    return normal;
}
//Implements Algorithm 1 from the paper
std::pair<int, std::vector<int>> detectCloud(const pcl::PointCloudXYZ::Ptr cloud,
                                            const std::vector<int>& samples,
                                            const std::vector<Eigen::Vector3f>& allNormals,
                                            const std::vector<int>& allOrientations,
                                            const bool ground,
                                            const double threshDistPlane,
                                            const int threshInliers,
                                            const float threshAngle,
                                            const float p)
{
    std::chrono::time_point<std::chrono::system_clock> tStart = std::chrono::system_clock::now();

    int Ncloud = cloud->points.size();
    int Ns = allNormals.size();

    int iIter = 0;
    int nIter = Ncloud;
    int curMaxInliers = 0;
    std::vector<int> bestInliers;
    int idxOI = -1;

    while (iIter < nIter)
    {
        int randIdx = std::rand() % Ns;
        idxOI = samples[randIdx];    
        std::vector<int> inliers;

        for (int iPoint = 0 ; iPoint < Ns ; ++iPoint)
        {
            if (iPoint != randIdx && ((ground && allOrientations.at(iPoint) == VERTICAL) ||
             (!ground)))
            {
                double dist = getDistanceToPlane(idxOI, samples.at(iPoint), cloud, allNormals.at(randIdx));
                if (dist < threshDistPlane)
                {
                    inliers.push_back(samples.at(iPoint));
                }
            }
        }

        size_t nInliers = inliers.size();

        if (nInliers > threshInliers)
        {
            if (nInliers > curMaxInliers)
            {
                curMaxInliers = nInliers;
                bestInliers = inliers;
                double e = 1 - nInliers / Ns;
                nIter = std::log(1 - p) / std::log(1- (1-e));

            }
        }
        
        ++iIter;
    }
    std::chrono::time_point<std::chrono::system_clock> tEnd = std::chrono::system_clock::now();
    double elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count();

    std::cerr << "Converged after " << iIter << " iterations to a plane with " << curMaxInliers << " inliers [process took: " <<
    elapsedTime<< " ms]"<< std::endl;

    return std::make_pair(idxOI, bestInliers);
}

void visualizePlanes(const std::vector<std::vector<pcl::PointXYZ>>& planes,
                     const pcl::PointCloudXYZ::Ptr remainingCloud)
{
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    
    pcl::PointCloudXYZRGB::Ptr coloredCloud(new pcl::PointCloudXYZRGB);
    int i = 0;
    for (std::vector<pcl::PointXYZ> plane : planes)
    {
        for (pcl::PointXYZ point : plane)
        {
            coloredCloud->points.push_back(pcl::PointXYZRGB(point.x, point.y, point.z, r[(i*43)%255]*255, g[(i*43)%255]*255, b[(i*43)%255]*255));
        }
        ++i;
    }
    std::cerr << "[INFO] Total " << coloredCloud->points.size() << " points in colored cloud" << std::endl;

    viewer->addPointCloud<pcl::PointXYZRGB> (coloredCloud, "sample cloud");

    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();

   while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
    };
}

std::pair<std::vector<Eigen::Vector3f>, std::vector<int>> computeAllNormals(const std::vector<int>& samples,
                                                                            const int K,
                                                                            const pcl::PointCloudXYZ::Ptr cloud,
                                                                            const float threshAngle)
{
    Eigen::Vector3f verticalVector(0,0,1); //used for ground detection

    //Kd-tree construction
    pcl::KdTreeFLANN<pcl::PointXYZ> kdTree;
    kdTree.setInputCloud(cloud);

    std::vector<Eigen::Vector3f> allNormals;
    std::vector<int> allOrientations;
    allNormals.reserve(samples.size());
    allOrientations.reserve(samples.size());

    for (int idSampled: samples)
    {
        std::vector<int> idNN = getNearestNeighbors(idSampled, cloud, K, kdTree);
        Eigen::Vector3f curNormal = computeSVDNormal(idNN, idSampled, cloud);


        double angleToVerticalAxis = CLIP_ANGLE(std::acos(curNormal.dot(verticalVector)));
        int orientationLabel;

        if (angleToVerticalAxis < threshAngle)
        {
            orientationLabel = VERTICAL;
        }
        else
        {
            if (angleToVerticalAxis > M_PI/2 - threshAngle && angleToVerticalAxis < M_PI/2)
            {
                orientationLabel = HORIZONTAL;
            }
            else
            {
                orientationLabel = OTHER;
            }
        }
    
        allNormals.push_back(curNormal);
        allOrientations.push_back(orientationLabel);
    }

    return std::make_pair(allNormals, allOrientations);
}

void process(const pcl::PointCloudXYZ::Ptr cloud,
            const int nbPlanes,
            const bool verbose,
            const double alphaS,
            const int K,
            const double threshDistPlane,
            const int threshInliers,
            const float threshAngle,
            const float p)
{

    if (verbose)
    {
        std::cerr << "======= Current parameters =======" << std::endl;
        std::cerr << "Sampling rate: " << alphaS*100 << "%" << std::endl;
        std::cerr << "# nearest neighbors: " << K << std::endl;
        std::cerr << "Minimum number of inliers to approve a plane: " << threshInliers << std::endl;
        std::cerr << "Tolerance to vertical axis (ground detection) in degrees: " << threshAngle * 180 / M_PI << std::endl;
        std::cerr << "Distance threshold to plane: " << threshDistPlane << std::endl;
        std::cerr << "RANSAC probability for adaptive number of iterations: " << p  << std::endl;
        std::cerr << "==================================" << std::endl;
    }

    size_t sMini = 100;
    size_t sInliers = 0;

    std::vector<std::vector<pcl::PointXYZ>> planes; 
    std::pair<int, std::vector<int>> inliersPlane;
    pcl::PointCloudXYZ originalCloud(*cloud);
    int iIter = 0;

    // Step 1: Draw samples
    int Ncloud = cloud->points.size();
    int Ns = (int)round(alphaS * Ncloud);
    std::vector<int> ps;
    std::vector<int> allPointsIdx(Ncloud);
    for (int i = 0 ; i < Ncloud; ++i)
        allPointsIdx.at(i) = i;

    std::sample(allPointsIdx.begin(), allPointsIdx.end(),std::back_inserter(ps), Ns, std::mt19937{std::random_device{}()});

    // Step 2: Compute normals
    std::pair<std::vector<Eigen::Vector3f>, std::vector<int>> result = computeAllNormals(ps, K, cloud);
    std::vector<Eigen::Vector3f> allNormals = result.first;
    std::vector<int> allOrientations = result.second;
    
    // Step 3: Detect planes
    do
    {
        inliersPlane = detectCloud(cloud, ps, allNormals, allOrientations, false, threshDistPlane, threshInliers, threshAngle, p);
              
        Eigen::Vector3f planeNormal = computeSVDNormal(inliersPlane.second, inliersPlane.first , cloud);
            
        // Remove inliers from cloud
        pcl::ExtractIndices<pcl::PointXYZ> extractIndices;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        
        sInliers = inliersPlane.second.size();   
        Eigen::Vector3f centroid(0,0,0);
        for (int idx : inliersPlane.second)
        {
            inliers->indices.push_back(idx);
            centroid += (cloud->points[idx]).getVector3fMap() / sInliers;
        }

        if (verbose)
        {
            std::cerr << "[INFO] Plane normal: (" << planeNormal(0) <<" ;" << planeNormal(1) << " ; " << planeNormal(2) << ")" << std::endl;
            std::cerr << "       Centroid: (" << centroid(0) << ";" << centroid(1) << ";" << centroid(2) << ")" << std::endl;
        }

        extractIndices.setInputCloud(cloud);
        extractIndices.setIndices(inliers);
        extractIndices.setNegative(true);
        extractIndices.filter(*cloud);

        std::vector<pcl::PointXYZ> curPlane;
        for (int idx: inliersPlane.second)
        {
            curPlane.push_back(cloud->points[idx]);
        }

        planes.push_back(curPlane);
        ++iIter;

    }while(iIter < nbPlanes);
    
    // Step 4: Merge planes
    //TODO 

    // Step 5: Visualize result
    visualizePlanes(planes, cloud);
}