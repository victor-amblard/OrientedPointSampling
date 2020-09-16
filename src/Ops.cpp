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

float getDistanceToPlane(const Eigen::Vector3f& p,
                        const Plane& P)
{
    Eigen::Vector3f diff = (p - P.second.second);
    float distance = std::fabs(diff.dot(P.second.first)); //normal vector is already of norm 1  

    return distance;
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
Eigen::Vector3f computeGlobalSVD(const std::vector<pcl::PointXYZ>& allPoints)
{
    int N = allPoints.size();
    Eigen::MatrixXd A(3,N);

    for (int i = 0 ; i < N ; ++i)
    {
        Eigen::Vector3d eigPoint = allPoints.at(i).getVector3fMap().cast<double>();
        A.block(0,i,3,1) = eigPoint;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV | Eigen::ComputeFullU );
    svd.computeV();
    Eigen::Vector3d normal(svd.matrixV()(2,0), svd.matrixV()(2,1), svd.matrixV()(2,2)); 
    return normal.cast<float>().normalized();

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
std::pair<int, std::set<int>> detectCloud(const pcl::PointCloudXYZ::Ptr cloud,
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
    int Ns = samples.size();

    int iIter = 0;
    int nIter = Ncloud;
    int curMaxInliers = 0;
    std::set<int> bestInliers = {};
    int idxOI = -1;
    bool converged = true;
    int maxIter = 3000;
    size_t nInliers = 0;

    while (iIter < nIter)
    {
        int randIdx = std::rand() % Ns;
        idxOI = samples.at(randIdx); 
        std::set<int> inliers;

        if (ground || allOrientations.at(randIdx) == HORIZONTAL)
        {

            for (int iPoint = 0 ; iPoint < Ns ; ++iPoint)
            {
                if (iPoint != randIdx && CLIP_ANGLE(std::acos(allNormals.at(iPoint).dot(allNormals.at(randIdx)))) < threshAngle)
                {
                    double dist = getDistanceToPlane(idxOI, samples.at(iPoint), cloud, allNormals.at(randIdx));
                    if (dist < threshDistPlane )
                    {
                        inliers.insert(iPoint); // 2 criteria : distance to plane and common orientation
                    }
                }
            }

            nInliers = inliers.size();

            if (nInliers > threshInliers)
            {
                if (nInliers > curMaxInliers)
                {
                    curMaxInliers = nInliers;
                    bestInliers = inliers;
                    double e = 1 - (float)(nInliers) / Ns;
                    nIter = std::log(1 - p) / std::log(1- (1-e));

                }
            }
        }   
        
        if (iIter > maxIter){
            if (curMaxInliers == 0) // To avoid waiting 1h until the end of the loop
                converged = false;
            break;
        }
        ++iIter;
    }
    std::chrono::time_point<std::chrono::system_clock> tEnd = std::chrono::system_clock::now();
    double elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count();
    if (converged)
        std::cerr << "Converged after " << iIter << " iterations to a plane with " << curMaxInliers << " inliers [process took: " <<
        elapsedTime<< " ms]"<< std::endl;
    else
        std::cerr << "Failed to converge!" << std::endl;

    return std::make_pair(idxOI, bestInliers);
}

void visualizePlanes(const std::vector<Plane>& planes,
                     const pcl::PointCloudXYZ::Ptr remainingCloud)
{
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    
    pcl::PointCloudXYZRGB::Ptr coloredCloud(new pcl::PointCloudXYZRGB);
    int i = 0;
    for (auto plane : planes)
    {
        for (pcl::PointXYZ point : plane.first)
        {
            coloredCloud->points.push_back(pcl::PointXYZRGB(point.x, point.y, point.z, r[(i*43)%255]*255, g[(i*43)%255]*255, b[(i*43)%255]*255));
        }
        ++i;
    }
    std::cerr << "[INFO] Total " << coloredCloud->points.size() << " points in colored cloud" << std::endl;

    viewer->addPointCloud<pcl::PointXYZRGB> (coloredCloud, "sample cloud");

    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
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

    std::vector<Plane> planes; 
    std::pair<int, std::set<int>> inliersPlane;
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
                          
        // Remove inliers from cloud
        pcl::ExtractIndices<pcl::PointXYZ> extractIndices;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        
        sInliers = inliersPlane.second.size();   

        if (!sInliers)
            break; // Failed to converge

        Eigen::Vector3f centroid(0,0,0);
        std::vector<pcl::PointXYZ> curPlane;

        for (auto it = inliersPlane.second.begin() ; it != inliersPlane.second.end();++it)
        {
            centroid += (cloud->points[ps.at(*it)]).getVector3fMap() / sInliers;
            curPlane.push_back(cloud->points[ps.at(*it)]);
        }

        std::vector<int> tmpPs;
        std::vector<Eigen::Vector3f> tmpNormals;
        std::vector<int> tmpOrientation;
        tmpOrientation.reserve(Ns-sInliers);
        tmpNormals.reserve(Ns-sInliers);
        tmpPs.reserve(Ns-sInliers);

        for (int i = 0 ; i < Ns ; ++i)
        {
            if (inliersPlane.second.find(i) == inliersPlane.second.end())
            {
                tmpPs.push_back(ps.at(i));
                tmpNormals.push_back(allNormals.at(i));
                tmpOrientation.push_back(allOrientations.at(i));
            }
        }
        allOrientations = tmpOrientation;
        allNormals = tmpNormals;
        ps = tmpPs;
         
        Ns = ps.size();
        Eigen::Vector3f planeNormal = computeGlobalSVD(curPlane);
        planes.push_back(std::make_pair(curPlane, std::make_pair(planeNormal, centroid)));

        if (verbose)
        {
            std::cerr << "[INFO] Plane normal: (" << planeNormal(0) <<" ;" << planeNormal(1) << " ; " << planeNormal(2) << ")" << std::endl;
            std::cerr << "       Centroid: (" << centroid(0) << ";" << centroid(1) << ";" << centroid(2) << ")" << std::endl;
        }

        ++iIter;
        /*
        extractIndices.setInputCloud(cloud);
        extractIndices.setIndices(inliers);
        extractIndices.setNegative(true);
        extractIndices.filter(*cloud);*/

    }while(ps.size() > threshInliers);
    
    // Step 4: Merge planes
    size_t planesNumber;
    
    do
    {
        planesNumber = planes.size();
        mergePlanes(planes);
        std::cerr << planes.size() << std::endl;

    }while(planes.size() != planesNumber);
    
    getFinitePlanes(planes, cloud);
    // Step 5: Visualize result
    visualizePlanes(planes, cloud);
}

void mergePlanes(std::vector<Plane>& planes)
{
    for (auto itA = planes.begin() ; itA != planes.end();)
    {
        for (auto itB = itA; itB != planes.end() ;)
        {
            if (itA != itB)
            {
                bool toMerge = comparePlanes(*itA, *itB); 
                
                if (toMerge)
                {
                    std::vector<pcl::PointXYZ> inliersA = (*itA).first;
                    std::vector<pcl::PointXYZ> inliersB = (*itB).first;
                    size_t nA = inliersA.size();
                    size_t nB = inliersB.size();

                    for (auto elem : inliersA)
                        itB->first.push_back(elem);
                    
                    Eigen::Vector3f nCentroid = 1 / (nA+nB) * (nA * itA->second.second + nB * itB->second.second); //updated centroid
                    Eigen::Vector3f nNormal = computeGlobalSVD(itB->first); //Recompute normals with all inliers
                    itB->second.first = nNormal;
                    itB->second.second = nCentroid;
                    
                    itA = planes.erase(itA);

                    break;
                }
                else
                {
                    ++itB;
                }
            }else
            {
                ++itB;
            }
        }
        ++itA;
    }
}

bool comparePlanes(const Plane& A, 
                   const Plane& B)
{
    float distA = getDistanceToPlane(A.second.second, B);
    float distB  = getDistanceToPlane(B.second.second, A);
    float angle = CLIP_ANGLE(std::acos(A.second.first.dot(B.second.first)));

    return (distA < defaultParams::threshDistToPlane) && (distB < defaultParams::threshDistToPlane) && (angle < defaultParams::threshAngleToAxis);
}

