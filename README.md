C++ implementation for *Oriented Point Sampling for Plane Detection in Unorganized Point Clouds* by Bo Sun and Philippos Mordohai

[Link to their paper](https://arxiv.org/pdf/1905.02553.pdf) 

The algorithm detects all planes in an unorganized point cloud.
Normals are computed only for a small fraction of all the points in the point cloud (usually between 0.3% and 3%). This allows for single point hypotheses (unlike most approaches that use three unoriented points) without being computationally intensive.
 
### Build
Dependencies are PCL 1.10 (point cloud processing and visualization) and Boost library
```
mkdir build && cd build
cmake ..
make -j6
```
### Usage
`./orientedPointSampling <inputFilename>`
*input_filename* is a `.pcd` file containing a point cloud

### ToDO
- [x] Implement merging process
- [ ] Implement ground removal
