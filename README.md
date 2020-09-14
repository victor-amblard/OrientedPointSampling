C++ implementation for *Oriented Point Sampling for Plane Detection in Unorganized Point Clouds*

### Build
Dependencies are PCL 1.10 and Boost library
```
mkdir build && cd build
cmake ..
make -j6
```
### Usage
`./orientedPointSampling <inputFilename>`
*input_filename* is a `.pcd` file containing a point cloud
