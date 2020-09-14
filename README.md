C++ implementation for *Oriented Point Sampling for Plane Detection in Unorganized Point Clouds* by Bo Sun and Philippos Mordohai
[Link to their paper](https://arxiv.org/pdf/1905.02553.pdf) 
### Build
Dependencies are PCL 1.10 and Boost library
```
mkdir build && cd build
cmake ..
make -j6
```
### Usage
`./orientedPointSampling <inputFilename> <nbPlanes>`
*input_filename* is a `.pcd` file containing a point cloud

### ToDO
[]Implement the merging process
