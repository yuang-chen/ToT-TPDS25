# ToT: Triangle Counting on Tensor Cores

This repository contains the implementation of ToT, an triangle counting algorithm leveraging Tensor Cores on GPUs for acceleration.


### Warning
This implementation utilizes Tensor Cores with 16-bit `half` type, which may introduce precision errors during count summation, particularly with large graphs.  Using the `-e 1` flag to extract the upper triangular portion of the graph can mitigate precision problems in some cases.


## Requirements

- CMake (version 3.22 or higher)
- CUDA Toolkit
- OpenMP
- C++20 compatible compiler
- NVIDIA GPUs with SM >= 80 

## Building the Project

To build the project, follow these steps:
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Running the Program

The program can be used as follows:
```bash
Usage: ./build/apps/tot ... 
              [-i input_file]
              [-e extract_upper_triangular (1 or 0)]
              [-v verify (1 or 0)]
```
Example command:

```bash
./build/apps/tot -i toydata/pli.mtx -v 1
```
## Citation

If you use this code in your research, please cite:
```bibtex
@inproceedings{chen2025triangle,
  title={Triangle Counting on Tensor Cores},
  author={Chen, YuAng and Yu, Jeffrey Xu},
  booktitle={Proceedings of the 30th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming},
  pages={560--562},
  year={2025}
}
```
