# BLAST
BLAS (Basic Linear Algebra Subprograms) is the backbone of a lot of the modern mathematical libraies, such as LAPACK, NumPy, etc. Although using those will be much easier (No, I do not endorse or encourage any use of FORTRAN,
although learning it will be pretty cool), I feel like to be able to understand how the math works (efficiently) underneath, is vital to how I can apply those modern libraries effectively. Hence, 
I wrote some CUDA kernels to execute these linear algebraic subroutines. BLAST's T is because I used a lot of template programming, but later on, I dropped it in favor of strict half precision.

## So, What did I Learn?
After building these tools I had to benchmark them, because otherwise I would have no way to find how efficient these kernels are on Nvidia GPUs. For
 reference, this is the data from running the kernels with at least one dimension of the size being 1<<20, (Level 2's Matrix Vector multiplication had to be tested with size 1<<15 because I don't want my GPU to blow up) and the host hardware being the Tesla T4 on 
 the Turing architecture, measured by Nvidia Nsight Compute.

 Anyways, here are some highlights about the work done:
1. Optimized Triangular Matrix-Vector Multiplication (with a nice report on optimized Matrix-Vector Multiplication for reference)
<img width="1244" height="336" alt="image" src="https://github.com/user-attachments/assets/89e70be1-ce91-4d10-997c-4085392bdc3e" />

<img width="1244" height="269" alt="image" src="https://github.com/user-attachments/assets/0843bc59-184b-4e19-90f4-f418a120d3cc" />

<img width="963" height="358" alt="image" src="https://github.com/user-attachments/assets/cb3115bf-ec73-4efe-88bc-68a961f0104e" />

<img width="1052" height="326" alt="image" src="https://github.com/user-attachments/assets/56cdbcdf-7710-4db8-9942-1854f1952b6a" />

<img width="1052" height="271" alt="image" src="https://github.com/user-attachments/assets/8d6342eb-d8d4-41c3-af9a-91d091030630" />

That is about a 10x speed up, (could be 25x for small n like n = 1024) because of the warp level computation and one warp reduction, rather than letting one thread handle a row.
Also, that is some impressive achieved occupancy.

2. The fixes for modern CC 7.x and rewrites on parallel reduction for template programming support
```
template <unsigned int blockSize, typename T>
__device__ void warpReduce(/*volatile*/ T *sdata, unsigned int tid) {
    if (blockSize >= 64) {sdata[tid] += sdata[tid + 32]; __syncwarp();}
    if (blockSize >= 32) {sdata[tid] += sdata[tid + 16]; __syncwarp();}
    if (blockSize >= 16) {sdata[tid] += sdata[tid + 8]; __syncwarp();}
    if (blockSize >= 8) {sdata[tid] += sdata[tid + 4]; __syncwarp();}
    if (blockSize >= 4) {sdata[tid] += sdata[tid + 2]; __syncwarp();}
    if (blockSize >= 2) {sdata[tid] += sdata[tid + 1]; __syncwarp();}
}
```
Notice how I got rid of volatile? the ```volatile``` keyword was used for the compiler to not optimize the operations around the variable, such that the compiler will not mess up the operation for this program. Long before the Turing architecture, which this code was running on, the hardware used to guarantee the synchronization for you, called a _**lockstep**_, but later on, that feature was removed. So, using ```__syncwarp()``` allowed me to hit two birds with one stone, avoiding the compiler from reordering the accesses within a barrier, as well as maintaining the synchronization within a warp.

3. Enablement for Template Programming Within Parallel Reduction
Notice these lines:
```
    extern __shared__ char smem[];
    T * sdata = reinterpret_cast<T *>(smem);
```
well, if you use the template for that shared memory immediately, the compiler will scream at you. That is because in my test program, I used ```reduce``` many times, causing multiple different castings that confuses the linker for defining the same symbol with different types. (This is [Lei Mao's finding](https://leimao.github.io/blog/CUDA-Shared-Memory-Templated-Kernel/)).
To bypass that, you can let the linker see that we are always using char as the initial type, and then do a shady ```reinterpret_cast``` later on.
## And What Still Needs to Be Done?
Well, for starters, ```rotmg``` and ```rotm``` are not implemented at all. I will need to grind those out. A good chunk of level 2 is also missing, but I will probably put triangular solve (```trsv```) on the top of my list.
Once I get those out of the way, I will need to look into level 3. Starting off with something compute-bound like GEMM will be fun.
