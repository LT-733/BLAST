# BLAST
BLAS (Basic Linear Algebra Subprograms) is the backbone of a lot of the modern mathematical libraies, such as LAPACK, NumPy, etc. Although using those will be much easier (No, I do not endorse or encourage any use of FORTRAN,
although learning it will be pretty cool), I feel like to be able to understand how the math works (efficiently) underneath, is vital to how I can apply those modern libraries effectively. Hence, 
I wrote some CUDA kernels to execute these linear algebraic subroutines. BLAST's T is because I used a lot of template programming, but later on, I dropped it in favor of strict half precision.

## So, What did I Learn?
After building these tools I had to benchmark them, because otherwise I would have no way to find how efficient these kernels are on Nvidia GPUs. For
 reference, this is the data from running the kernels with at least one dimension of the size being 1<<20, and the host hardware being the Tesla T4 on 
 the Turing architecture, measured by Nvidia Nsight Compute.

 Anyways, here are some highlights about the work done:
1. Optimized Triangular Matrix-Vector Multiplication (with a nice report on optimized Matrix-Vector Multiplication for reference)
<img width="1244" height="336" alt="image" src="https://github.com/user-attachments/assets/89e70be1-ce91-4d10-997c-4085392bdc3e" />

<img width="1244" height="269" alt="image" src="https://github.com/user-attachments/assets/0843bc59-184b-4e19-90f4-f418a120d3cc" />

<img width="1244" height="349" alt="image" src="https://github.com/user-attachments/assets/a37c9813-4a1e-4ed7-9d37-3efa6c1a6a5d" />

<img width="1244" height="343" alt="image" src="https://github.com/user-attachments/assets/336b0fc8-fa3b-451a-aa59-12d074ba0b95" />

<img width="571" height="275" alt="image" src="https://github.com/user-attachments/assets/517280b1-7448-4d3b-9aa3-6096298f486a" />

That is about a 25x speed up, because of the warp level computation and one warp reduction, rather than letting one thread handle a row.

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
