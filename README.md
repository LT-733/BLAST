# BLAST
BLAS (Basic Linear Algebra Subprograms) is the backbone of a lot of the modern mathematical libraies, such as LAPACK, NumPy, etc. Although using those will be much easier (No, I do not endorse or encourage any use of FORTRAN,
although learning it will be pretty cool), I feel like to be able to understand how the math works (efficiently) underneath, is vital to how I can apply those modern libraries effectively. Hence, 
I wrote some CUDA kernels to execute these linear algebraic subroutines. BLAST's T is because I used a lot of template programming, but later on, I dropped it in favor of strict half precision.
