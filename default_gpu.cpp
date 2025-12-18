#include <mpi.h>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <sys/time.h>
using namespace std;

#define warpSize 64

__inline__ __device__ double warpReduceSum(double v) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) 
        v += __shfl_down(v, offset);
    return v;
}

__host__ __device__ inline int idx(int i, int j, int k, int nx, int ny, int nz) {
    return (k * ny + j) * nx + i; // in the enlarged block
}

__host__ __device__ inline double exact(double x, double y, double z){
    const double PI = 3.14159265358979323846; return sin(PI * x) * cos(PI * y) * sin(PI * z); //true solution
}

inline double rhs(double x, double y, double z){
    const double PI = 3.14159265358979323846;   return -3.0 * PI * PI * exact(x, y, z); //the value of f
}

void init_local_with_bc(double* u, double* f, int nx, int ny, int nz, int local_nx, int local_ny, int local_nz, 
                        int N, int coords[3], int dims[3]) // enlarged block
{
    double h = 1.0 / (N + 1); // N = # interior pts
    int gx0 = coords[0] * local_nx + 1; int gy0 = coords[1] * local_ny + 1; int gz0 = coords[2] * local_nz + 1;
        // starting coordinates
    for (int k = 1; k <= local_nz; k++){
        int K = gz0 + k - 1;    double z = K * h;
        for (int j = 1; j <= local_ny; j++){
            int J = gy0 + j - 1;    double y = J * h;
            for (int i = 1; i <= local_nx; i++){
                int I = gx0 + i - 1;    double x = I * h;
                int id = idx(i, j, k, nx, ny, nz);  u[id] = 0.0;  f[id] = rhs(x, y, z); //all in the enlarged block
            }
        }
    } // tackle all interior pts

    if (coords[0] == 0){ // x- boundary
        for (int k = 1; k <= local_nz; k++){
            double z = (gz0 + k - 1) * h;
            for (int j = 1; j <= local_ny; j++){
                double y = (gy0 + j - 1) * h;  u[idx(0, j, k, nx, ny, nz)] = exact(0.0, y, z); //only u
            }
        }
    }
    if (coords[0] == dims[0] - 1){ //x+ boundary
        for (int k = 1; k <= local_nz; k++){
            double z = (gz0 + k - 1) * h;
            for (int j = 1; j <= local_ny; j++){
                double y = (gy0 + j - 1) * h;  u[idx(local_nx + 1, j, k, nx, ny, nz)] = exact(1.0, y, z);
            }
        }
    }
    if (coords[1] == 0){ // y-
        for (int k = 1; k <= local_nz; k++){
            double z = (gz0 + k - 1) * h;
            for (int i = 1; i <= local_nx; i++){
                double x = (gx0 + i - 1) * h;  u[idx(i, 0, k, nx, ny, nz)] = exact(x, 0.0, z);
            }
        }
    }
    if (coords[1] == dims[1] - 1){ // y+
        for (int k = 1; k <= local_nz; k++){
            double z = (gz0 + k - 1) * h;
            for (int i = 1; i <= local_nx; i++){
                double x = (gx0 + i - 1) * h;  u[idx(i, local_ny + 1, k, nx, ny, nz)] = exact(x, 1.0, z);
            }
        }
    }
    if (coords[2] == 0){ // z-
        for (int j = 1; j <= local_ny; j++){
            double y = (gy0 + j - 1) * h;
            for (int i = 1; i <= local_nx; i++){
                double x = (gx0 + i - 1) * h;   u[idx(i, j, 0, nx, ny, nz)] = exact(x, y, 0.0);
            }
        }
    }
    if (coords[2] == dims[2] - 1){ // z+
        for (int j = 1; j <= local_ny; j++){
            double y = (gy0 + j - 1) * h;
            for (int i = 1; i <= local_nx; i++){
                double x = (gx0 + i - 1) * h;  u[idx(i, j, local_nz + 1, nx, ny, nz)] = exact(x, y, 1.0);
            }
        }
    }
}

__global__ void pack_all_faces(const double* u_d, double* buf_xm, double* buf_xp, double* buf_ym, double* buf_yp,
                               double* buf_zm, double* buf_zp, int local_nx, int local_ny, int local_nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  int j = blockIdx.y * blockDim.y + threadIdx.y; // interior
    int k = blockIdx.z * blockDim.z + threadIdx.z; // interior
    if(i >= local_nx || j >= local_ny || k >= local_nz) return;
    int Nx = local_nx + 2; int Ny = local_ny + 2;
    
    if(i == 0){
        int src = (k+1)*Ny*Nx + (j+1)*Nx + 1;  int dst = k * local_ny + j;
        buf_xm[dst] = u_d[src];
    }

    if(i == local_nx - 1){
        int src = (k+1)*Ny*Nx + (j+1)*Nx + local_nx;  int dst = k * local_ny + j;
        buf_xp[dst] = u_d[src];
    }

    if(j == 0){
        int src = (k+1)*Ny*Nx + 1*Nx + (i+1);  int dst = k * local_nx + i;
        buf_ym[dst] = u_d[src];
    }

    if(j == local_ny - 1){
        int src = (k+1)*Ny*Nx + local_ny*Nx + (i+1);  int dst = k * local_nx + i;
        buf_yp[dst] = u_d[src];
    }

    if(k == 0){
        int src = 1*Ny*Nx + (j+1)*Nx + (i+1);  int dst = j * local_nx + i;
        buf_zm[dst] = u_d[src];
    }

    if(k == local_nz - 1){
        int src = local_nz*Ny*Nx + (j+1)*Nx + (i+1);  int dst = j * local_nx + i;
        buf_zp[dst] = u_d[src];
    }
}

__global__ void unpack_all_faces(double* u_d, double* buf_xm, double* buf_xp, double* buf_ym, double* buf_yp,
                               double* buf_zm, double* buf_zp, int flagxm, int flagxp, int flagym, int flagyp, 
                               int flagzm, int flagzp, int local_nx, int local_ny, int local_nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;  int j = blockIdx.y * blockDim.y + threadIdx.y; // interior
    int k = blockIdx.z * blockDim.z + threadIdx.z; // interior
    if(i >= local_nx || j >= local_ny || k >= local_nz) return;
    int Nx = local_nx + 2; int Ny = local_ny + 2;

    if (flagxm == 1 && i == 0)
        u_d[(k+1)*Ny*Nx + (j+1)*Nx + 0] = buf_xm[k*local_ny + j];

    if (flagxp == 1 && i == local_nx - 1)
        u_d[(k+1)*Ny*Nx + (j+1)*Nx + (local_nx+1)] = buf_xp[k*local_ny + j];

    if (flagym == 1 && j == 0)
        u_d[(k+1)*Ny*Nx + 0*Nx + (i+1)] = buf_ym[k*local_nx + i];

    if (flagyp == 1 && j == local_ny - 1)
        u_d[(k+1)*Ny*Nx + (local_ny+1)*Nx + (i+1)] = buf_yp[k*local_nx + i];

    if (flagzm == 1 && k == 0)
        u_d[0*Ny*Nx + (j+1)*Nx + (i+1)] = buf_zm[j*local_nx + i];

    if (flagzp == 1 && k == local_nz - 1)
        u_d[(local_nz+1)*Ny*Nx + (j+1)*Nx + (i+1)] = buf_zp[j*local_nx + i];
}

__global__ void sor_kernel(double* u, const double* f, int nx, int ny, int nz, int local_nx, int local_ny, int local_nz,
                           int gx0, int gy0, int gz0, double h2, double omega, int color) // u is enlarged array
{
    extern __shared__ double sh[]; // the "enlarged block"

    int tx = threadIdx.x;  int ty = threadIdx.y;  int tz = threadIdx.z; //labels of pts in the interior  +1 is the "enlarged"

    int i = blockIdx.x * blockDim.x + tx + 1;   int I = gx0 + i - 1;
    int j = blockIdx.y * blockDim.y + ty + 1;   int J = gy0 + j - 1;
    int k = blockIdx.z * blockDim.z + tz + 1;   int K = gz0 + k - 1;

    if (i > local_nx || j > local_ny || k > local_nz) return; //not in the interior

    int Sx = blockDim.x + 2;  int Sy = blockDim.y + 2;  int slice = Sx * Sy;
    int S = (tz + 1) * slice + (ty + 1) * Sx + (tx + 1);
    int id = idx(i, j, k, nx, ny, nz); //in the "enlarged block"
    sh[S] = u[id];

    if (tx == 0)  sh[S - 1] = u[idx(i - 1, j, k, nx, ny, nz)];
    if (tx == blockDim.x - 1)  sh[S + 1] = u[idx(i + 1, j, k, nx, ny, nz)];
    if (ty == 0)  sh[S - Sx] = u[idx(i, j - 1, k, nx, ny, nz)];
    if (ty == blockDim.y - 1)  sh[S + Sx] = u[idx(i, j + 1, k, nx, ny, nz)];
    if (tz == 0)  sh[S - slice] = u[idx(i, j, k - 1, nx, ny, nz)];
    if (tz == blockDim.z - 1)  sh[S + slice] = u[idx(i, j, k + 1, nx, ny, nz)]; //halo exchange for each thread-block
    __syncthreads();

    if ((I + J + K) % 2 != color) return;

    double sum_nb = sh[S - 1] + sh[S + 1] + sh[S - Sx] + sh[S + Sx] + sh[S - slice] + sh[S + slice];
    double u_new = (sum_nb - h2 * f[id]) / 6.0;  u[id] = (1.0 - omega) * sh[S] + omega * u_new;
}


__global__ void residual_kernel(const double* u, const double* f, int nx, int ny, int nz,
                     int local_nx, int local_ny, int local_nz, double h2, double* block_sums)
{
    int tx = threadIdx.x;  int ty = threadIdx.y;  int tz = threadIdx.z;
    int i = blockIdx.x * blockDim.x + tx + 1;  int j = blockIdx.y * blockDim.y + ty + 1;  int k = blockIdx.z * blockDim.z + tz + 1;

    int tid = tx + blockDim.x * (ty + blockDim.y * tz);
    int nthreads = blockDim.x * blockDim.y * blockDim.z;
    int lane   = tid % warpSize;      int warpId = tid / warpSize;          
    int nWarps = (nthreads + warpSize - 1) / warpSize;

    extern __shared__ double sh[]; 

    int Sx = blockDim.x + 2; int Sy = blockDim.y + 2; int Sz = blockDim.z + 2; int slice = Sx * Sy;
    int tile_size = Sx * Sy * Sz;   

    double* tile_u    = sh;               
    double* warp_sums = sh + tile_size;    
    int S = (tz + 1) * slice + (ty + 1) * Sx + (tx + 1);

   
    if (i > local_nx || j > local_ny || k > local_nz) return;
    int id = idx(i, j, k, nx, ny, nz);

    tile_u[S] = u[id];

    if (tx == 0)              tile_u[S - 1] = u[idx(i - 1, j, k, nx, ny, nz)];
    if (tx == blockDim.x - 1) tile_u[S + 1] = u[idx(i + 1, j, k, nx, ny, nz)];
    if (ty == 0)             tile_u[S - Sx] = u[idx(i, j - 1, k, nx, ny, nz)];
    if (ty == blockDim.y - 1) tile_u[S + Sx] = u[idx(i, j + 1, k, nx, ny, nz)];
    if (tz == 0) tile_u[S - slice] = u[idx(i, j, k - 1, nx, ny, nz)];
    if (tz == blockDim.z - 1) tile_u[S + slice] = u[idx(i, j, k + 1, nx, ny, nz)];
    __syncthreads();

    double val = 0.0;

    if (i <= local_nx && j <= local_ny && k <= local_nz) {
        double uc = tile_u[S];
        double sum_nb = tile_u[S - 1] + tile_u[S + 1] + tile_u[S - Sx] + tile_u[S + Sx]
                        + tile_u[S - slice]  + tile_u[S + slice];
        double lap = (sum_nb - 6.0 * uc) / h2; double r = f[id] - lap;    val = r * r;
    }

    val = warpReduceSum(val);

    if (lane == 0 && warpId < nWarps) warp_sums[warpId] = val;
    __syncthreads();

    double blockSum = 0.0;
    if (warpId == 0) {
        blockSum = (lane < nWarps) ? warp_sums[lane] : 0.0;  blockSum = warpReduceSum(blockSum);
        if (lane == 0) {
            int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
            block_sums[bid] = blockSum;
        }
    }
}

__global__ void error_kernel(const double* u, int nx, int ny, int nz, int local_nx, int local_ny, int local_nz,
                  int gx0, int gy0, int gz0, double h, double* block_sums)
{
    extern __shared__ double sh[]; // only include interior points and put inside the residual

    int tx = threadIdx.x;  int ty = threadIdx.y;  int tz = threadIdx.z; //local coordinates in "interior block"

    int i = blockIdx.x * blockDim.x + tx + 1;  int j = blockIdx.y * blockDim.y + ty + 1;
    int k = blockIdx.z * blockDim.z + tz + 1; //after adding two halos so must be interior pts

    int tid = tx + blockDim.x * (ty + blockDim.y * tz); //place of the thread in the "interior array"
    int nthreads = blockDim.x * blockDim.y * blockDim.z;
    int lane   = tid % warpSize;      int warpId = tid / warpSize;            // block 内第几个 warp
    int nWarps = (nthreads + warpSize - 1) / warpSize;
    double val = 0.0;

    if (i <= local_nx && j <= local_ny && k <= local_nz){ // make sure they're interior pts
        int id = idx(i, j, k, nx, ny, nz); //in the "enlarged array"

        int I = gx0 + i - 1; int J = gy0 + j - 1; int K = gz0 + k - 1;
        double x = I * h;  double y = J * h;   double z = K * h;  double u_ex = exact(x, y, z);
        double diff = u[id] - u_ex;   val = diff * diff;
    }

    val = warpReduceSum(val);

    extern __shared__ double warp_sums[]; 
    if (lane == 0 && warpId < nWarps)  warp_sums[warpId] = val;

    __syncthreads();

    double blockSum = 0.0;
    if (warpId == 0) {
        blockSum = (lane < nWarps) ? warp_sums[lane] : 0.0; blockSum = warpReduceSum(blockSum);
        if (lane == 0) {
            int bid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
            block_sums[bid] = blockSum;
        }
    }
}

double local_residual_sq(double* u_d, double* f_d,
                         int nx, int ny, int nz,
                         int local_nx, int local_ny, int local_nz,
                         double h2)
{
    dim3 block(64, 4, 2);
    dim3 grid((local_nx + block.x - 1) / block.x, (local_ny + block.y - 1) / block.y,
              (local_nz + block.z - 1) / block.z);

    int num_blocks = grid.x * grid.y * grid.z;  int nthreads = block.x * block.y * block.z;
    int nWarps  = (nthreads + warpSize - 1) / warpSize;

    int Sx = block.x + 2; int Sy = block.y + 2; int Sz = block.z + 2; int tile_size = Sx * Sy * Sz;
    size_t shmem = (size_t)(tile_size + nWarps) * sizeof(double);

    double* d_block_sums = nullptr;  hipMalloc(&d_block_sums, num_blocks * sizeof(double));

    residual_kernel<<<grid, block, shmem>>>(u_d, f_d, nx, ny, nz, local_nx, local_ny, local_nz,
                                            h2, d_block_sums);
    hipDeviceSynchronize();

    double* h_block_sums = new double[num_blocks];
    hipMemcpy(h_block_sums, d_block_sums, num_blocks * sizeof(double), hipMemcpyDeviceToHost);

    double res2_local = 0.0;
    for (int i = 0; i < num_blocks; i++) res2_local += h_block_sums[i];

    delete [] h_block_sums; hipFree(d_block_sums); return res2_local;
}

double local_error_sq(double* u_d,
                      int nx, int ny, int nz,
                      int local_nx, int local_ny, int local_nz,
                      int N, int coords[3], int dims[3])
{
    if (local_nx <= 0 || local_ny <= 0 || local_nz <= 0)
        return 0.0;

    double h = 1.0 / (N + 1);
    int gx0 = coords[0] * local_nx + 1;
    int gy0 = coords[1] * local_ny + 1;
    int gz0 = coords[2] * local_nz + 1;

    dim3 block(64, 4, 2);
    dim3 grid((local_nx + block.x - 1) / block.x,
              (local_ny + block.y - 1) / block.y,
              (local_nz + block.z - 1) / block.z);

    int num_blocks = grid.x * grid.y * grid.z;  int nthreads = block.x * block.y * block.z;
    int nWarps  = (nthreads + warpSize - 1) / warpSize;
    size_t shmem = nWarps * sizeof(double);

    double* d_block_sums = nullptr; hipMalloc(&d_block_sums, num_blocks * sizeof(double));

    error_kernel<<<grid, block, shmem>>>(u_d, nx, ny, nz, local_nx, local_ny, local_nz,
                                         gx0, gy0, gz0, h, d_block_sums);
    hipDeviceSynchronize();

    double* h_block_sums = new double[num_blocks];
    hipMemcpy(h_block_sums, d_block_sums, num_blocks * sizeof(double), hipMemcpyDeviceToHost);

    double err2_local = 0.0;
    for (int i = 0; i < num_blocks; i++)
        err2_local += h_block_sums[i];

    delete [] h_block_sums;  hipFree(d_block_sums); return err2_local;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_size, rank;   MPI_Comm_size(MPI_COMM_WORLD, &world_size);  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = atoi(argv[1]);    int max_iter = atoi(argv[2]);    double omega = atof(argv[3]);

    int dims[3] = {0, 0, 0};   MPI_Dims_create(world_size, 3, dims);

    int periods[3] = {0, 0, 0};    MPI_Comm cart_comm;    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm);

    int coords[3];    MPI_Cart_coords(cart_comm, rank, 3, coords);

    int local_nx = N / dims[0];   int local_ny = N / dims[1];    int local_nz = N / dims[2];

    int nx = local_nx + 2;   int ny = local_ny + 2;    int nz = local_nz + 2;    int local_size = nx * ny * nz; //enlarged

    double* u_h = new double[local_size];      double* f_h = new double[local_size];
    for (int i = 0; i < local_size; i++){ u_h[i] = 0.0;  f_h[i] = 0.0; } 

    int nbr_xm, nbr_xp, nbr_ym, nbr_yp, nbr_zm, nbr_zp;

    MPI_Cart_shift(cart_comm, 0, 1, &nbr_xm, &nbr_xp);
    MPI_Cart_shift(cart_comm, 1, 1, &nbr_ym, &nbr_yp);
    MPI_Cart_shift(cart_comm, 2, 1, &nbr_zm, &nbr_zp); //denote the rank of adjacent mpi blocks

    init_local_with_bc(u_h, f_h, nx, ny, nz, local_nx, local_ny, local_nz, N, coords, dims);

    double h  = 1.0 / (N + 1);    double h2 = h * h;
    if (rank == 0){
        cout << "N = " << N << ", world_size = " << world_size
             << ", dims = (" << dims[0] << "," << dims[1] << "," << dims[2] << ")\n";
    }

    double* u_d;   double* f_d; 
    hipMalloc(&u_d, local_size * sizeof(double));    hipMalloc(&f_d, local_size * sizeof(double));
    hipMemcpy(u_d, u_h, local_size * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(f_d, f_h, local_size * sizeof(double), hipMemcpyHostToDevice); //the only memcpy

    const double tol = 1e-4;     double res2_global = 0.0;
    int count_x = local_ny * local_nz;  int count_y = local_nx * local_nz;  int count_z = local_nx * local_ny;

    double *send_xm_h, *send_xp_h, *recv_xm_h, *recv_xp_h, *send_xm_d, *send_xp_d, *recv_xm_d, *recv_xp_d;
    double *send_ym_h, *send_yp_h, *recv_ym_h, *recv_yp_h, *send_ym_d, *send_yp_d, *recv_ym_d, *recv_yp_d;
    double *send_zm_h, *recv_zm_h, *send_zp_h, *recv_zp_h, *send_zm_d, *recv_zm_d, *send_zp_d, *recv_zp_d;
    hipHostAlloc(&send_xm_h, count_x*sizeof(double), hipHostMallocDefault);
    hipHostAlloc(&recv_xm_h, count_x*sizeof(double), hipHostMallocDefault);
    hipHostAlloc(&send_xp_h, count_x*sizeof(double), hipHostMallocDefault);
    hipHostAlloc(&recv_xp_h, count_x*sizeof(double), hipHostMallocDefault);
    hipHostAlloc(&send_ym_h, count_y*sizeof(double), hipHostMallocDefault);
    hipHostAlloc(&recv_ym_h, count_y*sizeof(double), hipHostMallocDefault);
    hipHostAlloc(&send_yp_h, count_y*sizeof(double), hipHostMallocDefault);
    hipHostAlloc(&recv_yp_h, count_y*sizeof(double), hipHostMallocDefault);
    hipHostAlloc(&send_zm_h, count_z*sizeof(double), hipHostMallocDefault);
    hipHostAlloc(&recv_zm_h, count_z*sizeof(double), hipHostMallocDefault);
    hipHostAlloc(&send_zp_h, count_z*sizeof(double), hipHostMallocDefault);
    hipHostAlloc(&recv_zp_h, count_z*sizeof(double), hipHostMallocDefault); //pinned memory for halo exchange

    size_t sz_x = (size_t)count_x * sizeof(double); size_t sz_y = (size_t)count_y * sizeof(double); 
    size_t sz_z = (size_t)count_z * sizeof(double);

    hipMalloc(&send_xm_d, sz_x); hipMalloc(&recv_xm_d, sz_x); hipMalloc(&send_xp_d, sz_x); hipMalloc(&recv_xp_d, sz_x);
    hipMalloc(&send_ym_d, sz_y); hipMalloc(&recv_ym_d, sz_y); hipMalloc(&send_yp_d, sz_y); hipMalloc(&recv_yp_d, sz_y);
    hipMalloc(&send_zm_d, sz_z); hipMalloc(&recv_zm_d, sz_z); hipMalloc(&send_zp_d, sz_z); hipMalloc(&recv_zp_d, sz_z);

    dim3 blockpack(64,4,2);   dim3 gridpack((local_nx+blockpack.x-1)/blockpack.x, 
                                       (local_ny+blockpack.y-1)/blockpack.y, (local_nz+blockpack.z-1)/blockpack.z);
    dim3 blockunpack(64,4,2);  dim3 gridunpack((local_nx+blockunpack.x-1)/blockpack.x, 
                                    (local_ny+blockunpack.y-1)/blockunpack.y, (local_nz+blockunpack.z-1)/blockunpack.z);

    int gx0 = coords[0] * local_nx + 1;   int gy0 = coords[1] * local_ny + 1;  
    int gz0 = coords[2] * local_nz + 1; //starting point of each mpi block

    dim3 sorblock(64, 4, 2);
    dim3 sorgrid((local_nx + sorblock.x - 1) / sorblock.x, (local_ny + sorblock.y - 1) / sorblock.y, 
                 (local_nz + sorblock.z - 1) / sorblock.z);
              
    int Sx = sorblock.x + 2;   int Sy = sorblock.y + 2;   int Sz = sorblock.z + 2; // # threads in each "enlarged thread-block"
    size_t shmem = (size_t)Sx * Sy * Sz * sizeof(double); //the size of each shared memory

    int flagxm=0, flagxp=0, flagym=0, flagyp=0, flagzm=0, flagzp=0;
    if(nbr_xm != MPI_PROC_NULL) flagxm = 1;  if(nbr_xp != MPI_PROC_NULL) flagxp = 1;
    if(nbr_ym != MPI_PROC_NULL) flagym = 1;  if(nbr_yp != MPI_PROC_NULL) flagyp = 1;
    if(nbr_zm != MPI_PROC_NULL) flagzm = 1;  if(nbr_zp != MPI_PROC_NULL) flagzp = 1;

    hipEvent_t start, stop;  hipEventCreate(&start);  hipEventCreate(&stop);

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    for (int iter = 0; iter < max_iter; iter++){

        pack_all_faces<<<gridpack, blockpack>>>(u_d, send_xm_d, send_xp_d, send_ym_d, send_yp_d, send_zm_d, send_zp_d,
                                            local_nx, local_ny, local_nz);

        hipMemcpy(send_xm_h, send_xm_d, sz_x, hipMemcpyDeviceToHost); hipMemcpy(send_xp_h, send_xp_d, sz_x, hipMemcpyDeviceToHost);
        hipMemcpy(send_ym_h, send_ym_d, sz_y, hipMemcpyDeviceToHost); hipMemcpy(send_yp_h, send_yp_d, sz_y, hipMemcpyDeviceToHost);
        hipMemcpy(send_zm_h, send_zm_d, sz_z, hipMemcpyDeviceToHost); hipMemcpy(send_zp_h, send_zp_d, sz_z, hipMemcpyDeviceToHost);

        MPI_Request reqs[12];  int rcount = 0;

        if (nbr_xm != MPI_PROC_NULL) {
           MPI_Irecv(recv_xm_h, count_x, MPI_DOUBLE, nbr_xm, 101, cart_comm, &reqs[rcount++]);
           MPI_Isend(send_xm_h, count_x, MPI_DOUBLE, nbr_xm, 102, cart_comm, &reqs[rcount++]);
        }
       if (nbr_xp != MPI_PROC_NULL) {
        MPI_Irecv(recv_xp_h, count_x, MPI_DOUBLE, nbr_xp, 102, cart_comm, &reqs[rcount++]);
        MPI_Isend(send_xp_h, count_x, MPI_DOUBLE, nbr_xp, 101, cart_comm, &reqs[rcount++]);
       }
        if (nbr_ym != MPI_PROC_NULL) {
        MPI_Irecv(recv_ym_h, count_y, MPI_DOUBLE, nbr_ym, 201, cart_comm, &reqs[rcount++]);
        MPI_Isend(send_ym_h, count_y, MPI_DOUBLE, nbr_ym, 202, cart_comm, &reqs[rcount++]);
        }
        if (nbr_yp != MPI_PROC_NULL) {
        MPI_Irecv(recv_yp_h, count_y, MPI_DOUBLE, nbr_yp, 202, cart_comm, &reqs[rcount++]);
        MPI_Isend(send_yp_h, count_y, MPI_DOUBLE, nbr_yp, 201, cart_comm, &reqs[rcount++]);
       }
        if (nbr_zm != MPI_PROC_NULL) {
        MPI_Irecv(recv_zm_h, count_z, MPI_DOUBLE, nbr_zm, 301, cart_comm, &reqs[rcount++]);
        MPI_Isend(send_zm_h, count_z, MPI_DOUBLE, nbr_zm, 302, cart_comm, &reqs[rcount++]);
       }
        if (nbr_zp != MPI_PROC_NULL) {
        MPI_Irecv(recv_zp_h, count_z, MPI_DOUBLE, nbr_zp, 302, cart_comm, &reqs[rcount++]);
        MPI_Isend(send_zp_h, count_z, MPI_DOUBLE, nbr_zp, 301, cart_comm, &reqs[rcount++]);
        }
        if (rcount > 0) MPI_Waitall(rcount, reqs, MPI_STATUSES_IGNORE);

        hipDeviceSynchronize();


        if (nbr_xm != MPI_PROC_NULL) hipMemcpy(recv_xm_d, recv_xm_h, sz_x, hipMemcpyHostToDevice);
        if (nbr_xp != MPI_PROC_NULL) hipMemcpy(recv_xp_d, recv_xp_h, sz_x, hipMemcpyHostToDevice);
        if (nbr_ym != MPI_PROC_NULL) hipMemcpy(recv_ym_d, recv_ym_h, sz_y, hipMemcpyHostToDevice);
        if (nbr_yp != MPI_PROC_NULL) hipMemcpy(recv_yp_d, recv_yp_h, sz_y, hipMemcpyHostToDevice);
        if (nbr_zm != MPI_PROC_NULL) hipMemcpy(recv_zm_d, recv_zm_h, sz_z, hipMemcpyHostToDevice);
        if (nbr_zp != MPI_PROC_NULL) hipMemcpy(recv_zp_d, recv_zp_h, sz_z, hipMemcpyHostToDevice);
        

        unpack_all_faces<<<gridunpack, blockunpack>>>(u_d, recv_xm_d, recv_xp_d, recv_ym_d, recv_yp_d, recv_zm_d, recv_zp_d, flagxm, flagxp, flagym, 
                         flagyp, flagzm, flagzp, local_nx, local_ny, local_nz);
        hipDeviceSynchronize();

        hipEventRecord(start, 0);
        sor_kernel<<<sorgrid, sorblock, shmem>>>(u_d, f_d, nx, ny, nz, local_nx, local_ny, local_nz,
                                           gx0, gy0, gz0, h2, omega, 0);
        hipEventRecord(stop, 0);  hipEventSynchronize(stop);

        pack_all_faces<<<gridpack, blockpack>>>(u_d, send_xm_d, send_xp_d, send_ym_d, send_yp_d, send_zm_d, send_zp_d,
                                                local_nx, local_ny, local_nz);

        hipMemcpy(send_xm_h, send_xm_d, sz_x, hipMemcpyDeviceToHost); hipMemcpy(send_xp_h, send_xp_d, sz_x, hipMemcpyDeviceToHost);
        hipMemcpy(send_ym_h, send_ym_d, sz_y, hipMemcpyDeviceToHost); hipMemcpy(send_yp_h, send_yp_d, sz_y, hipMemcpyDeviceToHost);
        hipMemcpy(send_zm_h, send_zm_d, sz_z, hipMemcpyDeviceToHost); hipMemcpy(send_zp_h, send_zp_d, sz_z, hipMemcpyDeviceToHost);

        MPI_Request reqs2[12];  int rcount2 = 0;

        if (nbr_xm != MPI_PROC_NULL) {
           MPI_Irecv(recv_xm_h, count_x, MPI_DOUBLE, nbr_xm, 101, cart_comm, &reqs2[rcount2++]);
           MPI_Isend(send_xm_h, count_x, MPI_DOUBLE, nbr_xm, 102, cart_comm, &reqs2[rcount2++]);
        }
       if (nbr_xp != MPI_PROC_NULL) {
        MPI_Irecv(recv_xp_h, count_x, MPI_DOUBLE, nbr_xp, 102, cart_comm, &reqs2[rcount2++]);
        MPI_Isend(send_xp_h, count_x, MPI_DOUBLE, nbr_xp, 101, cart_comm, &reqs2[rcount2++]);
       }
        if (nbr_ym != MPI_PROC_NULL) {
        MPI_Irecv(recv_ym_h, count_y, MPI_DOUBLE, nbr_ym, 201, cart_comm, &reqs2[rcount2++]);
        MPI_Isend(send_ym_h, count_y, MPI_DOUBLE, nbr_ym, 202, cart_comm, &reqs2[rcount2++]);
        }
        if (nbr_yp != MPI_PROC_NULL) {
        MPI_Irecv(recv_yp_h, count_y, MPI_DOUBLE, nbr_yp, 202, cart_comm, &reqs2[rcount2++]);
        MPI_Isend(send_yp_h, count_y, MPI_DOUBLE, nbr_yp, 201, cart_comm, &reqs2[rcount2++]);
       }
        if (nbr_zm != MPI_PROC_NULL) {
        MPI_Irecv(recv_zm_h, count_z, MPI_DOUBLE, nbr_zm, 301, cart_comm, &reqs2[rcount2++]);
        MPI_Isend(send_zm_h, count_z, MPI_DOUBLE, nbr_zm, 302, cart_comm, &reqs2[rcount2++]);
       }
        if (nbr_zp != MPI_PROC_NULL) {
        MPI_Irecv(recv_zp_h, count_z, MPI_DOUBLE, nbr_zp, 302, cart_comm, &reqs2[rcount2++]);
        MPI_Isend(send_zp_h, count_z, MPI_DOUBLE, nbr_zp, 301, cart_comm, &reqs2[rcount2++]);
        }
        if (rcount2 > 0) MPI_Waitall(rcount2, reqs2, MPI_STATUSES_IGNORE);

        hipDeviceSynchronize();

        if (nbr_xm != MPI_PROC_NULL) hipMemcpy(recv_xm_d, recv_xm_h, sz_x, hipMemcpyHostToDevice);
        if (nbr_xp != MPI_PROC_NULL) hipMemcpy(recv_xp_d, recv_xp_h, sz_x, hipMemcpyHostToDevice);
        if (nbr_ym != MPI_PROC_NULL) hipMemcpy(recv_ym_d, recv_ym_h, sz_y, hipMemcpyHostToDevice);
        if (nbr_yp != MPI_PROC_NULL) hipMemcpy(recv_yp_d, recv_yp_h, sz_y, hipMemcpyHostToDevice);
        if (nbr_zm != MPI_PROC_NULL) hipMemcpy(recv_zm_d, recv_zm_h, sz_z, hipMemcpyHostToDevice);
        if (nbr_zp != MPI_PROC_NULL) hipMemcpy(recv_zp_d, recv_zp_h, sz_z, hipMemcpyHostToDevice);

        unpack_all_faces<<<gridunpack, blockunpack>>>(u_d, recv_xm_d, recv_xp_d, recv_ym_d, recv_yp_d, recv_zm_d, recv_zp_d, flagxm, flagxp, flagym, 
                         flagyp, flagzm, flagzp, local_nx, local_ny, local_nz);
        hipDeviceSynchronize();

        sor_kernel<<<sorgrid, sorblock, shmem>>>(u_d, f_d, nx, ny, nz, local_nx, local_ny, local_nz,
                                           gx0, gy0, gz0, h2, omega, 1);
        hipDeviceSynchronize();

        double res2_local = local_residual_sq(u_d, f_d, nx, ny, nz, local_nx, local_ny, local_nz, h2);
        hipDeviceSynchronize();
        MPI_Allreduce(&res2_local, &res2_global, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
        double res = sqrt(res2_global);

        double err2_local = local_error_sq(u_d, nx, ny, nz, local_nx, local_ny, local_nz, N, coords, dims);
        double err2_global = 0.0;
        MPI_Allreduce(&err2_local, &err2_global, 1, MPI_DOUBLE, MPI_SUM, cart_comm);
        double err = sqrt(err2_global * h * h * h);

        int local_stop = (res < tol ? 1 : 0);  int global_stop = 0;
        MPI_Allreduce(&local_stop, &global_stop, 1, MPI_INT, MPI_MIN, cart_comm);
        if(global_stop == 1) 
        {
            cout << "iterations " << iter << endl;  break;
        }
            
        if (iter % 50 == 0 || iter == max_iter - 1){
            if (rank == 0) {
            float ms;   hipEventElapsedTime(&ms, start, stop); // ms 
            cout << "iter " << iter << ", residual L2 = " << res << endl;
            cout << "SOR update kernel time = " << ms << " ms\n" << endl;
            cout << " error L2 = " << err << endl;
        }
    }
}
    hipEventDestroy(start);          hipEventDestroy(stop);
    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) * 1e-6;
    cout << "Total time = " << elapsed << endl;

    hipFree(u_d);   hipFree(f_d);   delete [] u_h;   delete [] f_h;
    MPI_Comm_free(&cart_comm);   MPI_Finalize();   return 0;
}
