#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <iostream>

inline int idx(int i, int j, int k, int nx, int ny) {
    return (k * ny + j) * nx + i;
}

inline double u_exact(double x, double y, double z) {
    return std::sin(M_PI * x) * std::cos(M_PI * y) * std::sin(M_PI * z);
}

inline double f_rhs(double x, double y, double z) {
    return 3.0 * M_PI * M_PI * u_exact(x, y, z);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    const int Nx = 256;     
    const int Ny = 256;
    const int Nz = 256;
    const int max_iter = 10000;
    const double tol = 1e-4;
    const double omega = 1.8;

    const double Lx = 1.0;
    const double Ly = 1.0;
    const double Lz = 1.0;

    const double hx = Lx / (Nx - 1);
    const double hy = Ly / (Ny - 1);
    const double hz = Lz / (Nz - 1);
    const double h2 = hx * hx;  
    const double vol =  hx * hy * hz;

    if (std::fabs(hx - hy) > 1e-12 || std::fabs(hx - hz) > 1e-12) {
        if (rank == 0)
            std::cerr << "This code assumes hx = hy = hz.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int nz_base = Nz / nprocs;
    int remainder = Nz % nprocs;

    int nz_local;     
    int k_start;       
    if (rank < remainder) {
        nz_local = nz_base + 1;
        k_start = rank * nz_local;
    } else {
        nz_local = nz_base;
        k_start = remainder * (nz_base + 1) + (rank - remainder) * nz_base;
    }
    int k_end = k_start + nz_local - 1;

    const int nz_with_halo = nz_local + 2;
    const int local_size = Nx * Ny * nz_with_halo;

    std::vector<double> u(local_size, 0.0);      
    std::vector<double> f(local_size, 0.0);     

    for (int local_k = 1; local_k <= nz_local; ++local_k) {
        int k = k_start + (local_k - 1); 
        double z = k * hz;
        for (int j = 0; j < Ny; ++j) {
            double y = j * hy;
            for (int i = 0; i < Nx; ++i) {
                double x = i * hx;
                int id = idx(i, j, local_k, Nx, Ny);
                f[id] = f_rhs(x, y, z);

                if (i == 0 || i == Nx - 1 ||
                    j == 0 || j == Ny - 1 ||
                    k == 0 || k == Nz - 1) {
                    u[id] = u_exact(x, y, z);
                }
            }
        }
    }

    int prev_rank = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int next_rank = (rank == nprocs - 1) ? MPI_PROC_NULL : rank + 1;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    double global_res2 = 0.0;
    double global_err2 = 0.0;
    int iter = 0;

    for (iter = 0; iter < max_iter; ++iter) {
        double local_res2 = 0.0;
        double local_err2 = 0.0;

        for (int color = 0; color < 2; ++color) {

            MPI_Request reqs[4];
            int nreq = 0;

            if (prev_rank != MPI_PROC_NULL) {
                MPI_Irecv(&u[idx(0, 0, 0, Nx, Ny)],
                          Nx * Ny, MPI_DOUBLE,
                          prev_rank, 100, MPI_COMM_WORLD, &reqs[nreq++]);
                MPI_Isend(&u[idx(0, 0, 1, Nx, Ny)],
                          Nx * Ny, MPI_DOUBLE,
                          prev_rank, 101, MPI_COMM_WORLD, &reqs[nreq++]);
            }
            if (next_rank != MPI_PROC_NULL) {
                MPI_Irecv(&u[idx(0, 0, nz_local + 1, Nx, Ny)],
                          Nx * Ny, MPI_DOUBLE,
                          next_rank, 101, MPI_COMM_WORLD, &reqs[nreq++]);
                MPI_Isend(&u[idx(0, 0, nz_local, Nx, Ny)],
                          Nx * Ny, MPI_DOUBLE,
                          next_rank, 100, MPI_COMM_WORLD, &reqs[nreq++]);
            }
            MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

            double local_res2_color = 0.0;
            double local_err2_color = 0.0;

            #pragma omp parallel for collapse(3) reduction(+:local_res2_color, local_err2_color)
            for (int local_k = 1; local_k <= nz_local; ++local_k) {
                for (int j = 1; j < Ny - 1; ++j) {
                    for (int i = 1; i < Nx - 1; ++i) {

                        int k = k_start + (local_k - 1); 
                        if (k == 0 || k == Nz - 1) continue;

                        double x = i * hx;
                        double y = j * hy;
                        double z = k * hz;

                        int id = idx(i, j, local_k, Nx, Ny);

                        if ((i + j + k) % 2 != color) continue;

                        double u_e = u[idx(i + 1, j,     local_k,     Nx, Ny)];
                        double u_w = u[idx(i - 1, j,     local_k,     Nx, Ny)];
                        double u_n = u[idx(i,     j + 1, local_k,     Nx, Ny)];
                        double u_s = u[idx(i,     j - 1, local_k,     Nx, Ny)];
                        double u_t = u[idx(i,     j,     local_k + 1, Nx, Ny)];
                        double u_b = u[idx(i,     j,     local_k - 1, Nx, Ny)];

                        double u_old = u[id];
                        double S    = u_e + u_w + u_n + u_s + u_t + u_b;
                        double rhs  = f[id];

                       
                        double u_GS = (S + h2 * rhs) / 6.0;
                        double u_new = (1.0 - omega) * u_old + omega * u_GS;
                        u[id] = u_new;

                        double lap = (S - 6.0 * u_new) / h2; 
                        double r   = -lap - rhs;
                        local_res2_color += r * r;

                        double ue  = u_exact(x, y, z);
                        double err = u_new - ue;
                        local_err2_color += err * err;
                    }
                }
            } 

            local_res2 += local_res2_color;
            local_err2 += local_err2_color;
        } 

        double vals[2] = { local_res2, local_err2 };
        MPI_Allreduce(MPI_IN_PLACE, vals, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        global_res2 = vals[0];
        global_err2 = vals[1];

        double resL2 = std::sqrt(global_res2);
        double errL2 = std::sqrt(global_err2 * vol);


        if (rank == 0 && (iter % 50 == 0 || resL2 < tol)) {
            std::printf("Iter %5d: residual L2 = %.3e, error L2 = %.3e\n",
                        iter, resL2, errL2);
        }

        if (resL2 < tol) {
            if (rank == 0)
                std::printf("Converged at iter %d with residual %.3e\n", iter, resL2);
            break;
        }
    } 

    double t1 = MPI_Wtime();
    if (rank == 0) {
        std::printf("Total iterations: %d\n", iter);
        std::printf("Total time: %.3f s\n", t1 - t0);

        int Nx_int = Nx - 2;
        int Ny_int = Ny - 2;
        int Nz_int = Nz - 2;
        double points_per_sweep = 1.0 * Nx_int * Ny_int * Nz_int;
        double flops_per_point = 16.0;
        double total_flops = 2.0 * iter * points_per_sweep * flops_per_point;
        std::printf("Estimated total FLOPs: %.3e\n", total_flops);
        std::printf("Average FLOP/s: %.3e\n", total_flops / (t1 - t0));
    }

    MPI_Finalize();
    return 0;
}
