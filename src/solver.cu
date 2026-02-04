#include "solver.cuh"
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cub/cub.cuh>

#pragma region 数学工具函数

#pragma endregion

#pragma region 单核数学操作函数

#pragma endregion

#pragma region 并行计算主调函数
__global__ void testBlinkKernel(float *T, int nx, int ny, float time)
{
    // 计算当前线程对应的全局网格索引
    int i = blockIdx.x * blockDim.x + threadIdx.x; // X 方向索引
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Y 方向索引

    // 边界检查，防止越界
    if (i < nx && j < ny)
    {
        int idx = j * nx + i; // 展平的一维索引

        // 生成一个随空间(i,j)和时间(time)变化的波
        // float val = 300.0f + 100.0f * sinf(time * 5.0f + i * 0.1f + j * 0.1f);
        float val = 400.0f + 300.0f * sinf(time * 2.0f + i / 50.0f);

        T[idx] = val; // 写入显存
    }
}
#pragma endregion

#pragma region 计算链顶层的求解器类内部方法实现
void CFDSolver::allocateMemory()
{
    if (_nx <= 0 || _ny <= 0)
        return;

    size_t num_cells = _nx * _ny;
    size_t float_size = num_cells * sizeof(float);
    size_t uint8_size = num_cells * sizeof(uint8_t);

    // 1. 分配内存给当前时间步守恒变量
    CUDA_CHECK(cudaMalloc((void **)&d_rho_, float_size));
    CUDA_CHECK(cudaMalloc((void **)&d_rho_u_, float_size));
    CUDA_CHECK(cudaMalloc((void **)&d_rho_v_, float_size));
    CUDA_CHECK(cudaMalloc((void **)&d_E_, float_size));
    CUDA_CHECK(cudaMalloc((void **)&d_rho_e_, float_size));

    // 2. 分配下一步守恒变量
    CUDA_CHECK(cudaMalloc((void **)&d_rho_new_, float_size));
    CUDA_CHECK(cudaMalloc((void **)&d_rho_u_new_, float_size));
    CUDA_CHECK(cudaMalloc((void **)&d_rho_v_new_, float_size));
    CUDA_CHECK(cudaMalloc((void **)&d_E_new_, float_size));
    CUDA_CHECK(cudaMalloc((void **)&d_rho_e_new_, float_size));

    // 3. 分配原始变量
    CUDA_CHECK(cudaMalloc((void **)&d_u_, float_size));
    CUDA_CHECK(cudaMalloc((void **)&d_v_, float_size));
    CUDA_CHECK(cudaMalloc((void **)&d_p_, float_size));
    CUDA_CHECK(cudaMalloc((void **)&d_T_, float_size));

    // 4. 分配辅助数据
    CUDA_CHECK(cudaMalloc((void **)&d_cell_type_, uint8_size));
    CUDA_CHECK(cudaMalloc((void **)&d_sdf_, float_size));

    // 初始化显存为0，避免脏数据
    CUDA_CHECK(cudaMemset(d_rho_, 0, float_size));
    CUDA_CHECK(cudaMemset(d_T_, 0, float_size));
}

void CFDSolver::freeMemory()
{
    // 释放所有指针
    if (d_rho_)
    {
        CUDA_CHECK(cudaFree(d_rho_));
        d_rho_ = nullptr;
    }
    if (d_rho_u_)
    {
        CUDA_CHECK(cudaFree(d_rho_u_));
        d_rho_u_ = nullptr;
    }
    if (d_rho_v_)
    {
        CUDA_CHECK(cudaFree(d_rho_v_));
        d_rho_v_ = nullptr;
    }
    if (d_E_)
    {
        CUDA_CHECK(cudaFree(d_E_));
        d_E_ = nullptr;
    }
    if (d_rho_e_)
    {
        CUDA_CHECK(cudaFree(d_rho_e_));
        d_rho_e_ = nullptr;
    }

    if (d_rho_new_)
    {
        CUDA_CHECK(cudaFree(d_rho_new_));
        d_rho_new_ = nullptr;
    }
    if (d_rho_u_new_)
    {
        CUDA_CHECK(cudaFree(d_rho_u_new_));
        d_rho_u_new_ = nullptr;
    }
    if (d_rho_v_new_)
    {
        CUDA_CHECK(cudaFree(d_rho_v_new_));
        d_rho_v_new_ = nullptr;
    }
    if (d_E_new_)
    {
        CUDA_CHECK(cudaFree(d_E_new_));
        d_E_new_ = nullptr;
    }
    if (d_rho_e_new_)
    {
        CUDA_CHECK(cudaFree(d_rho_e_new_));
        d_rho_e_new_ = nullptr;
    }

    if (d_u_)
    {
        CUDA_CHECK(cudaFree(d_u_));
        d_u_ = nullptr;
    }
    if (d_v_)
    {
        CUDA_CHECK(cudaFree(d_v_));
        d_v_ = nullptr;
    }
    if (d_p_)
    {
        CUDA_CHECK(cudaFree(d_p_));
        d_p_ = nullptr;
    }
    if (d_T_)
    {
        CUDA_CHECK(cudaFree(d_T_));
        d_T_ = nullptr;
    }

    if (d_cell_type_)
    {
        CUDA_CHECK(cudaFree(d_cell_type_));
        d_cell_type_ = nullptr;
    }
    if (d_sdf_)
    {
        CUDA_CHECK(cudaFree(d_sdf_));
        d_sdf_ = nullptr;
    }
}

void CFDSolver::updateCellTypes(const SimParams &params)
{
}

void computeSDF(const SimParams &params)
{
}
#pragma region

#pragma region 计算链顶层的求解器类公开接口实现
CFDSolver::CFDSolver()
{
}

CFDSolver::~CFDSolver()
{
    freeMemory();
    // 清理CUDA资源
}

// 初始化求解器，分配内存等
void CFDSolver::initialize(const SimParams &params)
{
    // 先清理旧内存，再设置尺寸，分配新内存
    freeMemory();
    _nx = params.nx;
    _ny = params.ny;
    allocateMemory();
}

// 修改网格分辨率后的重新初始化
void CFDSolver::resize(int nx, int ny)
{
    if (_nx == nx && _ny == ny)
        return;
    freeMemory();
    _nx = nx;
    _ny = ny;
    allocateMemory();
}

// 进行单步仿真计算
void CFDSolver::step(SimParams &params)
{
    // 定义线程块大小
    dim3 blockSize(16, 16);
    // 计算网格需要多少个块来覆盖 (向上取整)
    dim3 gridSize((_nx + blockSize.x - 1) / blockSize.x,
                  (_ny + blockSize.y - 1) / blockSize.y);

    // 启动 Kernel
    testBlinkKernel<<<gridSize, blockSize>>>(d_T_, _nx, _ny, params.t_current);

    // 检查 Kernel 启动是否出错（异步错误可能稍后才会捕获）
    CUDA_CHECK(cudaGetLastError());

    // CUDA kernel 是异步执行的，CPU 会立即继续执行
    // 只有在 cudaMemcpy 时 CPU 才会等待数据准备好

    // 更新时间
    params.t_current += params.dt;
    params.step++;
}

// 按照当前参数重置求解器状态
void CFDSolver::reset(const SimParams &params)
{
    initialize(params);
}

// 获取可视化数据（暂时不使用OpenGL-cuda互操作）
// 注意这里指针都是主机缓冲区数组
// 上线前一定要记得改成OpenGL-cuda互操作以提升性能
void CFDSolver::getTemperatureField(float *host_T)
{
    if (!d_T_)
        return;
    CUDA_CHECK(cudaMemcpy(host_T, d_T_, _nx * _ny * sizeof(float), cudaMemcpyDeviceToHost));
}

void CFDSolver::getPressureField(float *host_p)
{
    if (!d_p_)
        return;
    CUDA_CHECK(cudaMemcpy(host_p, d_p_, _nx * _ny * sizeof(float), cudaMemcpyDeviceToHost));
}

void CFDSolver::getVelocityField(float *host_u, float *host_v)
{
    if (!d_u_ || !d_v_)
        return;
    size_t size = _nx * _ny * sizeof(float);
    CUDA_CHECK(cudaMemcpy(host_u, d_u_, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_v, d_v_, size, cudaMemcpyDeviceToHost));
}

void CFDSolver::getDensityField(float *host_rho)
{
    if (!d_rho_)
        return;
    CUDA_CHECK(cudaMemcpy(host_rho, d_rho_, _nx * _ny * sizeof(float), cudaMemcpyDeviceToHost));
}

void CFDSolver::getCellTypes(uint8_t *host_types)
{
    if (!d_cell_type_)
        return;
    CUDA_CHECK(cudaMemcpy(host_types, d_cell_type_, _nx * _ny * sizeof(uint8_t), cudaMemcpyDeviceToHost));
}

// 计算稳定时间步长(CFL条件)
float CFDSolver::computeStableTimeStep(const SimParams &params)
{
    return 1e-6f;
}

// 获取统计数据
float CFDSolver::getMaxTemperature()
{
    return 1e-6f;
}

float CFDSolver::getMaxMach()
{
    return 1e-6f;
}

void CFDSolver::getGPUMemoryInfo(size_t &totalMem, size_t &freeMem)
{
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
}

size_t CFDSolver::getSimulationMemoryUsage()
{
    return (size_t)_nx * _ny * MEM_PER_UNIT; // 假设有10个浮点数组用于存储模拟数据
}
#pragma endregion