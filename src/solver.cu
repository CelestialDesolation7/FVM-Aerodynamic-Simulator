#include "solver.cuh"
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cub/cub.cuh>

CFDSolver::CFDSolver()
{
}

CFDSolver::~CFDSolver()
{
    // 清理CUDA资源
}

// 初始化求解器，分配内存等
void CFDSolver::initialize(const SimParams &params)
{
}

void CFDSolver::resize(int nx, int ny)
{
}

// 进行单步仿真计算
void CFDSolver::step(SimParams &params)
{
}

// 按照当前参数重置求解器状态
void CFDSolver::reset(const SimParams &params)
{
}

// 获取可视化数据（暂时不使用OpenGL-cuda互操作）
// 注意这里指针都是主机缓冲区数组
// 上线前一定要记得改成OpenGL-cuda互操作以提升性能
void CFDSolver::getTemperatureField(float *host_T)
{
}

void CFDSolver::getPressureField(float *host_p)
{
}

void CFDSolver::getVelocityField(float *host_u, float *host_v)
{
}

void CFDSolver::getDensityField(float *host_rho)
{
}

void CFDSolver::getCellTypes(uint8_t *host_types)
{
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