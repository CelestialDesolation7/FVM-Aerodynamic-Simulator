#pragma once

#include "common.h"

// 宏定义：检查 CUDA 函数调用的返回值
#define CUDA_CHECK(call)                                                             \
    do                                                                               \
    {                                                                                \
        cudaError_t err = call;                                                      \
        if (err != cudaSuccess)                                                      \
        {                                                                            \
            fprintf(stderr, "CUDA 相关函数调用返回错误结果 '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));                    \
            exit(EXIT_FAILURE);                                                      \
        }                                                                            \
    } while (0)

constexpr size_t MEM_PER_UNIT = 20; // 每个网格单元大约占用的字节数（估算值）

class CFDSolver
{
public:
    CFDSolver();
    ~CFDSolver();

    // 初始化求解器，分配内存等
    void initialize(const SimParams &params);
    void resize(int nx, int ny);

    // 进行单步仿真计算
    void step(SimParams &params);

    // 按照当前参数重置求解器状态
    void reset(const SimParams &params);

    // 获取可视化数据（暂时不使用OpenGL-cuda互操作）
    // 注意这里指针都是主机缓冲区数组
    // 上线前一定要记得改成OpenGL-cuda互操作以提升性能
    void getTemperatureField(float *host_T);
    void getPressureField(float *host_p);
    void getVelocityField(float *host_u, float *host_v);
    void getDensityField(float *host_rho);
    void getCellTypes(uint8_t *host_types);

    // 获取网格尺寸
    int getNx() const { return _nx; }
    int getNy() const { return _ny; }

    // 计算稳定时间步长(CFL条件)
    float computeStableTimeStep(const SimParams &params);

    // 获取统计数据
    float getMaxTemperature();
    float getMaxMach();

    static void getGPUMemoryInfo(size_t &totalMem, size_t &freeMem);

    size_t getSimulationMemoryUsage();

private:
    // 网格维度
    int _nx = 1024;
    int _ny = 512;
};