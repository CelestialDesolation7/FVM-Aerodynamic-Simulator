#pragma once

#include "common.h"

#pragma region 宏定义
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
#pragma endregion

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
    // 显存分配器
    void allocateMemory();
    void freeMemory();

    // 基于指定的障碍物几何形状初始化网格类型
    void updateCellTypes(const SimParams &params);

    // 计算SDF（有符号距离场）以便实现Ghost Cell方法
    void computeSDF(const SimParams &params);

    // 网格维度
    int _nx = 1024;
    int _ny = 512;

    // 保守量的显存位置指针
    float *d_rho_ = nullptr;   // 网格空气密度
    float *d_rho_u_ = nullptr; // 网格空气水平速度
    float *d_rho_v_ = nullptr; // 网格空气垂直速度
    float *d_E_ = nullptr;     // 网格内空气总能量（内能+动能）
    float *d_rho_e_ = nullptr; // 网格内空气

    // 为实现双缓冲需要再保存下一个状态的保守量
    float *d_rho_new_ = nullptr;
    float *d_rho_u_new_ = nullptr;
    float *d_rho_v_new_ = nullptr;
    float *d_E_new_ = nullptr;
    float *d_rho_e_new_ = nullptr;

    // 为实现可视化和通量计算保存的原始变量
    float *d_u_ = nullptr;
    float *d_v_ = nullptr;
    float *d_p_ = nullptr;
    float *d_T_ = nullptr;

    // 用于 IBM 的辅助数据场
    uint8_t *d_cell_type_ = nullptr; // 网格类型标记
    float *d_sdf_ = nullptr;         // 符号距离场
};