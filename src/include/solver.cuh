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

    // CPU数据传输路径（非零拷贝模式）
    void getTemperatureField(float *host_T);
    void getPressureField(float *host_p);
    void getVelocityField(float *host_u, float *host_v);
    void getDensityField(float *host_rho);
    void getCellTypes(uint8_t *host_types);

    // GPU零拷贝路径（CUDA-OpenGL互操作）
    void computeTemperatureToDevice(float *dev_dst);
    void computePressureToDevice(float *dev_dst);
    void computeDensityToDevice(float *dev_dst);
    void computeVelocityMagToDevice(float *dev_dst);
    void computeMachToDevice(float *dev_dst);

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
    float *d_rho_e_ = nullptr; // 网格内空气内能密度（用于双能量法）

    // 为实现双缓冲需要再保存下一个状态的保守量
    float *d_rho_new_ = nullptr;
    float *d_rho_u_new_ = nullptr;
    float *d_rho_v_new_ = nullptr;
    float *d_E_new_ = nullptr;
    float *d_rho_e_new_ = nullptr;

    // 为实现可视化和通量计算保存的原始变量
    float *d_u_ = nullptr; // x轴空气速度
    float *d_v_ = nullptr; // y轴空气速度
    float *d_p_ = nullptr; // 空气压强
    float *d_T_ = nullptr; // 空气温度

    // 用于 IBM 的辅助数据场
    uint8_t *d_cell_type_ = nullptr; // 网格类型标记
    float *d_sdf_ = nullptr;         // 符号距离场

    // 通量，中间存储
    float *d_flux_rho_x_ = nullptr;   // x轴-质量通量场
    float *d_flux_rho_u_x_ = nullptr; // x轴-x方向动量-通量场
    float *d_flux_rho_v_x_ = nullptr; // x轴-y方向动量-通量场
    float *d_flux_E_x_ = nullptr;     // x轴-能量-通量场
    float *d_flux_rho_e_x_ = nullptr; // x轴-内能-通量场

    float *d_flux_rho_y_ = nullptr;   // y轴-质量通量场
    float *d_flux_rho_u_y_ = nullptr; // y轴-x方向动量-通量场
    float *d_flux_rho_v_y_ = nullptr; // y轴-y方向动量-通量场
    float *d_flux_E_y_ = nullptr;     // y轴-能量-通量场
    float *d_flux_rho_e_y_ = nullptr; // y轴-内能-通量场

    // 归约缓冲区
    float *d_reduction_buffer_ = nullptr;

    // 粘性相关中间量 (Navier-Stokes方程)
    float *d_mu_ = nullptr;     // 动态更新的粘度值场
    float *d_k_ = nullptr;      // 热导率场
    float *d_tau_xx_ = nullptr; // x轴方向-粘性力动量-通量场
    float *d_tau_yy_ = nullptr; // y轴方向-粘性力动量-通量场
    float *d_tau_xy_ = nullptr; // x和y轴方向-摩擦力动量-通量场
    float *d_qx_ = nullptr;     // x轴方向-内能-通量场
    float *d_qy_ = nullptr;     // y轴方向-内能-通量场

    // 功能:启动初始化核函数，将全场设为来流条件
    // 输入:守恒变量数组指针，仿真参数，网格尺寸
    // 输出:初始化后的守恒变量场
    void launchInitializeKernel(float *rho, float *rho_u, float *rho_v, float *E, float *rho_e,
                                const SimParams &params, int nx, int ny);

    // 功能:启动原始变量计算核函数，从守恒变量推导原始变量
    // 输入:守恒变量场(rho, rho_u, rho_v, E, rho_e)，网格尺寸
    // 输出:原始变量场(u, v, p, T)，使用双能量法保证精度
    void launchComputePrimitivesKernel(const float *rho, const float *rho_u,
                                       const float *rho_v, const float *E,
                                       const float *rho_e,
                                       float *u, float *v, float *p, float *T,
                                       int nx, int ny);

    // 功能:启动通量计算核函数，使用MUSCL重构和HLLC Riemann求解器
    // 输入:守恒变量场，原始变量场，网格类型，仿真参数
    // 输出:X和Y方向的数值通量(flux_*_x/y)，包括质量/动量/能量/内能通量
    void launchComputeFluxesKernel(const float *rho, const float *rho_u,
                                   const float *rho_v, const float *E,
                                   const float *rho_e,
                                   const float *u, const float *v,
                                   const float *p, const float *T,
                                   const uint8_t *cell_type,
                                   float *flux_rho_x, float *flux_rho_u_x,
                                   float *flux_rho_v_x, float *flux_E_x,
                                   float *flux_rho_e_x,
                                   float *flux_rho_y, float *flux_rho_u_y,
                                   float *flux_rho_v_y, float *flux_E_y,
                                   float *flux_rho_e_y,
                                   const SimParams &params, int nx, int ny);

    // 功能:启动更新核函数，使用有限体积法和双能量法更新守恒变量
    // 输入:当前守恒变量，X/Y方向通量，网格类型，时间步长和网格间距
    // 输出:下一时间步的守恒变量(rho_new, rho_u_new等)，通过双缓冲实现
    void launchUpdateKernel(const float *rho, const float *rho_u,
                            const float *rho_v, const float *E, const float *rho_e,
                            const float *flux_rho_x, const float *flux_rho_u_x,
                            const float *flux_rho_v_x, const float *flux_E_x,
                            const float *flux_rho_e_x,
                            const float *flux_rho_y, const float *flux_rho_u_y,
                            const float *flux_rho_v_y, const float *flux_E_y,
                            const float *flux_rho_e_y,
                            const uint8_t *cell_type,
                            float *rho_new, float *rho_u_new,
                            float *rho_v_new, float *E_new, float *rho_e_new,
                            const SimParams &params, int nx, int ny);

    // 功能:启动边界条件应用核函数，处理所有边界类型
    // 输入:守恒变量，网格类型，SDF场，仿真参数
    // 输出:应用边界条件后的守恒变量(流入/流出/固体壁面/Ghost Cell)
    void launchApplyBoundaryConditionsKernel(float *rho, float *rho_u, float *rho_v, float *E,
                                             float *rho_e,
                                             const uint8_t *cell_type, const float *sdf,
                                             const SimParams &params, int nx, int ny);

    // 功能:启动SDF计算核函数，计算带符号距离场并初始化网格类型
    // 输入:障碍物几何参数(位置/尺寸/旋转/形状)，网格尺寸
    // 输出:SDF场和网格类型标记(流体/固体/虚拟/流入/流出)
    void launchComputeSDFKernel(float *sdf, uint8_t *cell_type,
                                const SimParams &params, int nx, int ny);

    // 功能:使用CUB库归约计算全场最大温度
    // 输入:温度场数组，网格尺寸
    // 输出:最大温度值(用于监控激波强度和数值稳定性)
    float launchComputeMaxTemperature(const float *T, int nx, int ny);

    // 功能:归约计算全场最大马赫数
    // 输入:速度场(u,v)，压强场，密度场，网格尺寸
    // 输出:最大马赫数 Ma = |v| / c (用于判断流动类型：亚音速/超音速)
    float launchComputeMaxMach(const float *u, const float *v, const float *p,
                               const float *rho, int nx, int ny);

    // 功能:归约计算全场最大波速(速度+声速)
    // 输入:速度场(u,v)，压强场，密度场，网格尺寸
    // 输出:最大波速 = |v| + c (用于CFL条件的时间步长限制)
    float launchComputeMaxWaveSpeed(const float *u, const float *v, const float *p,
                                    const float *rho, int nx, int ny);

    // 功能:启动粘性计算核函数，使用Sutherland公式
    // 输入:温度场，网格尺寸
    // 输出:动力粘性系数场 mu 和热导率场 k (用于Navier-Stokes粘性项)
    void launchComputeViscosityKernel(const float *T, float *mu, float *k, int nx, int ny);

    // 功能:启动应力张量计算核函数，基于Stokes假设
    // 输入:速度场(u,v)，粘性系数场，网格类型，网格间距
    // 输出:应力张量分量(tau_xx, tau_yy, tau_xy)，用于粘性力计算
    void launchComputeStressTensorKernel(const float *u, const float *v, const float *mu,
                                         float *tau_xx, float *tau_yy, float *tau_xy,
                                         const uint8_t *cell_type,
                                         float dx, float dy, int nx, int ny);

    // 功能:启动热通量计算核函数，使用Fourier定律
    // 输入:温度场，热导率场，网格类型，网格间距
    // 输出:热通量分量(qx, qy)，q = -k * grad(T)，用于热传导计算
    void launchComputeHeatFluxKernel(const float *T, const float *k,
                                     float *qx, float *qy,
                                     const uint8_t *cell_type,
                                     float dx, float dy, int nx, int ny);

    // 功能:启动粘性扩散步核函数，积分粘性力和热传导项
    // 输入:守恒变量，应力张量，热通量，速度场，时间步长，网格间距
    // 输出:更新后的动量和能量(包括粘性耗散和热传导效应)
    void launchDiffusionStepKernel(float *rho_u, float *rho_v, float *E, float *rho_e,
                                   const float *tau_xx, const float *tau_yy, const float *tau_xy,
                                   const float *qx, const float *qy,
                                   const float *u, const float *v,
                                   const uint8_t *cell_type,
                                   float dt, float dx, float dy, int nx, int ny);

    // 功能:归约计算全场最大运动粘性系数 nu = mu / rho
    // 输入:动力粘性系数场，密度场，网格尺寸
    // 输出:最大运动粘性系数(用于粘性CFL条件: dt <= dx^2 / nu)
    float launchComputeMaxViscousNumber(const float *mu, const float *rho, int nx, int ny);
};
