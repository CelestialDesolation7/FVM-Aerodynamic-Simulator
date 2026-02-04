#pragma once

// #include <cuda_gl_interop.h>  // 已移至renderer.cpp，避免CUDA编译器与GL头文件冲突
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

constexpr float GAMMA = 1.4f; // 空气比热比
constexpr float R_GAS = 287.05f; // 空气比气体常数，单位 J/(kg·K)
constexpr float CV = R_GAS / (GAMMA - 1.0f); // 定容比热容，单位 J/(kg·K)
constexpr float CP = GAMMA * CV; // 定压比热容，单位 J/(kg·K)

// 用于计算空气粘性的Sutherland公式常数
constexpr float MU_REF = 1.716e-5f; // 参考粘性系数，单位 Pa·s
constexpr float T_REF = 273.15f;    // 参考温度，单位 K
constexpr float S_SUTHERLAND = 110.4f; // Sutherland常数，单位 K

// 普朗特数
constexpr float PRANDTL_NUMBER = 0.71f;

// 仿真参数配置
struct SimParams{
    // 网格参数
    int nx = 512;          // 网格在X方向的划分数
    int ny = 256;          // 网格在Y方向的划分数
    float dx = 1.0f;
    float dy = 1.0f;
    float domain_width = 10.0f;  // 计算域宽度，单位米
    float domain_height = 5.0f;  // 计算域高度，单位

    //来流参数
    float mach = 3.0f;          // 来流马赫数
    float T_inf = 300.0f;       // 来流温度，单位 K
    float p_inf = 101325.0f;    // 来流静压，单位 Pa
    float rho_inf;     // 来流密度，单位 kg/m³
    float u_inf;       // 来流水平速度，单位 m/s
    float v_inf;         // 来流垂直速度，单位 m/s
    float c_inf;       // 来流声速，单位 m/s

    // 障碍物参数
    float obstacle_x = 2.5f;    // 障碍物中心X坐标，单位米
    float obstacle_y = 2.5f;    // 障碍物中心Y坐标，单位米
    float obstacle_r = 0.5f;    // 障碍物半径，单位米
    float obstacle_rotation = 0.0f; // 障碍物旋转角度，单位度
    int obstacle_shape = 0;      // 障碍物形状，枚举值

    // 时间参数
    float cfl = 0.5f;            // CFL数
    float dt = 1e-6f;            // 时间步长，单位秒
    float t_current = 0.0f;     // 当前仿真时间，单位秒
    int step = 0;               // 当前仿真步数

    // 粘性设置
    bool enable_viscosity = false; // 是否启用粘性项
    float cfl_visc = 0.4f;          // 粘性CFL数
    bool adiabatic_wall = true;    // 是否启用绝热壁面条件
    float T_wall = 300.0f;        // 壁面温度，单位 K

    // 输出设置
    float T_min = 250.0f;        // 温度映射最小值，单位 K
    float T_max = 800.0f;        // 温度映射最大值，单位 K
    bool show_velocity = false;   // 是否显示速度矢量场
    bool paused = true;           // 仿真是否暂停

    // 配置更新时调用
    __host__ void computeDerived(){
        rho_inf = p_inf / (R_GAS * T_inf);      // 来流密度 理想气体状态方程
        c_inf = sqrtf(GAMMA * R_GAS * T_inf);   // 音速 只和温度有关
        u_inf = mach * c_inf;                   // 将马赫数转化为实际速度
        dx = domain_width / (float)nx;          // 更新每个网格的尺度
        dy = domain_height / (float)ny;
        obstacle_y = domain_height / 2.0f;      // 障碍物中心y轴坐标放中间
    }
};

// 保存守恒变量的结构体
struct ConservedVars{
    float rho;   // 密度
    float rho_u; // x方向动量
    float rho_v; // y方向动量
    float E;     // 总能量
};

// 保存原始变量的结构体
struct PrimitiveVars{
    float rho; // 密度
    float u;   // x方向速度
    float v;   // y方向速度
    float p;   // 压强
    float T;   // 温度
};

// 提供五种障碍物
enum class ObstacleShape{
    CIRCLE = 0,  // 圆形
    STAR = 1,    // 五角星
    DIAMOND = 2, // 菱形
    CAPSULE = 3, // 胶囊形（圆角矩形）
    TRIANGLE = 4 // 三角形
};

// 单元格分类
enum CellType : uint8_t
{
    CELL_FLUID = 0,     // 完全流体
    CELL_SOLID = 1,     // 完全固体
    CELL_GHOST = 2,     // 固液交界
    CELL_INFLOW = 3,    // 左侧，入流面边界
    CELL_OUTFLOW = 4    // 右侧，出流面边界
};