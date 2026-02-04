#include "solver.cuh"
#include <cstdio>
#include <algorithm>
#include <vector>
#include <cub/cub.cuh>

#pragma region 数学工具函数

// 功能:计算点到线段的最短距离(用于多边形SDF)
// 输入:点坐标(px, py), 线段端点A(ax, ay)和B(bx, by)
// 输出:点到线段的欧几里得距离
__device__ float distToSegment(float px, float py, float ax, float ay, float bx, float by)
{
    // 线段向量 AB = B - A
    float abx = bx - ax, aby = by - ay;
    // 点向量 AP = P - A
    float apx = px - ax, apy = py - ay;

    // 投影参数 t = (AP · AB) / |AB|^2
    // t 表示点P在线段AB上的投影位置(0表示A点，1表示B点)
    float t = (apx * abx + apy * aby) / (abx * abx + aby * aby + 1e-10f);

    // 将 t 限制在 [0,1] 范围内(对应线段上的点)
    t = fmaxf(0.0f, fminf(1.0f, t));

    // 线段上最近点的坐标 = A + t * AB
    float closestX = ax + t * abx;
    float closestY = ay + t * aby;

    // 计算点到最近点的距离
    float dx = px - closestX;
    float dy = py - closestY;
    return sqrtf(dx * dx + dy * dy);
}

// 功能:计算二维向量的叉积(用于判断点的相对位置和环绕数)
// 输入:两个二维向量 A(ax, ay) 和 B(bx, by)
// 输出:叉积标量值 = ax*by - ay*bx (正值表示B在A的左侧)
__device__ float cross2d(float ax, float ay, float bx, float by)
{
    return ax * by - ay * bx;
}

#pragma endregion

#pragma region 多边形SDF场计算算法

// 功能:计算点到圆形的带符号距离
// 输入:点坐标(px, py), 圆心(cx, cy), 半径 r
// 输出:带符号距离(负值表示在圆内，正值表示在圆外)
__device__ float sdfCircle(float px, float py, float cx, float cy, float r)
{
    // SDF(圆) = |点到圆心距离| - 半径
    return sqrtf((px - cx) * (px - cx) + (py - cy) * (py - cy)) - r;
}

// 功能:计算点到五角星的带符号距离
// 输入:点坐标(px, py), 中心(cx, cy), 外接圆半径 r, 旋转角度 rotation
// 输出:带符号距离
__device__ float sdfStar(float px, float py, float cx, float cy, float r, float rotation)
{
    const float PI = 3.14159265359f;
    const int N = 5;                // 5个尖角
    const float outerR = r;         // 外圆半径(尖角处)
    const float innerR = r * 0.38f; // 内圆半径(凹陷处，0.38是五角星的标准比例)

    // 转换到局部坐标系(以中心为原点)
    float lx = px - cx;
    float ly = py - cy;

    // 应用旋转(旋转矩阵的逆变换)
    float cosR = cosf(-rotation);
    float sinR = sinf(-rotation);
    float qx = lx * cosR - ly * sinR;
    float qy = lx * sinR + ly * cosR;

    // 生成五角星的10个顶点(5个外顶点 + 5个内顶点，交替排列)
    float verts[20]; // 10个顶点 * 2坐标 = 20个浮点数
    for (int i = 0; i < N; i++)
    {
        // 外顶点角度(从-90度开始，保证尖角朝右)
        float outerAngle = 2.0f * PI * i / N - PI / 2.0f;
        // 内顶点角度(在两个外顶点中间)
        float innerAngle = outerAngle + PI / N;

        // 存储外顶点坐标
        verts[i * 4 + 0] = outerR * cosf(outerAngle);
        verts[i * 4 + 1] = outerR * sinf(outerAngle);
        // 存储内顶点坐标
        verts[i * 4 + 2] = innerR * cosf(innerAngle);
        verts[i * 4 + 3] = innerR * sinf(innerAngle);
    }

    // 计算点到所有边的最短距离
    float minDist = 1e10f;
    float windingSum = 0.0f; // 环绕数累加器

    for (int i = 0; i < 2 * N; i++)
    {
        // 当前边的两个端点(循环连接)
        int j = (i + 1) % (2 * N);
        float ax = verts[i * 2], ay = verts[i * 2 + 1];
        float bx = verts[j * 2], by = verts[j * 2 + 1];

        // 更新最短距离
        float d = distToSegment(qx, qy, ax, ay, bx, by);
        minDist = fminf(minDist, d);

        // 计算环绕数(winding number)贡献
        // 环绕数通过累加每条边对应的角度变化来判断点是否在多边形内
        float eax = ax - qx, eay = ay - qy; // 边起点到查询点的向量
        float ebx = bx - qx, eby = by - qy; // 边终点到查询点的向量
        // atan2(叉积, 点积) 给出两个向量之间的夹角
        windingSum += atan2f(cross2d(eax, eay, ebx, eby), eax * ebx + eay * eby);
    }

    // 根据环绕数判断内外
    // 如果环绕数的绝对值接近2*PI或更大，说明点在多边形内
    float sign = (fabsf(windingSum) > PI) ? -1.0f : 1.0f;

    return sign * minDist;
}

// 功能:计算点到菱形(旋转正方形)的带符号距离
// 输入:点坐标(px, py), 中心(cx, cy), 半边长 r, 旋转角度 rotation
// 输出:带符号距离
__device__ float sdfDiamond(float px, float py, float cx, float cy, float r, float rotation)
{
    // 转换到局部坐标
    float lx = px - cx;
    float ly = py - cy;

    // 应用旋转
    float cosR = cosf(-rotation);
    float sinR = sinf(-rotation);
    float rx = lx * cosR - ly * sinR;
    float ry = lx * sinR + ly * cosR;

    // 菱形的SDF使用L1范数(曼哈顿距离)
    // 将坐标归一化到半径
    float ndx = fabsf(rx) / r;
    float ndy = fabsf(ry) / r;

    // SDF = (|x| + |y| - 1) * r / sqrt(2)
    // 0.7071 = 1/sqrt(2)，用于将L1距离转换为欧几里得等效距离
    return (ndx + ndy - 1.0f) * r * 0.7071f;
}

// 功能:计算点到胶囊形(圆角矩形)的带符号距离
// 输入:点坐标(px, py), 中心(cx, cy), 长度 r, 旋转角度 rotation
// 输出:带符号距离
__device__ float sdfCapsule(float px, float py, float cx, float cy, float r, float rotation)
{
    // 转换到局部坐标
    float lx = px - cx;
    float ly = py - cy;

    // 应用旋转
    float cosR = cosf(-rotation);
    float sinR = sinf(-rotation);
    float rx = lx * cosR - ly * sinR;
    float ry = lx * sinR + ly * cosR;

    // 核心思想是，胶囊形可以看作是一个半径为 capRadius 的圆，沿着一条线段“扫过”形成的区域
    // 胶囊参数: 水平方向的伸展
    float halfWidth = r * 1.5f; // 胶囊的半长(中轴长度的一半)
    float capRadius = r * 0.5f; // 两端圆弧的半径

    // 将x坐标限制到中轴线段上
    // 中轴线段范围: [-halfWidth + capRadius, halfWidth - capRadius]
    float clampedX = fmaxf(-halfWidth + capRadius, fminf(halfWidth - capRadius, rx));

    // 计算点到中轴线段最近点的距离
    float dx = rx - clampedX;
    float dy = ry;

    // SDF = 点到中轴距离 - 半径
    return sqrtf(dx * dx + dy * dy) - capRadius;
}

// 功能:计算点到三角形(等边，尖端向右)的带符号距离
// 输入:点坐标(px, py), 中心(cx, cy), 外接圆半径 r, 旋转角度 rotation
// 输出:带符号距离
__device__ float sdfTriangle(float px, float py, float cx, float cy, float r, float rotation)
{
    // 转换到局部坐标
    float lx = px - cx;
    float ly = py - cy;

    // 应用旋转
    float cosR = cosf(-rotation);
    float sinR = sinf(-rotation);
    float qx = lx * cosR - ly * sinR;
    float qy = lx * sinR + ly * cosR;

    // 定义等边三角形的三个顶点
    // 顶点0(右尖): (r, 0)
    // 顶点1(左上): (-r/2, r*sqrt(3)/2)
    // 顶点2(左下): (-r/2, -r*sqrt(3)/2)
    const float sqrt3_2 = 0.866025404f; // sqrt(3)/2
    float v0x = r, v0y = 0.0f;
    float v1x = -r * 0.5f, v1y = r * sqrt3_2;
    float v2x = -r * 0.5f, v2y = -r * sqrt3_2;

    // 计算点到三条边的最短距离
    float d0 = distToSegment(qx, qy, v0x, v0y, v1x, v1y); // 边 v0-v1
    float d1 = distToSegment(qx, qy, v1x, v1y, v2x, v2y); // 边 v1-v2
    float d2 = distToSegment(qx, qy, v2x, v2y, v0x, v0y); // 边 v2-v0

    float minDist = fminf(d0, fminf(d1, d2));

    // 使用叉积判断点是否在三角形内
    // 对于逆时针排列的顶点，如果所有叉积同号，则点在内部
    float c0 = cross2d(v1x - v0x, v1y - v0y, qx - v0x, qy - v0y); // 边0的法向判断
    float c1 = cross2d(v2x - v1x, v2y - v1y, qx - v1x, qy - v1y); // 边1的法向判断
    float c2 = cross2d(v0x - v2x, v0y - v2y, qx - v2x, qy - v2y); // 边2的法向判断

    // 如果所有叉积都非负或都非正，说明点在三角形内
    bool inside = (c0 >= 0 && c1 >= 0 && c2 >= 0) || (c0 <= 0 && c1 <= 0 && c2 <= 0);

    // 内部返回负距离，外部返回正距离
    return inside ? -minDist : minDist;
}

// 功能:统一的SDF计算接口，根据形状类型调用相应的SDF函数
// 输入:点坐标(px, py), 形状中心(cx, cy), 尺寸参数 r, 旋转角度 rotation, 形状类型 shapeType
// 输出:带符号距离
__device__ float computeShapeSDF(float px, float py, float cx, float cy, float r,
                                 float rotation, int shapeType)
{
    // 根据形状类型分发到具体的SDF函数
    switch (shapeType)
    {
    case 0: // 圆形
        return sdfCircle(px, py, cx, cy, r);
    case 1: // 五角星
        return sdfStar(px, py, cx, cy, r, rotation);
    case 2: // 菱形
        return sdfDiamond(px, py, cx, cy, r, rotation);
    case 3: // 胶囊形
        return sdfCapsule(px, py, cx, cy, r, rotation);
    case 4: // 三角形
        return sdfTriangle(px, py, cx, cy, r, rotation);
    default: // 默认返回圆形
        return sdfCircle(px, py, cx, cy, r);
    }
}

// 功能:并行计算全场的SDF并初始化网格类型
// 输入:障碍物参数(位置、大小、旋转、形状), 网格间距 dx,dy, 网格尺寸 nx,ny
// 输出:SDF场 sdf[], 网格类型标记 cell_type[]
__global__ void computeSDFKernel(float *sdf, uint8_t *cell_type,
                                 float obs_x, float obs_y, float obs_r,
                                 float rotation, int shapeType,
                                 float dx, float dy, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny)
        return;

    int idx = j * nx + i;

    // 计算网格单元中心的物理坐标
    float x = (i + 0.5f) * dx;
    float y = (j + 0.5f) * dy;

    // 根据形状类型计算带符号距离
    float dist = computeShapeSDF(x, y, obs_x, obs_y, obs_r, rotation, shapeType);

    sdf[idx] = dist;

    // 根据距离分类网格单元
    // band 是边界层的宽度，通常取1-2个网格间距
    float band = 1.5f * fmaxf(dx, dy);

    if (dist < -band)
    {
        // 深入固体内部 -> 固体单元
        cell_type[idx] = CELL_SOLID;
    }
    else if (dist < band && dist >= -band)
    {
        // 距离边界很近 -> 虚拟单元(Ghost Cell，用于边界条件处理)
        cell_type[idx] = CELL_GHOST;
    }
    else
    {
        // 远离边界 -> 流体单元
        cell_type[idx] = CELL_FLUID;
    }

    // 计算域边界条件覆盖
    if (i == 0)
    {
        // 左边界 -> 流入边界
        cell_type[idx] = CELL_INFLOW;
    }
    else if (i == nx - 1)
    {
        // 右边界 -> 流出边界
        cell_type[idx] = CELL_OUTFLOW;
    }
}

#pragma endregion

#pragma region 物理逻辑操作核函数

#pragma endregion

#pragma region 并行计算任务发射函数
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
#pragma endregion

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