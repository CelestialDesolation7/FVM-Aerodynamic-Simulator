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

#pragma endregion

#pragma region 计算链顶层的求解器类内部方法实现

// 功能:分配所有GPU显存
// 说明:根据网格尺寸 _nx, _ny 分配守恒变量、通量、辅助变量等数组
void CFDSolver::allocateMemory()
{
    if (_nx <= 0 || _ny <= 0)
        return;

    size_t size = _nx * _ny * sizeof(float);
    size_t size_byte = _nx * _ny * sizeof(uint8_t);

    // 1. 分配守恒变量(当前时间步)
    CUDA_CHECK(cudaMalloc(&d_rho_, size));
    CUDA_CHECK(cudaMalloc(&d_rho_u_, size));
    CUDA_CHECK(cudaMalloc(&d_rho_v_, size));
    CUDA_CHECK(cudaMalloc(&d_E_, size));
    CUDA_CHECK(cudaMalloc(&d_rho_e_, size)); // 内能(双能量法)

    // 2. 分配守恒变量(下一时间步，用于双缓冲)
    CUDA_CHECK(cudaMalloc(&d_rho_new_, size));
    CUDA_CHECK(cudaMalloc(&d_rho_u_new_, size));
    CUDA_CHECK(cudaMalloc(&d_rho_v_new_, size));
    CUDA_CHECK(cudaMalloc(&d_E_new_, size));
    CUDA_CHECK(cudaMalloc(&d_rho_e_new_, size));

    // 3. 分配原始变量
    CUDA_CHECK(cudaMalloc(&d_u_, size));
    CUDA_CHECK(cudaMalloc(&d_v_, size));
    CUDA_CHECK(cudaMalloc(&d_p_, size));
    CUDA_CHECK(cudaMalloc(&d_T_, size));

    // 4. 分配网格类型和SDF
    CUDA_CHECK(cudaMalloc(&d_cell_type_, size_byte));
    CUDA_CHECK(cudaMalloc(&d_sdf_, size));

    // 5. 分配通量数组
    CUDA_CHECK(cudaMalloc(&d_flux_rho_x_, size));
    CUDA_CHECK(cudaMalloc(&d_flux_rho_u_x_, size));
    CUDA_CHECK(cudaMalloc(&d_flux_rho_v_x_, size));
    CUDA_CHECK(cudaMalloc(&d_flux_E_x_, size));
    CUDA_CHECK(cudaMalloc(&d_flux_rho_e_x_, size)); // 内能通量

    CUDA_CHECK(cudaMalloc(&d_flux_rho_y_, size));
    CUDA_CHECK(cudaMalloc(&d_flux_rho_u_y_, size));
    CUDA_CHECK(cudaMalloc(&d_flux_rho_v_y_, size));
    CUDA_CHECK(cudaMalloc(&d_flux_E_y_, size));
    CUDA_CHECK(cudaMalloc(&d_flux_rho_e_y_, size));

    // 6. 分配归约缓冲区
    CUDA_CHECK(cudaMalloc(&d_reduction_buffer_, sizeof(float)));

    // 7. 分配粘性相关数组(Navier-Stokes)
    CUDA_CHECK(cudaMalloc(&d_mu_, size));     // 动力粘性系数
    CUDA_CHECK(cudaMalloc(&d_k_, size));      // 热导率
    CUDA_CHECK(cudaMalloc(&d_tau_xx_, size)); // 应力张量XX分量
    CUDA_CHECK(cudaMalloc(&d_tau_yy_, size)); // 应力张量YY分量
    CUDA_CHECK(cudaMalloc(&d_tau_xy_, size)); // 应力张量XY分量
    CUDA_CHECK(cudaMalloc(&d_qx_, size));     // X方向热通量
    CUDA_CHECK(cudaMalloc(&d_qy_, size));     // Y方向热通量
}

// 功能:释放所有GPU显存
void CFDSolver::freeMemory()
{
    // 释放守恒变量
    if (d_rho_)
        cudaFree(d_rho_);
    if (d_rho_u_)
        cudaFree(d_rho_u_);
    if (d_rho_v_)
        cudaFree(d_rho_v_);
    if (d_E_)
        cudaFree(d_E_);
    if (d_rho_e_)
        cudaFree(d_rho_e_);

    // 释放双缓冲区
    if (d_rho_new_)
        cudaFree(d_rho_new_);
    if (d_rho_u_new_)
        cudaFree(d_rho_u_new_);
    if (d_rho_v_new_)
        cudaFree(d_rho_v_new_);
    if (d_E_new_)
        cudaFree(d_E_new_);
    if (d_rho_e_new_)
        cudaFree(d_rho_e_new_);

    // 释放原始变量
    if (d_u_)
        cudaFree(d_u_);
    if (d_v_)
        cudaFree(d_v_);
    if (d_p_)
        cudaFree(d_p_);
    if (d_T_)
        cudaFree(d_T_);

    // 释放辅助数据
    if (d_cell_type_)
        cudaFree(d_cell_type_);
    if (d_sdf_)
        cudaFree(d_sdf_);

    // 释放通量数组
    if (d_flux_rho_x_)
        cudaFree(d_flux_rho_x_);
    if (d_flux_rho_u_x_)
        cudaFree(d_flux_rho_u_x_);
    if (d_flux_rho_v_x_)
        cudaFree(d_flux_rho_v_x_);
    if (d_flux_E_x_)
        cudaFree(d_flux_E_x_);
    if (d_flux_rho_e_x_)
        cudaFree(d_flux_rho_e_x_);

    if (d_flux_rho_y_)
        cudaFree(d_flux_rho_y_);
    if (d_flux_rho_u_y_)
        cudaFree(d_flux_rho_u_y_);
    if (d_flux_rho_v_y_)
        cudaFree(d_flux_rho_v_y_);
    if (d_flux_E_y_)
        cudaFree(d_flux_E_y_);
    if (d_flux_rho_e_y_)
        cudaFree(d_flux_rho_e_y_);

    if (d_reduction_buffer_)
        cudaFree(d_reduction_buffer_);

    // 释放粘性相关数组
    if (d_mu_)
        cudaFree(d_mu_);
    if (d_k_)
        cudaFree(d_k_);
    if (d_tau_xx_)
        cudaFree(d_tau_xx_);
    if (d_tau_yy_)
        cudaFree(d_tau_yy_);
    if (d_tau_xy_)
        cudaFree(d_tau_xy_);
    if (d_qx_)
        cudaFree(d_qx_);
    if (d_qy_)
        cudaFree(d_qy_);

    // 置空所有指针(防止重复释放)
    d_rho_ = d_rho_u_ = d_rho_v_ = d_E_ = d_rho_e_ = nullptr;
    d_rho_new_ = d_rho_u_new_ = d_rho_v_new_ = d_E_new_ = d_rho_e_new_ = nullptr;
    d_u_ = d_v_ = d_p_ = d_T_ = nullptr;
    d_cell_type_ = nullptr;
    d_sdf_ = nullptr;
    d_flux_rho_x_ = d_flux_rho_u_x_ = d_flux_rho_v_x_ = d_flux_E_x_ = d_flux_rho_e_x_ = nullptr;
    d_flux_rho_y_ = d_flux_rho_u_y_ = d_flux_rho_v_y_ = d_flux_E_y_ = d_flux_rho_e_y_ = nullptr;
    d_reduction_buffer_ = nullptr;
    d_mu_ = d_k_ = d_tau_xx_ = d_tau_yy_ = d_tau_xy_ = d_qx_ = d_qy_ = nullptr;
}

void CFDSolver::updateCellTypes(const SimParams &params)
{
}

void computeSDF(const SimParams &params)
{
}
#pragma endregion

#pragma region 计算链顶层的求解器类公开接口实现
// 功能:求解器构造函数
// 说明:初始化为空状态，需要调用initialize()分配内存
CFDSolver::CFDSolver() {}

// 功能:求解器析构函数
// 说明:自动释放所有显存
CFDSolver::~CFDSolver()
{
    freeMemory();
}

// 功能:初始化求解器
// 输入:仿真参数 params(包含网格尺寸、来流条件等)
// 说明:设置网格尺寸，分配显存，调用reset初始化流场
void CFDSolver::initialize(const SimParams &params)
{
    _nx = params.nx;
    _ny = params.ny;

    freeMemory();     // 先释放旧内存
    allocateMemory(); // 分配新内存

    reset(params); // 初始化流场
}

// 功能:调整网格分辨率
// 输入:新的网格尺寸 nx, ny
// 说明:如果尺寸改变，重新分配显存
void CFDSolver::resize(int nx, int ny)
{
    if (nx == _nx && ny == _ny)
        return; // 尺寸未变，无需操作

    _nx = nx;
    _ny = ny;

    freeMemory();
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
    // testBlinkKernel<<<gridSize, blockSize>>>(d_T_, _nx, _ny, params.t_current);

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

#pragma region 统计数据的接口

// 功能:获取最大温度(使用CUB库的归约)
float CFDSolver::getMaxTemperature()
{
    return launchComputeMaxTemperature(d_T_, _nx, _ny);
}

// 功能:获取最大马赫数
float CFDSolver::getMaxMach()
{
    return launchComputeMaxMach(d_u_, d_v_, d_p_, d_rho_, _nx, _ny);
}

// 归约核函数实现(使用CUB库)
float launchComputeMaxTemperature(const float *T, int nx, int ny)
{
    float *d_out;
    cudaMalloc(&d_out, sizeof(float));

    // 第一次调用确定临时存储大小
    void *d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceReduce::Max(d_temp, temp_bytes, T, d_out, nx * ny);

    // 分配临时存储
    cudaMalloc(&d_temp, temp_bytes);

    // 第二次调用执行归约
    cub::DeviceReduce::Max(d_temp, temp_bytes, T, d_out, nx * ny);

    // 拷贝结果回主机
    float result;
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_temp);
    cudaFree(d_out);

    return result;
}

#pragma endregion
