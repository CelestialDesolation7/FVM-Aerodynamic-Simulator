#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <vector>
#include "common.h"

// 前向声明CUDA图形资源类型，避免在非CUDA文件中包含cuda_gl_interop.h
struct cudaGraphicsResource;

#pragma region 枚举类型定义
// 色图类型枚举
enum class ColormapType
{
    JET,     // 蓝-青-绿-黄-红
    HOT,     // 黑-红-黄-白
    PLASMA,  // 深紫-红紫-橙-黄
    INFERNO, // 黑-深红-橙-黄白
    VIRIDIS  // 深紫-蓝绿-绿黄
};

// 物理场显示类型枚举
enum class FieldType
{
    TEMPERATURE,  // 温度场
    PRESSURE,     // 压强场
    DENSITY,      // 密度场
    VELOCITY_MAG, // 速度大小场
    MACH          // 马赫数场
};
#pragma endregion

#pragma region 色图辅助函数
// 全局辅助函数：根据色图类型和归一化值(0~1)计算RGB颜色
// 参数:
//   colormap - 色图类型
//   t - 归一化值 [0, 1]
//   r, g, b - 输出的RGB分量 [0, 1]
inline void getColormapColor(ColormapType colormap, float t, float &r, float &g, float &b)
{
    // 将t限制在[0, 1]范围内
    t = std::max(0.0f, std::min(1.0f, t));

    switch (colormap)
    {
    case ColormapType::JET:
    {
        if (t < 0.125f)
        {
            r = 0;
            g = 0;
            b = 0.5f + 4 * t;
        }
        else if (t < 0.375f)
        {
            r = 0;
            g = 4 * (t - 0.125f);
            b = 1;
        }
        else if (t < 0.625f)
        {
            r = 4 * (t - 0.375f);
            g = 1;
            b = 1 - 4 * (t - 0.375f);
        }
        else if (t < 0.875f)
        {
            r = 1;
            g = 1 - 4 * (t - 0.625f);
            b = 0;
        }
        else
        {
            r = 1 - 4 * (t - 0.875f);
            g = 0;
            b = 0;
        }
        break;
    }
    case ColormapType::HOT:
    {
        r = std::min(1.0f, t * 2.5f);
        g = std::max(0.0f, std::min(1.0f, (t - 0.4f) * 2.5f));
        b = std::max(0.0f, (t - 0.8f) * 5.0f);
        break;
    }
    case ColormapType::PLASMA:
    case ColormapType::INFERNO:
    case ColormapType::VIRIDIS:
    default:
        // 默认为灰度
        r = g = b = t;
        break;
    }
}
#pragma endregion

#pragma region 渲染器类定义
// 渲染器类：负责使用 OpenGL 进行 CFD 仿真结果的可视化
class Renderer
{
public:
    Renderer();
    ~Renderer();

    // 初始化渲染器，创建并配置所有 OpenGL 资源
    bool initialize(int width, int height);

    // 清理所有已分配的 OpenGL 资源
    void cleanup();

    // 更新网格类型数据（用于显示障碍物和边界）
    void updateCellTypes(const uint8_t *types, int nx, int ny);

    // 渲染当前帧
    void render(const SimParams &params);

    // 设置和获取色图类型
    void setColormap(ColormapType cmap);
    ColormapType getColormap() const { return colormap_; }

    // 设置和获取矢量显示开关
    void setShowVectors(bool show) { showVectors_ = show; }
    bool getShowVectors() const { return showVectors_; }

    // 设置矢量密度（每多少个格子显示一个箭头）
    void setVectorDensity(int density) { vectorDensity_ = std::max(2, density); }
    int getVectorDensity() const { return vectorDensity_; }

    // 确保矢量VBO有足够的容量
    // 如果当前容量不足，会重新分配VBO并重新注册CUDA资源
    // 调用后保证VBO[writeIndex]已映射，可通过getMappedVectorPtr()获取设备指针
    void ensureVectorVBOCapacity(int requiredVertices);

    // 窗口尺寸调整
    void resize(int width, int height);

    // 获取窗口尺寸
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }

    // CUDA-OpenGL 互操作接口（GPU零拷贝路径）

    // 初始化CUDA-OpenGL互操作（在OpenGL初始化之后调用）
    bool initCudaInterop(int nx, int ny);

    // 清理CUDA互操作资源
    void cleanupCudaInterop();

    // 更新互操作纹理的尺寸（网格尺寸变化时调用）
    void resizeCudaInterop(int nx, int ny);

    // 设置场值范围（零拷贝模式下使用，无数据传输）
    void setFieldRange(float minVal, float maxVal, FieldType type);

    // 快照矢量箭头渲染状态，供并行渲染阶段安全使用（安全区域调用）
    void prepareVectorRender();

    // === 直接映射双缓冲接口（零拷贝，无staging中转） ===
    // GL线程预先映射PBO/VBO[writeIndex]，solver线程直接写入映射后的设备指针
    // 写入完成后GL线程提交：unmap → upload/swap → remap下一个缓冲区

    // 获取预映射的PBO设备指针（solver线程直接写入，无需GL上下文）
    float *getMappedFieldPtr() const { return mappedFieldPtr_; }

    // 获取预映射的VBO设备指针和容量（solver线程直接写入）
    float *getMappedVectorPtr() const { return mappedVectorPtr_; }
    int getMappedVectorCapacity() const { return mappedVectorMaxVertices_; }

    // GL线程：映射PBO[writeIndex]，返回设备指针（初始化或提交后调用）
    float *mapFieldForWriting();

    // GL线程：提交已写入的PBO（unmap → 上传纹理 → swap → 映射新PBO）
    // 返回新映射的设备指针（供下一帧solver使用）
    float *submitField();

    // GL线程：映射VBO[vectorWriteIndex]，返回设备指针
    float *mapVectorForWriting();

    // GL线程：提交已写入的VBO（unmap → 记录顶点数 → swap → 映射新VBO）
    // 返回新映射的设备指针
    float *submitVectors(int vertexCount);

    // 检查互操作是否已启用
    bool isCudaInteropEnabled() const { return cudaInteropEnabled_; }

    // 获取网格尺寸
    int getNx() const { return nx_; }
    int getNy() const { return ny_; }

private:
    // 窗口尺寸
    int width_ = 800;
    int height_ = 600;

    // 物理场网格维度
    int nx_ = 0;
    int ny_ = 0;

    // OpenGL 纹理对象
    GLuint fieldTexture_ = 0;    // 物理场数据纹理
    GLuint colormapTexture_ = 0; // 色图查找表纹理（1D）
    GLuint cellTypeTexture_ = 0; // 网格类型纹理

    // OpenGL 着色器程序
    GLuint shaderProgram_ = 0; // 主渲染着色器

    // 缓存的uniform位置（避免每帧字符串查找）
    GLint loc_minVal_ = -1;
    GLint loc_maxVal_ = -1;

    // OpenGL 顶点数组对象和缓冲区对象（主四边形）
    GLuint VAO_ = 0;
    GLuint VBO_ = 0;

    // 矢量箭头叠加层的着色器和缓冲区
    GLuint vectorShaderProgram_ = 0;
    GLuint vectorVAO_ = 0;
    GLuint vectorVBO_[2] = {0, 0};     // 双缓冲VBO
    
    // CUDA-OpenGL互操作：矢量箭头VBO（双缓冲）
    // - 一个用于CUDA写入（vectorWriteIndex_指向）
    // - 另一个用于OpenGL读取（1-vectorWriteIndex_指向）
    cudaGraphicsResource *cudaVectorVBOResource_[2] = {nullptr, nullptr};
    size_t vectorVBOCapacity_ = 0;     // 每个VBO容量（顶点数）
    int vectorVertexCount_[2] = {0, 0}; // 每个VBO实际使用的顶点数
    int vectorWriteIndex_ = 0;          // 当前CUDA写入的VBO索引（0或1）
    bool vectorVBOMapped_ = false;     // VBO是否已映射

    // 缓存的矢量渲染状态（安全区域设置，并行渲染使用）
    int vectorRenderReadIndex_ = 0;
    int vectorRenderVertexCount_ = 0;

    // 渲染设置
    ColormapType colormap_ = ColormapType::JET; // 当前色图类型
    bool showVectors_ = false;                  // 是否显示速度矢量
    int vectorDensity_ = 20;                    // 矢量箭头密度（每多少格子显示一个）


    // 当前物理场的值域范围
    float minVal_ = 0.0f;
    float maxVal_ = 1.0f;
    FieldType fieldType_ = FieldType::TEMPERATURE;

    // CUDA-OpenGL 互操作相关（双缓冲PBO）
    bool cudaInteropEnabled_ = false; // 互操作是否启用

    // 双缓冲PBO：两个PBO交替使用
    // - 一个用于CUDA写入（writeIndex_指向）
    // - 另一个用于OpenGL读取（1-writeIndex_指向）
    GLuint fieldPBO_[2] = {0, 0};                                   // 双缓冲Pixel Buffer Objects
    cudaGraphicsResource *cudaPBOResource_[2] = {nullptr, nullptr}; // 对应的CUDA资源句柄
    int writeIndex_ = 0;                                            // 当前CUDA写入的PBO索引（0或1）
    size_t mappedSize_ = 0;                                         // 单个PBO的缓冲区大小

    // 直接映射状态（GL线程预映射PBO/VBO，solver线程直接写入设备指针）
    float *mappedFieldPtr_ = nullptr;       // 预映射的PBO设备指针
    bool fieldPBOMapped_ = false;           // PBO[writeIndex]是否已映射
    float *mappedVectorPtr_ = nullptr;      // 预映射的VBO设备指针
    int mappedVectorMaxVertices_ = 0;       // 映射的VBO容量（顶点数）

    // 着色器创建和编译的工具函数
    bool createShaders();                                     // 创建主渲染着色器
    bool createVectorShader();                                // 创建矢量箭头着色器
    GLuint compileShader(const char *source, GLenum type);    // 编译单个着色器
    GLuint linkProgram(GLuint vertShader, GLuint fragShader); // 链接着色器程序

    // 色图纹理创建
    void createColormapTexture();

    // 各种色图的生成函数
    void generateJetColormap(std::vector<float> &colors);
    void generateHotColormap(std::vector<float> &colors);
    void generatePlasmaColormap(std::vector<float> &colors);
    void generateInfernoColormap(std::vector<float> &colors);
    void generateViridisColormap(std::vector<float> &colors);
};
#pragma endregion
