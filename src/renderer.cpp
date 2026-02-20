#include "renderer.h"
#include <cuda_gl_interop.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <iostream>

// CUDA错误检查宏
#define CUDA_CHECK_INTEROP(call)                                         \
    do                                                                   \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess)                                          \
        {                                                                \
            fprintf(stderr, "[CUDA互操作错误] %s in %s line %i : %s.\n", \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            return false;                                                \
        }                                                                \
    } while (0)

// 注意：这里的所有纹理输入数据都在渲染循环或初始化阶段绑定
#pragma region 着色器源码
// -----------------------------------------------------------
// 场数据渲染的顶点着色器
// 作用：绘制一个覆盖全屏的矩形，并传递 UV 坐标
// -----------------------------------------------------------
const char *FIELD_VERTEX_SHADER = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";

// -----------------------------------------------------------
// 场数据渲染的片元着色器
// 作用：
// 1. 从 fieldTexture 采样物理量的原始值
// 2. 从 cellTypeTexture 读取网格类型（流体/固体/虚拟网格）
// 3. 对固体网格显示深灰色，流体网格根据物理量归一化后查色图
// -----------------------------------------------------------
const char *FIELD_FRAGMENT_SHADER = R"(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D fieldTexture;       // 物理场数据纹理
uniform sampler1D colormapTexture;    // 色图纹理
uniform sampler2D cellTypeTexture;    // 网格类型纹理
uniform float minVal;                 // 物理量最小值（用于归一化）
uniform float maxVal;                 // 物理量最大值（用于归一化）

void main() {
    float value = texture(fieldTexture, TexCoord).r;
    float cellType = texture(cellTypeTexture, TexCoord).r;
    
    // 网格类型定义: CELL_FLUID = 0, CELL_SOLID = 1, CELL_GHOST = 2
    // 将固体网格（内部）和边界网格（Ghost Cell）都显示为深灰色
    if (cellType > 0.5 && cellType < 2.5) {
        FragColor = vec4(0.3, 0.3, 0.3, 1.0);
        return;
    }
    
    // 将物理量值归一化到 [0, 1] 范围
    float normalized = clamp((value - minVal) / (maxVal - minVal + 0.0001), 0.0, 1.0);
    
    // 从色图纹理中查找对应的颜色
    vec3 color = texture(colormapTexture, normalized).rgb;
    
    FragColor = vec4(color, 1.0);
}
)";


// -----------------------------------------------------------
// 矢量箭头的顶点着色器
// -----------------------------------------------------------
const char *vectorVertexShader = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;

out vec3 vertexColor;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    vertexColor = aColor;
}
)";

// -----------------------------------------------------------
// 矢量箭头的片元着色器
// -----------------------------------------------------------
const char *vectorFragmentShader = R"(
#version 330 core
in vec3 vertexColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(vertexColor, 1.0);
}
)";

#pragma endregion

#pragma region 构造和析构函数
Renderer::Renderer()
{
}

Renderer::~Renderer()
{
    cleanup();
}
#pragma endregion

#pragma region 初始化和清理
bool Renderer::initialize(int width, int height)
{
    width_ = width;
    height_ = height;

    // 创建主着色器程序
    if (!createShaders())
    {
        fprintf(stderr, "创建场渲染着色器失败\n");
        return false;
    }

    // 创建矢量箭头着色器程序
    if (!createVectorShader())
    {
        fprintf(stderr, "创建矢量箭头着色器失败\n");
        return false;
        ;
    }

    // 创建全屏四边形的 VAO 和 VBO
    // NDC坐标范围是 [-1, 1]，纹理坐标范围是 [0, 1]
    float quadVertices[] = {
        // 位置          // 纹理坐标
        -1.0f, 1.0f, 0.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
        1.0f, -1.0f, 1.0f, 0.0f,

        -1.0f, 1.0f, 0.0f, 1.0f,
        1.0f, -1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f};

    glGenVertexArrays(1, &VAO_);
    glGenBuffers(1, &VBO_);

    glBindVertexArray(VAO_);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    // 属性 0: 位置 (vec2)
    // 参数 1: 要配置的顶点属性位置值,与顶点着色器中的 layout(location = 0) 对应
    // 参数 2: 属性的组件数量，这里是 vec2，所以是 2
    // 参数 3: 数据类型，这里是 GL_FLOAT
    // 参数 4: 是否归一化，这里是 GL_FALSE
    // 参数 5: 步长，即每个顶点的字节数，这里是 4 * sizeof(float)
    // 参数 6: 偏移量，这里是 0，因为位置数据在每个顶点的开始
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);   // 启用位置属性
    // 属性 1: 纹理坐标 (vec2)
    // 参数 1: 要配置的顶点属性位置值,与顶点着色器中的 layout(location = 1) 对应
    // 参数 2: 属性的组件数量，这里是 vec2，所以是 2
    // 参数 3: 数据类型，这里是 GL_FLOAT
    // 参数 4: 是否归一化，这里是 GL_FALSE
    // 参数 5: 步长，即每个顶点的字节数，这里是 4 * sizeof(float)
    // 参数 6: 偏移量，这里是 2 * sizeof(float)，因为纹理坐标在每个顶点的第二个位置
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);   // 启用纹理坐标属性

    glBindVertexArray(0);   // 解绑VAO

    // 创建物理场纹理
    glGenTextures(1, &fieldTexture_);
    glBindTexture(GL_TEXTURE_2D, fieldTexture_);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // 创建网格类型纹理
    glGenTextures(1, &cellTypeTexture_);
    glBindTexture(GL_TEXTURE_2D, cellTypeTexture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // 创建色图纹理
    createColormapTexture();

    // 以下属性的值在这里无法确定，会在第一次渲染时更新


    // 创建矢量箭头的 VAO/VBO（双缓冲）
    glGenVertexArrays(1, &vectorVAO_);
    glGenBuffers(2, vectorVBO_);

    // 缓存uniform位置并设置常量 uniform（仅需设置一次）
    loc_minVal_ = glGetUniformLocation(shaderProgram_, "minVal");
    loc_maxVal_ = glGetUniformLocation(shaderProgram_, "maxVal");

    glUseProgram(shaderProgram_);
    glUniform1i(glGetUniformLocation(shaderProgram_, "fieldTexture"), 0);
    glUniform1i(glGetUniformLocation(shaderProgram_, "colormapTexture"), 1);
    glUniform1i(glGetUniformLocation(shaderProgram_, "cellTypeTexture"), 2);

    glUseProgram(0);

    return true;
}

void Renderer::cleanup()
{
    if (fieldTexture_)
        glDeleteTextures(1, &fieldTexture_);
    if (colormapTexture_)
        glDeleteTextures(1, &colormapTexture_);
    if (cellTypeTexture_)
        glDeleteTextures(1, &cellTypeTexture_);
    if (shaderProgram_)
        glDeleteProgram(shaderProgram_);
    if (VAO_)
        glDeleteVertexArrays(1, &VAO_);
    if (VBO_)
        glDeleteBuffers(1, &VBO_);
    if (vectorShaderProgram_)
        glDeleteProgram(vectorShaderProgram_);
    if (vectorVAO_)
        glDeleteVertexArrays(1, &vectorVAO_);
    if (vectorVBO_[0] || vectorVBO_[1])
        glDeleteBuffers(2, &vectorVBO_[0]);

    // 清理CUDA互操作资源
    cleanupCudaInterop();
}
#pragma endregion

#pragma region CUDA-OpenGL互操作
// 初始化CUDA-OpenGL互操作（使用双缓冲PBO方法）
bool Renderer::initCudaInterop(int nx, int ny)
{
    // 如果已经启用，先清理
    if (cudaInteropEnabled_)
    {
        cleanupCudaInterop();
    }

    nx_ = nx;
    ny_ = ny;
    size_t bufferSize = nx * ny * sizeof(float);

    // 创建两个PBO用于双缓冲
    glGenBuffers(2, fieldPBO_);

    for (int i = 0; i < 2; i++)
    {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, fieldPBO_[i]);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // 注册PBO为CUDA图形资源
        cudaError_t err = cudaGraphicsGLRegisterBuffer(
            &cudaPBOResource_[i],
            fieldPBO_[i],
            cudaGraphicsMapFlagsWriteDiscard // CUDA只写入
        );

        if (err != cudaSuccess)
        {
            fprintf(stderr, "[CUDA互操作] 注册PBO[%d]失败: %s\n", i, cudaGetErrorString(err));
            // 清理已创建的资源
            for (int j = 0; j <= i; j++)
            {
                if (cudaPBOResource_[j])
                {
                    cudaGraphicsUnregisterResource(cudaPBOResource_[j]);
                    cudaPBOResource_[j] = nullptr;
                }
            }
            glDeleteBuffers(2, fieldPBO_);
            fieldPBO_[0] = fieldPBO_[1] = 0;
            return false;
        }
    }

    // 确保纹理有正确的尺寸
    glBindTexture(GL_TEXTURE_2D, fieldTexture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, nx, ny, 0, GL_RED, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    mappedSize_ = bufferSize;
    writeIndex_ = 0;
    cudaInteropEnabled_ = true;
    std::cout << "[CUDA互操作] 双缓冲PBO初始化成功" << std::endl;
    std::cout << "  网格尺寸: " << nx << " x " << ny << std::endl;
    std::cout << "  每个PBO大小: " << bufferSize / 1024.0f << " KB" << std::endl;
    std::cout << "  总显存占用: " << (bufferSize * 2) / 1024.0f << " KB" << std::endl;
    return true;
}

// 清理CUDA互操作资源
void Renderer::cleanupCudaInterop()
{
    for (int i = 0; i < 2; i++)
    {
        if (cudaPBOResource_[i])
        {
            cudaGraphicsUnregisterResource(cudaPBOResource_[i]);
            cudaPBOResource_[i] = nullptr;
        }
    }
    if (fieldPBO_[0] || fieldPBO_[1])
    {
        glDeleteBuffers(2, fieldPBO_);
        fieldPBO_[0] = fieldPBO_[1] = 0;
    }
    
    // 清理矢量箭头VBO的CUDA资源（双缓冲）
    for (int i = 0; i < 2; i++)
    {
        if (cudaVectorVBOResource_[i])
        {
            cudaGraphicsUnregisterResource(cudaVectorVBOResource_[i]);
            cudaVectorVBOResource_[i] = nullptr;
        }
    }
    
    cudaInteropEnabled_ = false;
    mappedSize_ = 0;
    writeIndex_ = 0;
    vectorVBOCapacity_ = 0;
    vectorVertexCount_[0] = vectorVertexCount_[1] = 0;
    vectorWriteIndex_ = 0;
    vectorVBOMapped_ = false;
}

// 映射当前写入PBO以供CUDA写入，返回设备指针
float *Renderer::mapFieldTexture()
{
    if (!cudaInteropEnabled_ || !cudaPBOResource_[writeIndex_])
    {
        return nullptr;
    }

    // 映射当前writeIndex_对应的CUDA PBO资源，使得OpenGL无法访问它，CUDA可以写入
    cudaError_t err = cudaGraphicsMapResources(1, &cudaPBOResource_[writeIndex_], 0);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[CUDA互操作] 映射PBO[%d]失败: %s\n", writeIndex_, cudaGetErrorString(err));
        return nullptr;
    }

    // 获取设备指针（显卡内部地址）和大小
    float *devPtr = nullptr;
    size_t size = 0;
    err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&devPtr), &size, cudaPBOResource_[writeIndex_]);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[CUDA互操作] 获取设备指针失败: %s\n", cudaGetErrorString(err));
        // 避免资源泄漏导致死锁，立即取消映射
        cudaGraphicsUnmapResources(1, &cudaPBOResource_[writeIndex_], 0);
        return nullptr;
    }

    return devPtr;
}

// 取消映射，交换缓冲区，并将读取PBO数据传输到纹理
void Renderer::unmapFieldTexture()
{
    if (!cudaPBOResource_[writeIndex_])
        return;

    // 取消当前writeIndex_对应的CUDA PBO资源映射，这将允许OpenGL访问它
    cudaGraphicsUnmapResources(1, &cudaPBOResource_[writeIndex_], 0);

    // 交换缓冲区索引：下一帧CUDA写入另一个PBO
    int readIndex = writeIndex_;
    writeIndex_ = 1 - writeIndex_;

    // 将刚写入完成的PBO（现在是读取PBO）的数据传输到纹理
    // 这个传输在GPU内部进行，非常快
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, fieldPBO_[readIndex]);
    glBindTexture(GL_TEXTURE_2D, fieldTexture_);
    // 参数1-纹理目标：由上面这个函数绑定
    // 参数2-mipmap层级（即是否使用缩略图）：0（不使用）
    // 参数3-内部格式：GL_R32F（单通道32位浮点数）
    // 参数4-宽度：nx
    // 参数5-高度：ny
    // 参数6-历史包袱：0（不使用）
    // 参数7-格式：GL_RED（红色通道/单通道）
    // 参数8-源数据的类型：GL_FLOAT（浮点数）
    // 参数9-数据：nullptr（因为数据已经在PBO中，不需要提供CPU指针）
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx_, ny_, GL_RED, GL_FLOAT, nullptr);
    // 解绑
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

// 重新调整互操作缓冲区尺寸
void Renderer::resizeCudaInterop(int nx, int ny)
{
    if (nx != nx_ || ny != ny_)
    {
        if (cudaInteropEnabled_)
        {
            initCudaInterop(nx, ny);
        }
    }
}

// 设置场值范围
void Renderer::setFieldRange(float minVal, float maxVal, FieldType type)
{
    minVal_ = minVal;
    maxVal_ = maxVal;
    fieldType_ = type;
}
#pragma endregion

#pragma region 着色器编译和链接
bool Renderer::createShaders()
{
    GLuint vertShader = compileShader(FIELD_VERTEX_SHADER, GL_VERTEX_SHADER);
    GLuint fragShader = compileShader(FIELD_FRAGMENT_SHADER, GL_FRAGMENT_SHADER);

    if (!vertShader || !fragShader)
        return false;

    shaderProgram_ = linkProgram(vertShader, fragShader);

    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    return shaderProgram_ != 0;
}


bool Renderer::createVectorShader()
{
    GLuint vertShader = compileShader(vectorVertexShader, GL_VERTEX_SHADER);
    GLuint fragShader = compileShader(vectorFragmentShader, GL_FRAGMENT_SHADER);

    if (!vertShader || !fragShader)
        return false;

    vectorShaderProgram_ = linkProgram(vertShader, fragShader);

    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    return vectorShaderProgram_ != 0;
}

GLuint Renderer::compileShader(const char *source, GLenum type)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    // 检查编译错误
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        fprintf(stderr, "着色器编译错误: %s\n", infoLog);
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

GLuint Renderer::linkProgram(GLuint vertShader, GLuint fragShader)
{
    GLuint program = glCreateProgram();
    glAttachShader(program, vertShader);
    glAttachShader(program, fragShader);
    glLinkProgram(program);

    // 检查链接错误
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success)
    {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        fprintf(stderr, "着色器程序链接错误: %s\n", infoLog);
        glDeleteProgram(program);
        return 0;
    }

    return program;
}
#pragma endregion

#pragma region 色图纹理创建和管理
void Renderer::createColormapTexture()
{
    std::vector<float> colors;

    // 根据当前选中的色图类型生成对应的颜色数据
    switch (colormap_)
    {
    case ColormapType::JET:
        generateJetColormap(colors);
        break;
    case ColormapType::HOT:
        generateHotColormap(colors);
        break;
    case ColormapType::PLASMA:
        generatePlasmaColormap(colors);
        break;
    case ColormapType::INFERNO:
        generateInfernoColormap(colors);
        break;
    case ColormapType::VIRIDIS:
        generateViridisColormap(colors);
        break;
    }

    // 如果色图纹理还未创建，则创建
    if (colormapTexture_ == 0)
    {
        glGenTextures(1, &colormapTexture_);
    }

    // 上传色图数据到 1D 纹理
    glBindTexture(GL_TEXTURE_1D, colormapTexture_);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB32F, 256, 0, GL_RGB, GL_FLOAT, colors.data());
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
}
#pragma endregion

#pragma region 色图颜色映射表生成器
// Jet 色图生成器: 蓝 -> 青 -> 绿 -> 黄 -> 红
void Renderer::generateJetColormap(std::vector<float> &colors)
{
    colors.resize(256 * 3);

    for (int i = 0; i < 256; i++)
    {
        float t = i / 255.0f;
        float r, g, b;

        if (t < 0.125f)
        {
            r = 0.0f;
            g = 0.0f;
            b = 0.5f + 4.0f * t;
        }
        else if (t < 0.375f)
        {
            r = 0.0f;
            g = 4.0f * (t - 0.125f);
            b = 1.0f;
        }
        else if (t < 0.625f)
        {
            r = 4.0f * (t - 0.375f);
            g = 1.0f;
            b = 1.0f - 4.0f * (t - 0.375f);
        }
        else if (t < 0.875f)
        {
            r = 1.0f;
            g = 1.0f - 4.0f * (t - 0.625f);
            b = 0.0f;
        }
        else
        {
            r = 1.0f - 4.0f * (t - 0.875f);
            g = 0.0f;
            b = 0.0f;
        }

        colors[i * 3 + 0] = r;
        colors[i * 3 + 1] = g;
        colors[i * 3 + 2] = b;
    }
}

// Hot 色图生成器: 黑 -> 红 -> 黄 -> 白
void Renderer::generateHotColormap(std::vector<float> &colors)
{
    colors.resize(256 * 3);

    for (int i = 0; i < 256; i++)
    {
        float t = i / 255.0f;
        float r = std::min(1.0f, t * 2.5f);
        float g = std::max(0.0f, std::min(1.0f, (t - 0.4f) * 2.5f));
        float b = std::max(0.0f, (t - 0.8f) * 5.0f);

        colors[i * 3 + 0] = r;
        colors[i * 3 + 1] = g;
        colors[i * 3 + 2] = b;
    }
}

// Plasma 色图生成器: 深紫 -> 红紫 -> 橙 -> 黄
void Renderer::generatePlasmaColormap(std::vector<float> &colors)
{
    colors.resize(256 * 3);

    // Plasma 色图的多项式近似
    for (int i = 0; i < 256; i++)
    {
        float t = i / 255.0f;

        float r = 0.050383f + t * (2.028462f + t * (-1.312035f + t * 0.233106f));
        float g = 0.029803f + t * (0.192182f + t * (1.429032f + t * (-1.651143f + t * 0.500000f)));
        float b = 0.527975f + t * (1.573141f + t * (-2.481592f + t * 0.880392f));

        colors[i * 3 + 0] = std::clamp(r, 0.0f, 1.0f);
        colors[i * 3 + 1] = std::clamp(g, 0.0f, 1.0f);
        colors[i * 3 + 2] = std::clamp(b, 0.0f, 1.0f);
    }
}

// Inferno 色图生成器: 黑 -> 深红 -> 橙 -> 黄白
void Renderer::generateInfernoColormap(std::vector<float> &colors)
{
    colors.resize(256 * 3);

    for (int i = 0; i < 256; i++)
    {
        float t = i / 255.0f;

        // Inferno 色图的多项式近似
        float r = t < 0.5f ? 2.0f * t : 1.0f;
        float g = t < 0.25f ? 0.0f : (t < 0.75f ? (t - 0.25f) * 2.0f : 1.0f);
        float b = t < 0.5f ? 0.5f - t : 0.0f;

        // 更好的近似
        r = std::clamp(1.5f * t, 0.0f, 1.0f);
        g = std::clamp(2.0f * (t - 0.25f), 0.0f, 1.0f);
        b = std::clamp(0.6f - 1.2f * t, 0.0f, 0.6f);

        colors[i * 3 + 0] = r;
        colors[i * 3 + 1] = g;
        colors[i * 3 + 2] = b;
    }
}

// Viridis 色图生成器: 深紫 -> 蓝绿 -> 绿黄
void Renderer::generateViridisColormap(std::vector<float> &colors)
{
    colors.resize(256 * 3);

    for (int i = 0; i < 256; i++)
    {
        float t = i / 255.0f;

        // Viridis 色图的多项式近似
        float r = 0.267004f + t * (0.003991f + t * (0.494266f + t * 0.234652f));
        float g = 0.004874f + t * (1.014627f + t * (-0.531556f + t * 0.512042f));
        float b = 0.329415f + t * (1.423671f + t * (-2.252694f + t * 0.499477f));

        colors[i * 3 + 0] = std::clamp(r, 0.0f, 1.0f);
        colors[i * 3 + 1] = std::clamp(g, 0.0f, 1.0f);
        colors[i * 3 + 2] = std::clamp(b, 0.0f, 1.0f);
    }
}

// 切换色图
void Renderer::setColormap(ColormapType cmap)
{
    if (colormap_ != cmap)
    {
        colormap_ = cmap;
        createColormapTexture();
    }
}
#pragma endregion

#pragma region 数据更新
// 更新网格类型数据
void Renderer::updateCellTypes(const uint8_t *types, int nx, int ny)
{
    // 将 uint8_t 转换为 float 以便纹理上传
    std::vector<float> typeData(nx * ny);
    for (int i = 0; i < nx * ny; i++)
    {
        typeData[i] = static_cast<float>(types[i]);
    }

    glBindTexture(GL_TEXTURE_2D, cellTypeTexture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, nx, ny, 0, GL_RED, GL_FLOAT, typeData.data());
}

// 更新速度场数据（用于矢量可视化）
// 确保矢量VBO有足够的容量
void Renderer::ensureVectorVBOCapacity(int requiredVertices)
{
    if (!cudaInteropEnabled_ || vectorVBO_[0] == 0 || vectorVBO_[1] == 0)
    {
        return;
    }
    
    // 如果容量足够，直接返回
    if (vectorVBOCapacity_ >= static_cast<size_t>(requiredVertices))
    {
        return;
    }
    
    // 取消旧的CUDA注册（两个VBO）
    for (int i = 0; i < 2; i++)
    {
        if (cudaVectorVBOResource_[i])
        {
            cudaGraphicsUnregisterResource(cudaVectorVBOResource_[i]);
            cudaVectorVBOResource_[i] = nullptr;
        }
    }
    
    // 重新分配两个VBO（每个顶点5个float：x, y, r, g, b）
    size_t requiredBytes = requiredVertices * 5 * sizeof(float);
    for (int i = 0; i < 2; i++)
    {
        glBindBuffer(GL_ARRAY_BUFFER, vectorVBO_[i]);
        glBufferData(GL_ARRAY_BUFFER, requiredBytes, nullptr, GL_DYNAMIC_DRAW);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    vectorVBOCapacity_ = requiredVertices;
    vectorVertexCount_[0] = vectorVertexCount_[1] = 0;
    
    std::cout << "[信息] 矢量VBO双缓冲重新分配: " << requiredVertices << " 顶点 ("
              << (requiredBytes / 1024.0f / 1024.0f) << " MB × 2)\n";
}

// 映射矢量VBO以供CUDA写入（双缓冲：映射当前writeIndex的VBO）
float *Renderer::mapVectorVBO(int &outMaxVertices)
{
    if (!cudaInteropEnabled_ || vectorVBO_[vectorWriteIndex_] == 0)
    {
        outMaxVertices = 0;
        return nullptr;
    }
    
    if (vectorVBOMapped_)
    {
        std::cerr << "[警告] 矢量VBO已经处于映射状态\n";
        outMaxVertices = 0;
        return nullptr;
    }
    
    // 确保当前writeIndex的VBO已注册
    if (!cudaVectorVBOResource_[vectorWriteIndex_])
    {
        cudaError_t err = cudaGraphicsGLRegisterBuffer(
            &cudaVectorVBOResource_[vectorWriteIndex_],
            vectorVBO_[vectorWriteIndex_],
            cudaGraphicsRegisterFlagsWriteDiscard);
        
        if (err != cudaSuccess)
        {
            std::cerr << "[错误] 注册矢量VBO[" << vectorWriteIndex_ << "]失败: " 
                      << cudaGetErrorString(err) << "\n";
            outMaxVertices = 0;
            return nullptr;
        }
    }
    
    // 映射当前writeIndex的VBO资源
    cudaError_t err = cudaGraphicsMapResources(1, &cudaVectorVBOResource_[vectorWriteIndex_], 0);
    if (err != cudaSuccess)
    {
        std::cerr << "[错误] 映射矢量VBO[" << vectorWriteIndex_ << "]失败: " 
                  << cudaGetErrorString(err) << "\n";
        outMaxVertices = 0;
        return nullptr;
    }
    
    // 获取设备指针
    float *devPtr = nullptr;
    size_t mappedBytes = 0;
    err = cudaGraphicsResourceGetMappedPointer((void **)&devPtr, &mappedBytes,
                                               cudaVectorVBOResource_[vectorWriteIndex_]);
    if (err != cudaSuccess)
    {
        std::cerr << "[错误] 获取矢量VBO[" << vectorWriteIndex_ << "]设备指针失败: " 
                  << cudaGetErrorString(err) << "\n";
        cudaGraphicsUnmapResources(1, &cudaVectorVBOResource_[vectorWriteIndex_], 0);
        outMaxVertices = 0;
        return nullptr;
    }
    
    vectorVBOMapped_ = true;
    outMaxVertices = static_cast<int>(vectorVBOCapacity_);
    return devPtr;
}

// 取消映射矢量VBO，交换缓冲区索引
void Renderer::unmapVectorVBO(int actualVertices)
{
    if (!vectorVBOMapped_)
    {
        return;
    }
    
    // 取消当前writeIndex的VBO映射
    cudaGraphicsUnmapResources(1, &cudaVectorVBOResource_[vectorWriteIndex_], 0);
    vectorVBOMapped_ = false;
    
    // 保存当前VBO的顶点数量
    vectorVertexCount_[vectorWriteIndex_] = actualVertices;
    
    // 交换缓冲区索引：下一次CUDA写入另一个VBO
    vectorWriteIndex_ = 1 - vectorWriteIndex_;
}
#pragma endregion

#pragma region 渲染主循环
void Renderer::render(const SimParams &params)
{
    // 清屏
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // 第一层：渲染物理场
    glUseProgram(shaderProgram_);

    // 绑定物理场纹理到纹理单元 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, fieldTexture_);

    // 绑定色图纹理到纹理单元 1
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_1D, colormapTexture_);

    // 绑定网格类型纹理到纹理单元 2
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, cellTypeTexture_);

    // 传递归一化所需的最小值和最大值（使用缓存的uniform位置）
    glUniform1f(loc_minVal_, minVal_);
    glUniform1f(loc_maxVal_, maxVal_);

    // 绘制全屏四边形
    glBindVertexArray(VAO_);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // 第二层：渲染速度矢量箭头（双缓冲）
    // 注意：箭头顶点数据由solver生成，此处仅负责绘制
    // 绘制readIndex的VBO（CUDA正在写入另一个VBO）
    if (showVectors_)
    {
        int readIndex = 1 - vectorWriteIndex_;
        int vertexCount = vectorVertexCount_[readIndex];
        
        if (vertexCount > 0)
        {
            glUseProgram(vectorShaderProgram_);
            glBindVertexArray(vectorVAO_);
            glBindBuffer(GL_ARRAY_BUFFER, vectorVBO_[readIndex]);
            
            // 配置顶点属性（位置和颜色）
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(2 * sizeof(float)));
            glEnableVertexAttribArray(1);

            // 绘制箭头
            glLineWidth(2.0f);
            glDrawArrays(GL_LINES, 0, vertexCount);
        }
    }

    glBindVertexArray(0);
}

// 窗口尺寸调整
void Renderer::resize(int width, int height)
{
    width_ = width;
    height_ = height;
    glViewport(0, 0, width, height);
}
#pragma endregion
