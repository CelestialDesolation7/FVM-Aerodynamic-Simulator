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

// -----------------------------------------------------------
// 障碍物轮廓线的顶点着色器
// 作用：绘制障碍物的外形轮廓
// -----------------------------------------------------------
const char *obstacleVertexShader = R"(
#version 330 core
layout (location = 0) in vec2 aPos;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

// -----------------------------------------------------------
// 障碍物轮廓线的片元着色器
// 作用：绘制障碍物的颜色
// -----------------------------------------------------------
const char *obstacleFragmentShader = R"(
#version 330 core
out vec4 FragColor;

uniform vec4 obstacleColor;

void main() {
    FragColor = obstacleColor;
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

    // 创建障碍物轮廓着色器程序
    if (!createObstacleShader())
    {
        fprintf(stderr, "创建障碍物着色器失败\n");
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
    // 创建障碍物轮廓的 VAO/VBO
    glGenVertexArrays(1, &obstacleVAO_);
    glGenBuffers(1, &obstacleVBO_);

    // 创建矢量箭头的 VAO/VBO
    glGenVertexArrays(1, &vectorVAO_);
    glGenBuffers(1, &vectorVBO_);

    // 缓存uniform位置并设置常量 uniform（仅需设置一次）
    loc_minVal_ = glGetUniformLocation(shaderProgram_, "minVal");
    loc_maxVal_ = glGetUniformLocation(shaderProgram_, "maxVal");

    glUseProgram(shaderProgram_);
    glUniform1i(glGetUniformLocation(shaderProgram_, "fieldTexture"), 0);
    glUniform1i(glGetUniformLocation(shaderProgram_, "colormapTexture"), 1);
    glUniform1i(glGetUniformLocation(shaderProgram_, "cellTypeTexture"), 2);

    glUseProgram(obstacleShaderProgram_);
    glUniform4f(glGetUniformLocation(obstacleShaderProgram_, "obstacleColor"),
                0.0f, 0.0f, 0.0f, 1.0f);

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
    if (obstacleShaderProgram_)
        glDeleteProgram(obstacleShaderProgram_);
    if (obstacleVAO_)
        glDeleteVertexArrays(1, &obstacleVAO_);
    if (obstacleVBO_)
        glDeleteBuffers(1, &obstacleVBO_);
    if (vectorShaderProgram_)
        glDeleteProgram(vectorShaderProgram_);
    if (vectorVAO_)
        glDeleteVertexArrays(1, &vectorVAO_);
    if (vectorVBO_)
        glDeleteBuffers(1, &vectorVBO_);

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
    cudaInteropEnabled_ = false;
    mappedSize_ = 0;
    writeIndex_ = 0;
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

bool Renderer::createObstacleShader()
{
    GLuint vertShader = compileShader(obstacleVertexShader, GL_VERTEX_SHADER);
    GLuint fragShader = compileShader(obstacleFragmentShader, GL_FRAGMENT_SHADER);

    if (!vertShader || !fragShader)
        return false;

    obstacleShaderProgram_ = linkProgram(vertShader, fragShader);

    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    return obstacleShaderProgram_ != 0;
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
void Renderer::updateVelocityField(const float *u, const float *v, int nx, int ny, float u_inf)
{
    if (nx <= 0 || ny <= 0)
        return;

    size_t size = static_cast<size_t>(nx) * ny;
    velocityU_.resize(size);
    velocityV_.resize(size);

    std::copy(u, u + size, velocityU_.begin());
    std::copy(v, v + size, velocityV_.begin());

    u_inf_ = u_inf;
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

    // 第二层：渲染障碍物轮廓
    if (showObstacle_ && nx_ > 0 && ny_ > 0)
    {
        glUseProgram(obstacleShaderProgram_);
        // obstacleColor 已在初始化时设置

        // 生成障碍物顶点（在 NDC 坐标系中）
        std::vector<float> obstacleVerts;
        const float PI = 3.14159265f;

        // 从参数中获取障碍物的世界空间坐标
        float obs_cx = params.obstacle_x;
        float obs_cy = params.obstacle_y;
        float obs_r = params.obstacle_r;
        float rotation = params.obstacle_rotation;

        // 辅助函数：将世界坐标转换为 NDC 坐标
        auto worldToNDC = [&](float wx, float wy, float &nx, float &ny)
        {
            nx = (wx / params.domain_width) * 2.0f - 1.0f;
            ny = (wy / params.domain_height) * 2.0f - 1.0f;
        };

        // 辅助函数：在世界空间中对点进行旋转，然后转换为 NDC
        auto addRotatedPoint = [&](float localX, float localY)
        {
            // 在世界空间中进行旋转
            float cosR = cosf(rotation);
            float sinR = sinf(rotation);
            float worldX = obs_cx + localX * cosR - localY * sinR;
            float worldY = obs_cy + localX * sinR + localY * cosR;
            // 转换为 NDC
            float ndcX, ndcY;
            worldToNDC(worldX, worldY, ndcX, ndcY);
            obstacleVerts.push_back(ndcX);
            obstacleVerts.push_back(ndcY);
        };

        // 根据障碍物形状生成顶点
        switch (params.obstacle_shape)
        {
        case 0:
        { // 圆形
            int segments = 64;
            for (int i = 0; i <= segments; i++)
            {
                float angle = 2.0f * PI * i / segments;
                addRotatedPoint(obs_r * cosf(angle), obs_r * sinf(angle));
            }
            break;
        }
        case 1:
        { // 五角星
            int numPoints = 5;
            float outerR = obs_r;
            float innerR = obs_r * 0.38f;
            for (int i = 0; i <= numPoints * 2; i++)
            {
                float angle = PI * i / numPoints - PI / 2.0f;
                float r = (i % 2 == 0) ? outerR : innerR;
                addRotatedPoint(r * cosf(angle), r * sinf(angle));
            }
            // 闭合五角星
            addRotatedPoint(outerR * cosf(-PI / 2.0f), outerR * sinf(-PI / 2.0f));
            break;
        }
        case 2:
        { // 菱形
            float pts[5][2] = {
                {0, 1}, {1, 0}, {0, -1}, {-1, 0}, {0, 1}};
            for (int i = 0; i < 5; i++)
            {
                addRotatedPoint(obs_r * pts[i][0], obs_r * pts[i][1]);
            }
            break;
        }
        case 3:
        { // 胶囊形（圆角矩形）
            int segments = 16;
            float halfW = obs_r * 1.5f;
            float capR = obs_r * 0.5f;
            // 右侧半圆
            for (int i = 0; i <= segments; i++)
            {
                float angle = -PI / 2 + PI * i / segments;
                addRotatedPoint(halfW - capR + capR * cosf(angle), capR * sinf(angle));
            }
            // 左侧半圆
            for (int i = 0; i <= segments; i++)
            {
                float angle = PI / 2 + PI * i / segments;
                addRotatedPoint(-halfW + capR + capR * cosf(angle), capR * sinf(angle));
            }
            // 闭合
            addRotatedPoint(halfW - capR, -capR);
            break;
        }
        case 4:
        { // 三角形（尖端向右）
            float sqrt3_2 = 0.866025404f;
            addRotatedPoint(obs_r, 0);
            addRotatedPoint(-obs_r * 0.5f, obs_r * sqrt3_2);
            addRotatedPoint(-obs_r * 0.5f, -obs_r * sqrt3_2);
            addRotatedPoint(obs_r, 0); // 闭合
            break;
        }
        }

        // 上传障碍物顶点数据
        glBindVertexArray(obstacleVAO_);
        glBindBuffer(GL_ARRAY_BUFFER, obstacleVBO_);
        glBufferData(GL_ARRAY_BUFFER, obstacleVerts.size() * sizeof(float),
                     obstacleVerts.data(), GL_DYNAMIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
        glEnableVertexAttribArray(0);

        // 绘制障碍物轮廓
        glLineWidth(2.0f);
        glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(obstacleVerts.size() / 2));
    }

    // 第三层：渲染速度矢量箭头
    if (showVectors_ && nx_ > 0 && ny_ > 0 && !velocityU_.empty() && !velocityV_.empty())
    {
        glUseProgram(vectorShaderProgram_);

        // 生成矢量箭头顶点数据
        // 格式: x, y, r, g, b (位置 + 颜色)
        std::vector<float> vectorVerts;

        // 计算屏幕宽高比，用于调整箭头显示
        float aspectRatio = (float)width_ / (float)height_;
        float domainAspect = (float)nx_ / (float)ny_;

        // 箭头参数
        const float arrowHeadAngle = 0.5f;  // 箭头头部张角（弧度）
        const float arrowHeadLength = 0.3f; // 箭头头部相对于箭身的长度比例

        // 根据密度设置计算步长
        int step = vectorDensity_;

        // 计算单个格子在NDC中的尺寸
        float cellWidth = 2.0f / nx_;
        float cellHeight = 2.0f / ny_;

        // 箭头最大长度（相对于格子尺寸）
        float maxArrowLength = std::min(cellWidth, cellHeight) * (step * 0.8f);

        // 遍历网格，在每隔step个格子处绘制一个箭头
        for (int j = step / 2; j < ny_; j += step)
        {
            for (int i = step / 2; i < nx_; i += step)
            {
                int idx = j * nx_ + i;

                // 获取该点的速度分量
                float u = velocityU_[idx];
                float v = velocityV_[idx];

                // 计算速度大小
                float speed = sqrtf(u * u + v * v);
                if (speed < 1e-6f * u_inf_)
                    continue; // 跳过速度接近零的点

                // 归一化速度
                float normalizedSpeed = std::min(speed / (u_inf_ * 1.5f), 1.0f);

                // 计算箭头起点（NDC坐标）
                float startX = (float)i / nx_ * 2.0f - 1.0f;
                float startY = (float)j / ny_ * 2.0f - 1.0f;

                // 计算箭头方向和长度
                float dirX = u / speed;
                float dirY = v / speed;
                float arrowLength = maxArrowLength * normalizedSpeed;

                // 箭头终点
                float endX = startX + dirX * arrowLength;
                float endY = startY + dirY * arrowLength;

                // 使用黑色箭头，更加醒目
                float r = 0.0f, g = 0.0f, b = 0.0f;

                // 添加箭身线段
                vectorVerts.insert(vectorVerts.end(), {startX, startY, r, g, b});
                vectorVerts.insert(vectorVerts.end(), {endX, endY, r, g, b});

                // 计算箭头头部的两个点
                float headLen = arrowLength * arrowHeadLength;
                float cosA = cosf(arrowHeadAngle);
                float sinA = sinf(arrowHeadAngle);

                // 旋转箭头方向得到头部两个边
                float head1X = endX - headLen * (dirX * cosA - dirY * sinA);
                float head1Y = endY - headLen * (dirX * sinA + dirY * cosA);
                float head2X = endX - headLen * (dirX * cosA + dirY * sinA);
                float head2Y = endY - headLen * (-dirX * sinA + dirY * cosA);

                // 添加箭头头部两条线段
                vectorVerts.insert(vectorVerts.end(), {endX, endY, r, g, b});
                vectorVerts.insert(vectorVerts.end(), {head1X, head1Y, r, g, b});
                vectorVerts.insert(vectorVerts.end(), {endX, endY, r, g, b});
                vectorVerts.insert(vectorVerts.end(), {head2X, head2Y, r, g, b});
            }
        }

        if (!vectorVerts.empty())
        {
            // 上传矢量箭头顶点数据
            glBindVertexArray(vectorVAO_);
            glBindBuffer(GL_ARRAY_BUFFER, vectorVBO_);
            glBufferData(GL_ARRAY_BUFFER, vectorVerts.size() * sizeof(float),
                         vectorVerts.data(), GL_DYNAMIC_DRAW);

            // 位置属性
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)0);
            glEnableVertexAttribArray(0);
            // 颜色属性
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *)(2 * sizeof(float)));
            glEnableVertexAttribArray(1);

            // 绘制箭头（使用较粗的线条）
            glLineWidth(2.0f);
            glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(vectorVerts.size() / 5));
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
