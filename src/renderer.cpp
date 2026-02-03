#include "renderer.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>


#pragma region 着色器
// -----------------------------------------------------------
// 顶点着色器 (Vertex Shader)
// 作用：绘制一个覆盖全屏的矩形，并传递 UV 坐标
// -----------------------------------------------------------
const char* VERTEX_SHADER_SOURCE = R"(
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
// 片元着色器 (Fragment Shader)
// 作用：
// 1. 从 fieldTexture 采样归一化的物理量 (value)
// 2. 用 value 去 colormapTexture 查找对应的颜色 (RGB)
// -----------------------------------------------------------
const char* FRAGMENT_SHADER_SOURCE = R"(
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

// 绑定点 0: 物理场数据 (单通道，存的是 0.0~1.0 的数值)
uniform sampler2D fieldTexture;
// 绑定点 1: 色图纹理 (1D 颜色条)
uniform sampler1D colormapTexture;

void main() {
    // 1. 获取物理量值 (只取红色通道，因为我们存的是 R float)
    float value = texture(fieldTexture, TexCoord).r;
    
    // 2. 查表获取颜色
    // clamp 防止超出 [0,1] 导致采样错误
    vec3 color = texture(colormapTexture, clamp(value, 0.0, 1.0)).rgb;
    
    FragColor = vec4(color, 1.0);
}
)";

#pragma endregion

Renderer::Renderer() {
}

Renderer::~Renderer() {
    cleanup();
}

void Renderer::cleanup() {
    if (VAO_) glDeleteVertexArrays(1, &VAO_);
    if (VBO_) glDeleteBuffers(1, &VBO_);
    if (shaderProgram_) glDeleteProgram(shaderProgram_);
    if (fieldTexture_) glDeleteTextures(1, &fieldTexture_);
    if (colormapTexture_) glDeleteTextures(1, &colormapTexture_);
}

bool Renderer::initialize(int width, int height) {
    width_ = width;
    height_ = height;

    // 1. 编译核心着色器
    if (!createShaders()) {
        return false;
    }

    // 2. 创建全屏的坐标 (两个三角形组成一个矩形)
    // 格式: x, y, u, v
    float vertices[] = {
        // 位置          // 纹理坐标 
        -1.0f,  1.0f,   0.0f, 1.0f, // 左上
        -1.0f, -1.0f,   0.0f, 0.0f, // 左下
         1.0f, -1.0f,   1.0f, 0.0f, // 右下

        -1.0f,  1.0f,   0.0f, 1.0f, // 左上
         1.0f, -1.0f,   1.0f, 0.0f, // 右下
         1.0f,  1.0f,   1.0f, 1.0f  // 右上
    };

    // 申请一个VAO
    glGenVertexArrays(1, &VAO_);
    glGenBuffers(1, &VBO_);

    glBindVertexArray(VAO_);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // 属性 0: 位置 (vec2)
    // 属性序号：0  属性所含元素数：2  属性类型：GL_FLOAT  是否标准化：拒绝  内存访问步长：4个float的字节  起始偏移量：0
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // 属性 1: 纹理坐标 (vec2)
    // 属性序号：1  属性所含元素数：2  属性类型：GL_FLOAT  是否标准化：拒绝  内存访问步长：4个float的字节  起始偏移量：2个float的字节
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // 顶点数据设置完成，接触状态机到当前VBO和VAO的绑定
    glBindBuffer(GL_ARRAY_BUFFER, 0); 
    glBindVertexArray(0);

    // 3. 创建纹理
    // (A) 物理场纹理: 使用 GL_R32F (单通道浮点数)
    glGenTextures(1, &fieldTexture_);
    glBindTexture(GL_TEXTURE_2D, fieldTexture_);
    // 设置纹理的环绕：填充到边缘
    // 过滤方式：线性插值
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // (B) 色图纹理
    createColormapTexture();

    return true;
}

// 拦截窗口大小调整，确保renderer内部状态有更新
void Renderer::resize(int width, int height) {
    width_ = width;
    height_ = height;
    glViewport(0, 0, width, height);
}

// 创建完整的shaders
bool Renderer::createShaders() {
    GLuint vertexShader = compileShader(VERTEX_SHADER_SOURCE, GL_VERTEX_SHADER);
    if (!vertexShader) return false;

    GLuint fragmentShader = compileShader(FRAGMENT_SHADER_SOURCE, GL_FRAGMENT_SHADER);
    if (!fragmentShader) return false;

    shaderProgram_ = linkProgram(vertexShader, fragmentShader);
    
    // 链接完成后，Shader对象可以删除了
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    if (!shaderProgram_) return false;
    return true;
}

// 编译一个Shader
GLuint Renderer::compileShader(const char* source, GLenum type) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    // 检查编译错误
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "着色器编译错误 (" << (type == GL_VERTEX_SHADER ? "Vertex" : "Fragment") << "):\n" << infoLog << std::endl;
        return 0;
    }
    return shader;
}

// 链接顶点着色器和片元着色器
GLuint Renderer::linkProgram(GLuint vertShader, GLuint fragShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertShader);
    glAttachShader(program, fragShader);
    glLinkProgram(program);

    // 检查链接错误
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "着色器链接错误:\n" << infoLog << std::endl;
        return 0;
    }
    return program;
}


// 初始化色图纹理
void Renderer::createColormapTexture() {
    // 申请纹理句柄
    glGenTextures(1, &colormapTexture_);
    glBindTexture(GL_TEXTURE_1D, colormapTexture_);
    
    // 生成默认色图 (Jet) 数据
    std::vector<float> colors;
    generateJetColormap(colors); // 后面会实现这个函数

    // 上传到 1D 纹理
    // 256个颜色点, RGB格式, float类型
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB32F, 256, 0, GL_RGB, GL_FLOAT, colors.data());

    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

    setColormap(ColormapType::JET);
}

// 切换色图时调用
void Renderer::setColormap(ColormapType cmap) {
    colormap_ = cmap;
    std::vector<float> colors;

    switch (cmap) {
        case ColormapType::JET: generateJetColormap(colors); break;
        case ColormapType::HOT: generateHotColormap(colors); break;
        // 其他色图暂留空，你可以参考EXAMPLE自行补充
        default: generateJetColormap(colors); break; 
    }

    // 更新纹理数据
    glBindTexture(GL_TEXTURE_1D, colormapTexture_);
    glTexSubImage1D(GL_TEXTURE_1D, 0, 0, 256, GL_RGB, GL_FLOAT, colors.data());
}

#pragma region 色图颜色映射表生成器
void Renderer::generateJetColormap(std::vector<float>& colors) {
    colors.resize(256 * 3);
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        float r = 0, g = 0, b = 0;
        
        // Jet Color Logic
        if (t < 0.125f) { r=0; g=0; b=0.5f + 4*t; }
        else if (t < 0.375f) { r=0; g=4*(t-0.125f); b=1; }
        else if (t < 0.625f) { r=4*(t-0.375f); g=1; b=1-4*(t-0.375f); }
        else if (t < 0.875f) { r=1; g=1-4*(t-0.625f); b=0; }
        else { r=1-4*(t-0.875f); g=0; b=0; }

        colors[i*3 + 0] = r;
        colors[i*3 + 1] = g;
        colors[i*3 + 2] = b;
    }
}

void Renderer::generateHotColormap(std::vector<float>& colors) {
    colors.resize(256 * 3);
    
    for (int i = 0; i < 256; i++) {
        float t = i / 255.0f;
        float r = std::min(1.0f, t * 2.5f);
        float g = std::max(0.0f, std::min(1.0f, (t - 0.4f) * 2.5f));
        float b = std::max(0.0f, (t - 0.8f) * 5.0f);
        
        colors[i * 3 + 0] = r;
        colors[i * 3 + 1] = g;
        colors[i * 3 + 2] = b;
    }
}

void Renderer::generatePlasmaColormap(std::vector<float>& colors) {
    colors.resize(256 * 3);
    
    // Plasma colormap approximation
    for (int i = 0; i < 256; i++) {
        float t = i / 255.0f;
        
        float r = 0.050383f + t * (2.028462f + t * (-1.312035f + t * 0.233106f));
        float g = 0.029803f + t * (0.192182f + t * (1.429032f + t * (-1.651143f + t * 0.500000f)));
        float b = 0.527975f + t * (1.573141f + t * (-2.481592f + t * 0.880392f));
        
        colors[i * 3 + 0] = std::clamp(r, 0.0f, 1.0f);
        colors[i * 3 + 1] = std::clamp(g, 0.0f, 1.0f);
        colors[i * 3 + 2] = std::clamp(b, 0.0f, 1.0f);
    }
}

void Renderer::generateInfernoColormap(std::vector<float>& colors) {
    colors.resize(256 * 3);
    
    for (int i = 0; i < 256; i++) {
        float t = i / 255.0f;
        
        // Inferno colormap approximation
        float r = t < 0.5f ? 2.0f * t : 1.0f;
        float g = t < 0.25f ? 0.0f : (t < 0.75f ? (t - 0.25f) * 2.0f : 1.0f);
        float b = t < 0.5f ? 0.5f - t : 0.0f;
        
        // Better approximation
        r = std::clamp(1.5f * t, 0.0f, 1.0f);
        g = std::clamp(2.0f * (t - 0.25f), 0.0f, 1.0f);
        b = std::clamp(0.6f - 1.2f * t, 0.0f, 0.6f);
        
        colors[i * 3 + 0] = r;
        colors[i * 3 + 1] = g;
        colors[i * 3 + 2] = b;
    }
}

void Renderer::generateViridisColormap(std::vector<float>& colors) {
    colors.resize(256 * 3);
    
    for (int i = 0; i < 256; i++) {
        float t = i / 255.0f;
        
        // Viridis colormap approximation
        float r = 0.267004f + t * (0.003991f + t * (0.494266f + t * 0.234652f));
        float g = 0.004874f + t * (1.014627f + t * (-0.531556f + t * 0.512042f));
        float b = 0.329415f + t * (1.423671f + t * (-2.252694f + t * 0.499477f));
        
        colors[i * 3 + 0] = std::clamp(r, 0.0f, 1.0f);
        colors[i * 3 + 1] = std::clamp(g, 0.0f, 1.0f);
        colors[i * 3 + 2] = std::clamp(b, 0.0f, 1.0f);
    }
}
#pragma endregion

void Renderer::updateField(const float* data, int nx, int ny, float minVal, float maxVal, FieldType type) {
    if (!data) return;

    // 如果网格尺寸变了，需要重新重置纹理大小
    if (nx != nx_ || ny != ny_) {
        nx_ = nx;
        ny_ = ny;
        glBindTexture(GL_TEXTURE_2D, fieldTexture_);
        // 重新分配显存大小，这里用 nullptr 表示只分配不传数据
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, nx, ny, 0, GL_RED, GL_FLOAT, nullptr);
        
        // 调整 host 端 buffer 大小
        normalizedData_.resize(nx * ny);
    }

    // CPU 端归一化处理
    float range = maxVal - minVal;
    if (range < 1e-5f) range = 1e-5f; // 防止除零

    #pragma omp parallel for
    for (int i = 0; i < nx * ny; ++i) {
        float val = data[i];
        // 简单的线性映射
        float norm = (val - minVal) / range;
        normalizedData_[i] = norm; 
    }

    // 上传处理好的数据到纹理
    glBindTexture(GL_TEXTURE_2D, fieldTexture_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, ny, GL_RED, GL_FLOAT, normalizedData_.data());
}

// 占位函数，目前先留空
void Renderer::updateCellTypes(const uint8_t* types, int nx, int ny) {
    // 以后用来画障碍物
}

bool Renderer::createGridShader() { return true; } // 占位
bool Renderer::createCircleShader() { return true; } // 占位

void Renderer::render(const SimParams& params) {
    // 1. 设置背景色
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // 2. 激活 Shader
    glUseProgram(shaderProgram_);

    // 3. 绑定纹理到对应的 Unit
    // Unit 0 -> 物理场
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, fieldTexture_);
    glUniform1i(glGetUniformLocation(shaderProgram_, "fieldTexture"), 0);

    // Unit 1 -> 色图
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_1D, colormapTexture_);
    glUniform1i(glGetUniformLocation(shaderProgram_, "colormapTexture"), 1);

    // 4. 绘制全屏矩形
    glBindVertexArray(VAO_);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    // 如果有 Grid 或 Overlay 的绘制逻辑放在这里
}