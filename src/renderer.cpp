#include "renderer.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

// 注意：这里的所有纹理输入数据都在渲染循环或初始化阶段绑定
#pragma region 着色器源码
// -----------------------------------------------------------
// 场数据渲染的顶点着色器
// 作用：绘制一个覆盖全屏的矩形，并传递 UV 坐标
// -----------------------------------------------------------
const char* FIELD_VERTEX_SHADER = R"(
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
const char* FIELD_FRAGMENT_SHADER = R"(
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
    float cellType = texture(cellTypeTexture, TexCoord).r * 255.0;
    
    // 网格类型定义: CELL_SOLID = 1, CELL_GHOST = 2
    // 固体网格: 显示为深灰色
    if (cellType > 0.5 && cellType < 1.5) {
        FragColor = vec4(0.15, 0.15, 0.15, 1.0);
        return;
    }
    
    // 虚拟网格（边界层）: 显示为稍亮的灰色
    // 注释掉以下代码可以查看虚拟网格的物理量，用于调试
    if (cellType > 1.5 && cellType < 2.5) {
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
// 网格线叠加层的顶点着色器
// 作用：绘制网格线条
// -----------------------------------------------------------
const char* gridVertexShader = R"(
#version 330 core
layout (location = 0) in vec2 aPos;

uniform mat4 projection;

void main() {
    gl_Position = projection * vec4(aPos, 0.0, 1.0);
}
)";

// -----------------------------------------------------------
// 网格线叠加层的片元着色器
// 作用：绘制半透明的网格线
// -----------------------------------------------------------
const char* gridFragmentShader = R"(
#version 330 core
out vec4 FragColor;

uniform vec4 gridColor;

void main() {
    FragColor = gridColor;
}
)";

// -----------------------------------------------------------
// 障碍物轮廓线的顶点着色器
// 作用：绘制障碍物的外形轮廓
// -----------------------------------------------------------
const char* circleVertexShader = R"(
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
const char* circleFragmentShader = R"(
#version 330 core
out vec4 FragColor;

uniform vec4 circleColor;

void main() {
    FragColor = circleColor;
}
)";
#pragma endregion

#pragma region 构造和析构函数
Renderer::Renderer() {}

Renderer::~Renderer() {
    cleanup();
}
#pragma endregion

#pragma region 初始化和清理
bool Renderer::initialize(int width, int height) {
    width_ = width;
    height_ = height;
    
    // 创建主着色器程序
    if (!createShaders()) {
        fprintf(stderr, "创建场渲染着色器失败\n");
        return false;
    }
    
    // 创建网格着色器程序
    if (!createGridShader()) {
        fprintf(stderr, "创建网格着色器失败\n");
        return false;
    }
    
    // 创建障碍物轮廓着色器程序
    if (!createCircleShader()) {
        fprintf(stderr, "创建障碍物着色器失败\n");
        return false;
    }
    
    // 创建全屏四边形的 VAO 和 VBO
    float quadVertices[] = {
        // 位置          // 纹理坐标
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    
    glGenVertexArrays(1, &VAO_);
    glGenBuffers(1, &VBO_);
    
    glBindVertexArray(VAO_);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    
    // 属性 0: 位置 (vec2)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // 属性 1: 纹理坐标 (vec2)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
    
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
    
    // 创建网格线的 VAO/VBO（稍后填充数据）
    glGenVertexArrays(1, &gridVAO_);
    glGenBuffers(1, &gridVBO_);
    
    // 创建障碍物轮廓的 VAO/VBO
    glGenVertexArrays(1, &circleVAO_);
    glGenBuffers(1, &circleVBO_);
    
    return true;
}

void Renderer::cleanup() {
    if (fieldTexture_) glDeleteTextures(1, &fieldTexture_);
    if (colormapTexture_) glDeleteTextures(1, &colormapTexture_);
    if (cellTypeTexture_) glDeleteTextures(1, &cellTypeTexture_);
    if (shaderProgram_) glDeleteProgram(shaderProgram_);
    if (VAO_) glDeleteVertexArrays(1, &VAO_);
    if (VBO_) glDeleteBuffers(1, &VBO_);
    if (gridShaderProgram_) glDeleteProgram(gridShaderProgram_);
    if (gridVAO_) glDeleteVertexArrays(1, &gridVAO_);
    if (gridVBO_) glDeleteBuffers(1, &gridVBO_);
    if (circleShaderProgram_) glDeleteProgram(circleShaderProgram_);
    if (circleVAO_) glDeleteVertexArrays(1, &circleVAO_);
    if (circleVBO_) glDeleteBuffers(1, &circleVBO_);
}
#pragma endregion

#pragma region 着色器编译和链接
bool Renderer::createShaders() {
    GLuint vertShader = compileShader(FIELD_VERTEX_SHADER, GL_VERTEX_SHADER);
    GLuint fragShader = compileShader(FIELD_FRAGMENT_SHADER, GL_FRAGMENT_SHADER);
    
    if (!vertShader || !fragShader) return false;
    
    shaderProgram_ = linkProgram(vertShader, fragShader);
    
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);
    
    return shaderProgram_ != 0;
}

bool Renderer::createGridShader() {
    GLuint vertShader = compileShader(gridVertexShader, GL_VERTEX_SHADER);
    GLuint fragShader = compileShader(gridFragmentShader, GL_FRAGMENT_SHADER);
    
    if (!vertShader || !fragShader) return false;
    
    gridShaderProgram_ = linkProgram(vertShader, fragShader);
    
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);
    
    return gridShaderProgram_ != 0;
}

bool Renderer::createCircleShader() {
    GLuint vertShader = compileShader(circleVertexShader, GL_VERTEX_SHADER);
    GLuint fragShader = compileShader(circleFragmentShader, GL_FRAGMENT_SHADER);
    
    if (!vertShader || !fragShader) return false;
    
    circleShaderProgram_ = linkProgram(vertShader, fragShader);
    
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);
    
    return circleShaderProgram_ != 0;
}

GLuint Renderer::compileShader(const char* source, GLenum type) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    // 检查编译错误
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        fprintf(stderr, "着色器编译错误: %s\n", infoLog);
        glDeleteShader(shader);
        return 0;
    }
    
    return shader;
}

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
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        fprintf(stderr, "着色器程序链接错误: %s\n", infoLog);
        glDeleteProgram(program);
        return 0;
    }
    
    return program;
}
#pragma endregion

#pragma region 色图纹理创建和管理
void Renderer::createColormapTexture() {
    std::vector<float> colors;
    
    // 根据当前选中的色图类型生成对应的颜色数据
    switch (colormap_) {
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
    if (colormapTexture_ == 0) {
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
void Renderer::generateJetColormap(std::vector<float>& colors) {
    colors.resize(256 * 3);
    
    for (int i = 0; i < 256; i++) {
        float t = i / 255.0f;
        float r, g, b;
        
        if (t < 0.125f) {
            r = 0.0f;
            g = 0.0f;
            b = 0.5f + 4.0f * t;
        } else if (t < 0.375f) {
            r = 0.0f;
            g = 4.0f * (t - 0.125f);
            b = 1.0f;
        } else if (t < 0.625f) {
            r = 4.0f * (t - 0.375f);
            g = 1.0f;
            b = 1.0f - 4.0f * (t - 0.375f);
        } else if (t < 0.875f) {
            r = 1.0f;
            g = 1.0f - 4.0f * (t - 0.625f);
            b = 0.0f;
        } else {
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

// Plasma 色图生成器: 深紫 -> 红紫 -> 橙 -> 黄
void Renderer::generatePlasmaColormap(std::vector<float>& colors) {
    colors.resize(256 * 3);
    
    // Plasma 色图的多项式近似
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

// Inferno 色图生成器: 黑 -> 深红 -> 橙 -> 黄白
void Renderer::generateInfernoColormap(std::vector<float>& colors) {
    colors.resize(256 * 3);
    
    for (int i = 0; i < 256; i++) {
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
void Renderer::generateViridisColormap(std::vector<float>& colors) {
    colors.resize(256 * 3);
    
    for (int i = 0; i < 256; i++) {
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
void Renderer::setColormap(ColormapType cmap) {
    if (colormap_ != cmap) {
        colormap_ = cmap;
        createColormapTexture();
    }
}
#pragma endregion


#pragma region 数据更新
// 更新物理场数据
void Renderer::updateField(const float* data, int nx, int ny,
                           float minVal, float maxVal, FieldType type) {
    nx_ = nx;
    ny_ = ny;
    minVal_ = minVal;
    maxVal_ = maxVal;
    fieldType_ = type;
    
    // 上传数据到纹理
    glBindTexture(GL_TEXTURE_2D, fieldTexture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, nx, ny, 0, GL_RED, GL_FLOAT, data);
}

// 更新网格类型数据
void Renderer::updateCellTypes(const uint8_t* types, int nx, int ny) {
    // 将 uint8_t 转换为 float 以便纹理上传
    std::vector<float> typeData(nx * ny);
    for (int i = 0; i < nx * ny; i++) {
        typeData[i] = static_cast<float>(types[i]);
    }
    
    glBindTexture(GL_TEXTURE_2D, cellTypeTexture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, nx, ny, 0, GL_RED, GL_FLOAT, typeData.data());
}
#pragma endregion

#pragma region 渲染主循环
void Renderer::render(const SimParams& params) {
    // 清屏
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    // 第一层：渲染物理场
    glUseProgram(shaderProgram_);
    
    // 绑定物理场纹理到纹理单元 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, fieldTexture_);
    glUniform1i(glGetUniformLocation(shaderProgram_, "fieldTexture"), 0);
    
    // 绑定色图纹理到纹理单元 1
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_1D, colormapTexture_);
    glUniform1i(glGetUniformLocation(shaderProgram_, "colormapTexture"), 1);
    
    // 绑定网格类型纹理到纹理单元 2
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, cellTypeTexture_);
    glUniform1i(glGetUniformLocation(shaderProgram_, "cellTypeTexture"), 2);
    
    // 传递归一化所需的最小值和最大值
    glUniform1f(glGetUniformLocation(shaderProgram_, "minVal"), minVal_);
    glUniform1f(glGetUniformLocation(shaderProgram_, "maxVal"), maxVal_);
    
    // 绘制全屏四边形
    glBindVertexArray(VAO_);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    
    // 第二层：渲染障碍物轮廓
    if (showObstacle_ && nx_ > 0 && ny_ > 0) {
        glUseProgram(circleShaderProgram_);
        glUniform4f(glGetUniformLocation(circleShaderProgram_, "circleColor"), 
                    0.0f, 0.0f, 0.0f, 1.0f); // 黑色轮廓
        
        // 生成障碍物顶点（在 NDC 坐标系中）
        std::vector<float> obstacleVerts;
        const float PI = 3.14159265f;
        
        // 从参数中获取障碍物的世界空间坐标
        float obs_cx = params.obstacle_x;
        float obs_cy = params.obstacle_y;
        float obs_r = params.obstacle_r;
        float rotation = params.obstacle_rotation;
        
        // 辅助函数：将世界坐标转换为 NDC 坐标
        auto worldToNDC = [&](float wx, float wy, float& nx, float& ny) {
            nx = (wx / params.domain_width) * 2.0f - 1.0f;
            ny = (wy / params.domain_height) * 2.0f - 1.0f;
        };
        
        // 辅助函数：在世界空间中对点进行旋转，然后转换为 NDC
        auto addRotatedPoint = [&](float localX, float localY) {
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
        switch (params.obstacle_shape) {
            case 0: { // 圆形
                int segments = 64;
                for (int i = 0; i <= segments; i++) {
                    float angle = 2.0f * PI * i / segments;
                    addRotatedPoint(obs_r * cosf(angle), obs_r * sinf(angle));
                }
                break;
            }
            case 1: { // 五角星
                int numPoints = 5;
                float outerR = obs_r;
                float innerR = obs_r * 0.38f;
                for (int i = 0; i <= numPoints * 2; i++) {
                    float angle = PI * i / numPoints - PI / 2.0f;
                    float r = (i % 2 == 0) ? outerR : innerR;
                    addRotatedPoint(r * cosf(angle), r * sinf(angle));
                }
                // 闭合五角星
                addRotatedPoint(outerR * cosf(-PI / 2.0f), outerR * sinf(-PI / 2.0f));
                break;
            }
            case 2: { // 菱形
                float pts[5][2] = {
                    {0, 1}, {1, 0}, {0, -1}, {-1, 0}, {0, 1}
                };
                for (int i = 0; i < 5; i++) {
                    addRotatedPoint(obs_r * pts[i][0], obs_r * pts[i][1]);
                }
                break;
            }
            case 3: { // 胶囊形（圆角矩形）
                int segments = 16;
                float halfW = obs_r * 1.5f;
                float capR = obs_r * 0.5f;
                // 右侧半圆
                for (int i = 0; i <= segments; i++) {
                    float angle = -PI/2 + PI * i / segments;
                    addRotatedPoint(halfW - capR + capR * cosf(angle), capR * sinf(angle));
                }
                // 左侧半圆
                for (int i = 0; i <= segments; i++) {
                    float angle = PI/2 + PI * i / segments;
                    addRotatedPoint(-halfW + capR + capR * cosf(angle), capR * sinf(angle));
                }
                // 闭合
                addRotatedPoint(halfW - capR, -capR);
                break;
            }
            case 4: { // 三角形（尖端向右）
                float sqrt3_2 = 0.866025404f;
                addRotatedPoint(obs_r, 0);
                addRotatedPoint(-obs_r * 0.5f, obs_r * sqrt3_2);
                addRotatedPoint(-obs_r * 0.5f, -obs_r * sqrt3_2);
                addRotatedPoint(obs_r, 0);  // 闭合
                break;
            }
        }
        
        // 上传障碍物顶点数据
        glBindVertexArray(circleVAO_);
        glBindBuffer(GL_ARRAY_BUFFER, circleVBO_);
        glBufferData(GL_ARRAY_BUFFER, obstacleVerts.size() * sizeof(float), 
                     obstacleVerts.data(), GL_DYNAMIC_DRAW);
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        // 绘制障碍物轮廓
        glLineWidth(2.0f);
        glDrawArrays(GL_LINE_STRIP, 0, obstacleVerts.size() / 2);
    }
    
    // 第三层：渲染网格线
    if (showGrid_ && nx_ > 0 && ny_ > 0) {
        glUseProgram(gridShaderProgram_);
        
        // 创建投影矩阵（NDC 坐标系下为单位矩阵）
        float projection[16] = {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        };
        glUniformMatrix4fv(glGetUniformLocation(gridShaderProgram_, "projection"), 
                          1, GL_FALSE, projection);
        glUniform4f(glGetUniformLocation(gridShaderProgram_, "gridColor"), 
                    0.3f, 0.3f, 0.3f, 0.5f); // 半透明灰色
        
        // 生成网格线顶点
        std::vector<float> gridVerts;
        
        // 限制网格线数量以提高性能
        int step = std::max(1, std::min(nx_, ny_) / 50);
        
        // 生成竖直网格线
        for (int i = 0; i <= nx_; i += step) {
            float x = (float)i / nx_ * 2.0f - 1.0f;
            gridVerts.push_back(x);
            gridVerts.push_back(-1.0f);
            gridVerts.push_back(x);
            gridVerts.push_back(1.0f);
        }
        
        // 生成水平网格线
        for (int j = 0; j <= ny_; j += step) {
            float y = (float)j / ny_ * 2.0f - 1.0f;
            gridVerts.push_back(-1.0f);
            gridVerts.push_back(y);
            gridVerts.push_back(1.0f);
            gridVerts.push_back(y);
        }
        
        // 上传网格线顶点数据
        glBindVertexArray(gridVAO_);
        glBindBuffer(GL_ARRAY_BUFFER, gridVBO_);
        glBufferData(GL_ARRAY_BUFFER, gridVerts.size() * sizeof(float), 
                     gridVerts.data(), GL_DYNAMIC_DRAW);
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        // 启用混合以支持半透明
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDrawArrays(GL_LINES, 0, gridVerts.size() / 2);
        glDisable(GL_BLEND);
    }
    
    glBindVertexArray(0);
}

// 窗口尺寸调整
void Renderer::resize(int width, int height) {
    width_ = width;
    height_ = height;
    glViewport(0, 0, width, height);
}
#pragma endregion
