# pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <string>
#include "common.h"

// 数值-颜色映射方式
enum class ColormapType {
    JET,
    HOT,
    PLASMA,
    INFERNO,
    VIRIDIS
};

enum class FieldType {
    TEMPERATURE,
    PRESSURE,
    DENSITY,
    VELOCITY_MAG,
    MACH
};

class Renderer {
    public:
    Renderer();
    ~Renderer();

    void draw();
    void updateTexture(int width, int height, const std::vector<float>& data);
    void resize(int width, int height);
    void setColormap(ColormapType type);
    bool initRendererData();

    private:
    // 初始化资源
    
    void initShaders();

    // Shader工具函数
    void checkCompileErrors(GLuint shader,std::string type);

    GLuint quadVAO = 0; // 顶点数组对象
    GLuint quadVBO = 0; // 顶点缓冲对象
    GLuint quadEBO = 0; // 索引缓冲对象

    GLuint shaderProgram = 0; // 着色器程序
    GLuint fieldTexture = 0; // 存储物理场数据的纹理
};