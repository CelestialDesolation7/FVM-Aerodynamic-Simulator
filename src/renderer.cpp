#include "renderer.h"
#include <iostream>
#include <cmath>
#include <algorithm>

#pragma region 内嵌着色器

const char* vertexShaderSource = R"(
#version 430 core
layout (location = 0) in vec3 aPos;     //顶点位置，输入
layout (location = 1) in vec2 aTexCoord; //纹理坐标，输入

out vec2 TexCoord;

void main(){
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
)";

const char* fragmentShaderSource = R"(
#version 430 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D fieldTexture;

void main(){
    // 从纹理中采样数值 (r 分量)
    float value = texture(fieldTexture, TexCoord).r;
    
    // 简单的可视化：数值越大越亮，显示为红色通道
    // 暂时用 0.0 - 1.0 的范围显示灰度
    FragColor = vec4(value, value, value, 1.0);
}
)";

#pragma endregion

Renderer::Renderer()
{
}

Renderer::~Renderer()
{
    // 清理OpenGL资源
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteBuffers(1, &quadEBO);
    glDeleteProgram(shaderProgram);
    glDeleteTextures(1, &fieldTexture);
}

void Renderer::resize(int width, int height)
{
    glViewport(0, 0, width, height);
    // 调整渲染尺寸
}

void Renderer::setColormap(ColormapType type)
{
    // 设置颜色映射
}

bool Renderer::initRendererData() {
    // 定义全屏矩形的顶点数据
    // 格式: X, Y, Z, U, V (U,V是纹理坐标, 范围0-1)
    float vertices[] = {
        // 位置              // 纹理坐标
         1.0f,  1.0f, 0.0f,  1.0f, 1.0f, // 右上
         1.0f, -1.0f, 0.0f,  1.0f, 0.0f, // 右下
        -1.0f, -1.0f, 0.0f,  0.0f, 0.0f, // 左下
        -1.0f,  1.0f, 0.0f,  0.0f, 1.0f  // 左上
    };
    unsigned int indices[] = {  
        0, 1, 3, // 第一个三角形
        1, 2, 3  // 第二个三角形
    };

    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glGenBuffers(1, &quadEBO);

    // 1. 绑定 VAO
    glBindVertexArray(quadVAO);

    // 2. 绑定并填充 VBO (顶点数据)
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // 3. 绑定并填充 EBO (索引数据)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // 4. 设置顶点属性指针
    // 属性0：位置 (3个float)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // 属性1：纹理坐标 (2个float)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // 解绑
    glBindVertexArray(0); 

    initShaders();

    return true;
}

void Renderer::updateTexture(int width, int height, const std::vector<float>& data) {
    if (fieldTexture == 0) {
        glGenTextures(1, &fieldTexture);
    }
    
    glBindTexture(GL_TEXTURE_2D, fieldTexture);

    // 设置纹理环绕方式
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // 设置纹理过滤方式 (线性插值让画面平滑)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // 上传数据
    // internalFormat=GL_R32F (GPU内部存为32位浮点单通道)
    // format=GL_RED (输入数据是单通道)
    // type=GL_FLOAT (输入数据类型是float)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, data.data());
}

void Renderer::draw() {
    if(shaderProgram == 0 || quadVAO == 0) return;

    glUseProgram(shaderProgram);
    
    // 绑定纹理到纹理单元 0
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, fieldTexture);
    // 设置 shader 中的 uniform 变量 fieldTexture 对应纹理单元 0
    glUniform1i(glGetUniformLocation(shaderProgram, "fieldTexture"), 0);

    glBindVertexArray(quadVAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}


void Renderer::initShaders() {
    // 1. 编译顶点着色器
    GLuint vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vertexShaderSource, NULL);
    glCompileShader(vertex);
    checkCompileErrors(vertex, "VERTEX");

    // 2. 编译片元着色器
    GLuint fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragment);
    checkCompileErrors(fragment, "FRAGMENT");

    // 3. 链接程序
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertex);
    glAttachShader(shaderProgram, fragment);
    glLinkProgram(shaderProgram);
    checkCompileErrors(shaderProgram, "PROGRAM");

    // 清理无用的着色器对象
    glDeleteShader(vertex);
    glDeleteShader(fragment);
}

void Renderer::checkCompileErrors(GLuint shader, std::string type) {
    GLint success;
    GLchar infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "SHADER编译错误: " << type << "\n" << infoLog << "\n" << std::endl;
        }
    } else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "SHADER编译错误: " << type << "\n" << infoLog << "\n" << std::endl;
        }
    }
}