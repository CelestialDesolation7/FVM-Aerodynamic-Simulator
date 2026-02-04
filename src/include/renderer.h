#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>
#include "common.h"

#pragma region 枚举类型定义
// 色图类型枚举
enum class ColormapType {
    JET,        // 蓝-青-绿-黄-红
    HOT,        // 黑-红-黄-白
    PLASMA,     // 深紫-红紫-橙-黄
    INFERNO,    // 黑-深红-橙-黄白
    VIRIDIS     // 深紫-蓝绿-绿黄
};

// 物理场显示类型枚举
enum class FieldType {
    TEMPERATURE,    // 温度场
    PRESSURE,       // 压强场
    DENSITY,        // 密度场
    VELOCITY_MAG,   // 速度大小场
    MACH            // 马赫数场
};
#pragma endregion
#pragma endregion

#pragma region 色图辅助函数
// 全局辅助函数：根据色图类型和归一化值(0~1)计算RGB颜色
// 参数:
//   colormap - 色图类型
//   t - 归一化值 [0, 1]
//   r, g, b - 输出的RGB分量 [0, 1]
inline void getColormapColor(ColormapType colormap, float t, float& r, float& g, float& b) {
    // 将t限制在[0, 1]范围内
    t = std::max(0.0f, std::min(1.0f, t));
    
    switch (colormap) {
        case ColormapType::JET:
        {
            if (t < 0.125f) {
                r = 0;
                g = 0;
                b = 0.5f + 4 * t;
            }
            else if (t < 0.375f) {
                r = 0;
                g = 4 * (t - 0.125f);
                b = 1;
            }
            else if (t < 0.625f) {
                r = 4 * (t - 0.375f);
                g = 1;
                b = 1 - 4 * (t - 0.375f);
            }
            else if (t < 0.875f) {
                r = 1;
                g = 1 - 4 * (t - 0.625f);
                b = 0;
            }
            else {
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
#pragma endregion

#pragma region 渲染器类定义
// 渲染器类：负责使用 OpenGL 进行 CFD 仿真结果的可视化
class Renderer {
public:
    Renderer();
    ~Renderer();
    
    // 初始化渲染器，创建并配置所有 OpenGL 资源
    bool initialize(int width, int height);
    
    // 清理所有已分配的 OpenGL 资源
    void cleanup();
    
    // 更新物理场数据到 GPU 纹理
    // 参数:
    //   data - 主机端物理场数据指针
    //   nx, ny - 网格尺寸
    //   minVal, maxVal - 物理量的最小值和最大值（用于归一化）
    //   type - 要显示的物理场类型
    void updateField(const float* data, int nx, int ny, 
                     float minVal, float maxVal, FieldType type);
    
    // 更新网格类型数据（用于显示障碍物和边界）
    void updateCellTypes(const uint8_t* types, int nx, int ny);
    
    // 渲染当前帧
    void render(const SimParams& params);
    
    // 设置和获取色图类型
    void setColormap(ColormapType cmap);
    ColormapType getColormap() const { return colormap_; }
    
    // 设置和获取矢量显示开关
    void setShowVectors(bool show) { showVectors_ = show; }
    bool getShowVectors() const { return showVectors_; }
    
    // 设置矢量密度（每多少个格子显示一个箭头）
    void setVectorDensity(int density) { vectorDensity_ = std::max(2, density); }
    int getVectorDensity() const { return vectorDensity_; }
    
    // 更新速度场数据（用于矢量可视化）
    void updateVelocityField(const float* u, const float* v, int nx, int ny, float u_inf);
    
    // 设置障碍物显示开关
    void setShowObstacle(bool show) { showObstacle_ = show; }
    
    // 窗口尺寸调整
    void resize(int width, int height);
    
    // 获取窗口尺寸
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    
private:
    // 窗口尺寸
    int width_ = 800;
    int height_ = 600;
    
    // 物理场网格维度
    int nx_ = 0;
    int ny_ = 0;
    
    // OpenGL 纹理对象
    GLuint fieldTexture_ = 0;       // 物理场数据纹理
    GLuint colormapTexture_ = 0;    // 色图查找表纹理（1D）
    GLuint cellTypeTexture_ = 0;    // 网格类型纹理
    
    // OpenGL 着色器程序
    GLuint shaderProgram_ = 0;      // 主渲染着色器
    
    // OpenGL 顶点数组对象和缓冲区对象（主四边形）
    GLuint VAO_ = 0;
    GLuint VBO_ = 0;
    
    // 网格线叠加层的着色器和缓冲区（现用于矢量箭头）
    GLuint gridShaderProgram_ = 0;
    GLuint gridVAO_ = 0;
    GLuint gridVBO_ = 0;
    
    // 障碍物轮廓叠加层的着色器和缓冲区
    GLuint circleShaderProgram_ = 0;
    GLuint circleVAO_ = 0;
    GLuint circleVBO_ = 0;
    
    // 矢量箭头叠加层的着色器和缓冲区
    GLuint vectorShaderProgram_ = 0;
    GLuint vectorVAO_ = 0;
    GLuint vectorVBO_ = 0;
    
    // 渲染设置
    ColormapType colormap_ = ColormapType::JET;  // 当前色图类型
    bool showVectors_ = false;                    // 是否显示速度矢量
    int vectorDensity_ = 20;                      // 矢量箭头密度（每多少格子显示一个）
    bool showObstacle_ = true;                    // 是否显示障碍物轮廓
    
    // 速度场数据（用于矢量可视化）
    std::vector<float> velocityU_;
    std::vector<float> velocityV_;
    float u_inf_ = 1.0f;  // 来流速度，用于归一化箭头长度
    
    // 当前物理场的值域范围
    float minVal_ = 0.0f;
    float maxVal_ = 1.0f;
    FieldType fieldType_ = FieldType::TEMPERATURE;
    
    // 主机端归一化数据缓冲区
    std::vector<float> normalizedData_;
    
    // 着色器创建和编译的工具函数
    bool createShaders();           // 创建主渲染着色器
    bool createGridShader();        // 创建网格线着色器（现用于矢量箭头）
    bool createCircleShader();      // 创建障碍物轮廓着色器
    bool createVectorShader();      // 创建矢量箭头着色器
    GLuint compileShader(const char* source, GLenum type);  // 编译单个着色器
    GLuint linkProgram(GLuint vertShader, GLuint fragShader); // 链接着色器程序
    
    // 色图纹理创建
    void createColormapTexture();
    
    // 各种色图的生成函数
    void generateJetColormap(std::vector<float>& colors);
    void generateHotColormap(std::vector<float>& colors);
    void generatePlasmaColormap(std::vector<float>& colors);
    void generateInfernoColormap(std::vector<float>& colors);
    void generateViridisColormap(std::vector<float>& colors);
};
#pragma endregion
