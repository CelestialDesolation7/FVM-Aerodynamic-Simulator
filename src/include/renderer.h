#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>
#include "common.h"

// Colormap types
enum class ColormapType {
    JET,
    HOT,
    PLASMA,
    INFERNO,
    VIRIDIS
};

// Field display types
enum class FieldType {
    TEMPERATURE,
    PRESSURE,
    DENSITY,
    VELOCITY_MAG,
    MACH
};

// 全局辅助函数：根据色图类型和归一化值(0~1)计算RGB颜色
// 返回值：RGB各分量范围为[0, 1]
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

class Renderer {
public:
    Renderer();
    ~Renderer();
    
    // 初始化-注册OpenGL资源
    bool initialize(int width, int height);
    // 清理注册的OpenGL资源
    void cleanup();
    
    // 更新
    void updateField(const float* data, int nx, int ny, 
                     float minVal, float maxVal, FieldType type);
    
    // Update cell types for grid overlay
    void updateCellTypes(const uint8_t* types, int nx, int ny);
    
    // Render the field
    void render(const SimParams& params);
    
    // Settings
    void setColormap(ColormapType cmap);
    ColormapType getColormap() const { return colormap_; }
    
    void setShowGrid(bool show) { showGrid_ = show; }
    bool getShowGrid() const { return showGrid_; }
    
    void setShowObstacle(bool show) { showObstacle_ = show; }
    
    // Window resize
    void resize(int width, int height);
    
    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    
private:
    // Window dimensions
    int width_ = 800;
    int height_ = 600;
    
    // Field dimensions
    int nx_ = 0;
    int ny_ = 0;
    
    // OpenGL objects
    GLuint fieldTexture_ = 0;
    GLuint colormapTexture_ = 0;
    GLuint cellTypeTexture_ = 0;
    GLuint shaderProgram_ = 0;
    GLuint VAO_ = 0;
    GLuint VBO_ = 0;
    
    // Grid overlay shader
    GLuint gridShaderProgram_ = 0;
    GLuint gridVAO_ = 0;
    GLuint gridVBO_ = 0;
    
    // Circle overlay shader
    GLuint circleShaderProgram_ = 0;
    GLuint circleVAO_ = 0;
    GLuint circleVBO_ = 0;
    
    // Settings
    ColormapType colormap_ = ColormapType::JET;
    bool showGrid_ = false;
    bool showObstacle_ = true;
    
    // Current field range
    float minVal_ = 0.0f;
    float maxVal_ = 1.0f;
    FieldType fieldType_ = FieldType::TEMPERATURE;
    
    // Host buffer for normalized data
    std::vector<float> normalizedData_;
    
    // Create shaders
    bool createShaders();
    bool createGridShader();
    bool createCircleShader();
    
    // Create colormap texture
    void createColormapTexture();
    
    // Generate colormap data
    void generateJetColormap(std::vector<float>& colors);
    void generateHotColormap(std::vector<float>& colors);
    void generatePlasmaColormap(std::vector<float>& colors);
    void generateInfernoColormap(std::vector<float>& colors);
    void generateViridisColormap(std::vector<float>& colors);
    
    // Compile shader
    GLuint compileShader(const char* source, GLenum type);
    GLuint linkProgram(GLuint vertShader, GLuint fragShader);
};
