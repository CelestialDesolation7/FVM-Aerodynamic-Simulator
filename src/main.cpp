// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <filesystem>

#include "common.h"
#include "solver.cuh"
#include "renderer.h"

#ifdef _WIN32
#include <windows.h>
#endif
// clang-format on

#pragma region 常量和全局变量

// 窗口尺寸
int windowWidth = 1600;
int windowHeight = 900;
const float DEFAULT_FONT_SIZE = 18.0f;
const std::string DEFAULT_FONT_PATH = "assets/fonts/msyh.ttc";

// 性能监控
float fps = 0.0f;
float simTimePerStep = 0.0f; // 每步仿真耗时，单位秒
int stepsPerFrame = 1; // 进行多少步仿真计算后再渲染一帧

// 仿真参数和求解器
SimParams params;
CFDSolver solver;

// 主机存储缓冲区
std::vector<float> h_temperature;
std::vector<float> h_pressure;
std::vector<float> h_density;
std::vector<float> h_u;
std::vector<float> h_v;
std::vector<uint8_t> h_cellTypes;

// 可视化
Renderer renderer;
FieldType currentField = FieldType::TEMPERATURE;
const char *colormapNames[] = {"Jet", "Hot", "Plasma", "Inferno", "Viridis"};
int currentColormap = 0;

// 颜色映射范围控制变量
float p_min_ratio = 0.5f;
float p_max_ratio = 5.0f;
float rho_min_ratio = 0.5f;
float rho_max_ratio = 5.0f;
float v_max_ratio = 1.5f;
float mach_max_ratio = 1.5f;

// CUDA-OpenGL互操作控制
bool enableCudaInterop = false;    // 是否启用互操作加速
bool cudaInteropInitialized = false;  // 互操作是否已初始化

#pragma endregion


#pragma region 窗口回调函数
void framebufferSizeCallback(GLFWwindow *window, int width, int height)
{
    windowWidth = width;
    windowHeight = height;
    renderer.resize(width, height);
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    {
        params.paused = !params.paused;
    }
    if (key == GLFW_KEY_R && action == GLFW_PRESS)
    {
        params.t_current = 0.0f;
        params.step = 0;
        solver.reset(params);
    }
}
#pragma endregion


#pragma region 工具函数
void setupImGuiFont(ImGuiIO& io, const std::string& fontPath, float fontSize){
    ImFontConfig fontConfig;
    fontConfig.OversampleH = 3; // 水平过采样
    fontConfig.OversampleV = 1; // 垂直过采样
    fontConfig.PixelSnapH = true; // 像素对齐

    if(std::filesystem::exists(fontPath)){
        io.Fonts->AddFontFromFileTTF(fontPath.c_str(), fontSize, &fontConfig, 
        io.Fonts->GetGlyphRangesChineseFull());
    } else {
        std::cerr << "[警告] 字体文件未找到，请检查assets/fonts，使用默认字体。" << std::endl;
        std::cerr << "程序正在试图查找的字体路径：" << fontPath << std::endl;
        std::cerr << "当前工作目录：" << std::filesystem::current_path() << std::endl;
        io.Fonts->AddFontDefault(&fontConfig);
    }
}
#pragma endregion


#pragma region 求解器设定
// 临时。后续应当改用互操作
void resizeBuffers(){
    size_t size = params.nx * params.ny;
    h_temperature.resize(size);
    h_pressure.resize(size);
    h_density.resize(size);
    h_u.resize(size);
    h_v.resize(size);
    h_cellTypes.resize(size);
}

bool initializeSimulation(){
    params.computeDerived();
    resizeBuffers();
    solver.initialize(params);

    // Get initial cell types
    solver.getCellTypes(h_cellTypes.data());
    renderer.updateCellTypes(h_cellTypes.data(), params.nx, params.ny);
    return true;
}
#pragma endregion


#pragma region 控制面板渲染
void renderUI(){
        ImGui::Begin(u8"有限体积法空气动力学模拟控制面板");
        static float inputFontSize = DEFAULT_FONT_SIZE;
        if(ImGui::SliderFloat(u8"字体大小", &inputFontSize, 20.0f, 32.0f)){
            ImGuiIO& io = ImGui::GetIO();
            io.FontGlobalScale = inputFontSize / DEFAULT_FONT_SIZE;
        };

        if(ImGui::CollapsingHeader("性能监控")){
        ImGui::Text(u8"帧率: %.1f FPS", fps);
        ImGui::Text(u8"单步耗时: %.3f 毫秒", simTimePerStep * 1000.0f);
        ImGui::Text(u8"仿真时间: %.6f 秒", params.t_current);
        ImGui::Text(u8"迭代步数: %d", params.step);
        ImGui::SliderInt(u8"每帧迭代数", &stepsPerFrame, 1, 100);

        ImGui::Separator();

        // GPU 显存信息
        size_t freeMem, totalMem;
        CFDSolver::getGPUMemoryInfo(freeMem, totalMem);
        float usedMem = (totalMem - freeMem) / (1024.0f * 1024.0f);
        float totalMemMB = totalMem / (1024.0f * 1024.0f);
        float memUsagePercent = (totalMem - freeMem) * 100.0f / totalMem;
        
        ImGui::Text(u8"GPU 显存使用");
        ImGui::SameLine();

        // 用一个进度条显示
        // RGBA 绿色->黄色->红色
        ImVec4 barColor = (memUsagePercent < 70.0f) ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) : 
                          (memUsagePercent < 90.0f) ? ImVec4(1.0f, 1.0f, 0.0f, 1.0f) : 
                                                     ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
        char memInfoLabel[64];
        std::snprintf(memInfoLabel, sizeof(memInfoLabel), u8"GPU 显存使用: %.1f MB / %.1f MB (%.1f%%)", usedMem, totalMemMB, memUsagePercent);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, barColor);
        ImGui::ProgressBar(memUsagePercent / 100.0f, ImVec2(200, 0), memInfoLabel);
        ImGui::PopStyleColor();

        size_t simMemory = solver.getSimulationMemoryUsage();
        ImGui::Text(u8"仿真数据占用显存: %.1f MB", simMemory / (1024.0f * 1024.0f));
        
        ImGui::Separator();
        
        // CUDA-OpenGL 互操作开关
        if (ImGui::Checkbox(u8"启用GPU零拷贝加速", &enableCudaInterop)) {
            if (enableCudaInterop && !cudaInteropInitialized) {
                // 尝试初始化互操作
                if (renderer.initCudaInterop(params.nx, params.ny)) {
                    cudaInteropInitialized = true;
                    std::cout << "[主程序] CUDA-OpenGL互操作已启用" << std::endl;
                } else {
                    enableCudaInterop = false;
                    std::cerr << "[主程序] CUDA-OpenGL互操作初始化失败" << std::endl;
                }
            } else if (!enableCudaInterop && cudaInteropInitialized) {
                // 清理互操作资源
                renderer.cleanupCudaInterop();
                cudaInteropInitialized = false;
                std::cout << "[主程序] CUDA-OpenGL互操作已禁用" << std::endl;
            }
        }
        if (cudaInteropInitialized) {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), u8"(已激活)");
        }
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), u8"零拷贝: GPU直接写入显示纹理，跳过CPU中转");
        }
        
        ImGui::Separator();

        if(ImGui::CollapsingHeader(u8"仿真控制")){
            if (ImGui::Button(params.paused ? u8"开始（space）" : u8"暂停（space）")){
                params.paused = !params.paused;
            }
            ImGui::SameLine();
            if (ImGui::Button(u8"重置（R）")){
                solver.reset(params);
                params.t_current = 0.0f;
                params.step = 0;
            }
            ImGui::SameLine();
            if (ImGui::Button(u8"单步（N）") && params.paused){
                if(params.paused){
                    solver.step(params);
                    params.t_current += params.dt;
                    params.step += 1;
                }
            }
        }

        ImGui::Separator();

        if(ImGui::CollapsingHeader(u8"网格设置")){
            static int nx_ui = 1024;
            static int ny_ui = 512;
            
            ImGui::SliderInt(u8"X轴网格分辨率", &nx_ui, 64, 4096);
            ImGui::SliderInt(u8"Y轴网格分辨率", &ny_ui, 32, 4096);

            // 如果调整了，先显示再决定要不要应用修改
            ImGui::Text(u8"当前网格分辨率：%d x %d", params.nx, params.ny);
            if (nx_ui != params.nx || ny_ui != params.ny){
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), u8" -> %d x %d", nx_ui, ny_ui);
            }

            if (ImGui::Button(u8"应用网格尺寸"))
            {
                params.nx = nx_ui;
                params.ny = ny_ui;
                params.computeDerived();
                // 重新初始化仿真（重新分配显存和缓冲区）
                initializeSimulation();
                // 重置时间记录
                params.t_current = 0.0f;
                params.step = 0;
                
                // 如果互操作已启用，需要重新初始化互操作缓冲区
                if (cudaInteropInitialized) {
                    renderer.resizeCudaInterop(params.nx, params.ny);
                }
            }

            ImGui::Text(u8"dx = %.4f m, dy = %.4f m", params.dx, params.dy);
            ImGui::Text(u8"计算域: %.1f x %.1f m", params.domain_width, params.domain_height);
            ImGui::Text(u8"总网格数: %d", params.nx * params.ny);
        }

        ImGui::Separator();

        if (ImGui::CollapsingHeader(u8"来流条件", ImGuiTreeNodeFlags_DefaultOpen))
        {
            bool changed = false;

            changed |= ImGui::SliderFloat(u8"马赫数", &params.mach, 0.01f, 10.0f);
            changed |= ImGui::SliderFloat(u8"来流温度 (K)", &params.T_inf, 200.0f, 400.0f);
            changed |= ImGui::SliderFloat(u8"来流压强 (Pa)", &params.p_inf, 10000.0f, 101325.0f);

            if (changed)
            {
                params.computeDerived();
            }

            ImGui::Text(u8"来流密度 = %.4f kg/m^3", params.rho_inf);
            ImGui::Text(u8"来流速度 = %.1f m/s", params.u_inf);
            ImGui::Text(u8"声速 = %.1f m/s", params.c_inf);

            ImGui::SliderFloat(u8"CFL数", &params.cfl, 0.1f, 0.9f);
            ImGui::Text(u8"时间步长 = %.2e s", params.dt);
        }

        ImGui::Separator();

        if (ImGui::CollapsingHeader(u8"粘性设置 (Navier-Stokes)"))
        {
            bool viscosityChanged = false;

            if (ImGui::Checkbox(u8"启用粘性模拟", &params.enable_viscosity))
            {
                viscosityChanged = true;
            }

            if (params.enable_viscosity)
            {
                ImGui::SliderFloat(u8"扩散CFL数", &params.cfl_visc, 0.1f, 0.5f);

                ImGui::Separator();

                ImGui::Text(u8"壁面边界条件:");

                if (ImGui::Checkbox(u8"绝热壁面", &params.adiabatic_wall))
                {
                    viscosityChanged = true;
                }

                if (!params.adiabatic_wall)
                {
                    if (ImGui::SliderFloat(u8"壁面温度 (K)", &params.T_wall, 200.0f, 1000.0f))
                    {
                        viscosityChanged = true;
                    }
                }

                ImGui::Separator();

                // 计算来流粘性，用Sutherland公式，只代表刚飞进来的气体
                float mu_inf = MU_REF * powf(params.T_inf / T_REF, 1.5f) *
                            (T_REF + S_SUTHERLAND) / (params.T_inf + S_SUTHERLAND);
                // 雷诺值，惯性力与粘性力之比
                float Re = params.rho_inf * params.u_inf * (2.0f * params.obstacle_r) / mu_inf;
                ImGui::Text(u8"雷诺数 Re ≈ %.0f", Re);
                ImGui::Text(u8"来流粘性 μ = %.2e Pa·s", mu_inf);

                ImGui::Separator();
                ImGui::TextWrapped(u8"注意：启用粘性后，计算量增加约50%%。粘性CFL通常比对流CFL更严格。");
            }
            else
            {
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), u8"注意：当前为无粘性欧拉方程求解");
            }

            if (viscosityChanged)
            {
                solver.reset(params);
            }
        }

        ImGui::Separator();

    // 障碍物设置
    if (ImGui::CollapsingHeader(u8"障碍物设置", ImGuiTreeNodeFlags_DefaultOpen))
    {
        bool changed = false;

        // Quick shape buttons
        ImGui::Text(u8"障碍物形状:");
        ImGui::SameLine();
        if (ImGui::Button(u8"圆形"))
        {
            params.obstacle_shape = 0;
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(u8"五角星"))
        {
            params.obstacle_shape = 1;
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(u8"菱形"))
        {
            params.obstacle_shape = 2;
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(u8"胶囊形"))
        {
            params.obstacle_shape = 3;
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(u8"三角形"))
        {
            params.obstacle_shape = 4;
            changed = true;
        }

        ImGui::Separator();

        changed |= ImGui::SliderFloat(u8"中心 X 坐标", &params.obstacle_x, 0.5f, params.domain_width * 0.5f);
        changed |= ImGui::SliderFloat(u8"大小 (半径)", &params.obstacle_r, 0.1f, 1.5f);

        // 障碍物旋转角度
        float rotationDeg = params.obstacle_rotation * 180.0f / 3.14159265f;
        if (ImGui::SliderFloat(u8"旋转角度 (度)", &rotationDeg, -180.0f, 180.0f))
        {
            params.obstacle_rotation = rotationDeg * 3.14159265f / 180.0f;
            changed = true;
        }
        if (ImGui::InputFloat(u8"精确旋转角度 (度)", &rotationDeg))
        {
            params.obstacle_rotation = rotationDeg * 3.14159265f / 180.0f;
            changed = true;
        }

        if (changed)
        {
            params.obstacle_y = params.domain_height / 2.0f;
            solver.reset(params);
            solver.getCellTypes(h_cellTypes.data());
            renderer.updateCellTypes(h_cellTypes.data(), params.nx, params.ny);
        }
    }
    ImGui::Separator();

    if (ImGui::CollapsingHeader(u8"可视化设置", ImGuiTreeNodeFlags_DefaultOpen))
    {
        const char *fieldNames[] = {u8"温度", u8"压强", u8"密度", u8"速度大小", u8"马赫数"};
        int fieldIdx = static_cast<int>(currentField);
        if (ImGui::Combo(u8"显示物理量", &fieldIdx, fieldNames, 5))
        {
            currentField = static_cast<FieldType>(fieldIdx);
        }

        if (ImGui::Combo(u8"色图", &currentColormap, colormapNames, 5))
        {
            renderer.setColormap(static_cast<ColormapType>(currentColormap));
        }

        // 根据当前显示的物理量动态调整范围控制
        ImGui::Separator();
        ImGui::Text(u8"颜色映射范围调整:");
        
        switch (currentField)
        {
        case FieldType::TEMPERATURE:
            ImGui::SliderFloat(u8"温度下限 (K)", &params.T_min, 100.0f, 500.0f);
            ImGui::SliderFloat(u8"温度上限 (K)", &params.T_max, 300.0f, 2000.0f);
            break;
        case FieldType::PRESSURE:
            ImGui::SliderFloat(u8"压强下限 (倍数)", &p_min_ratio, 0.1f, 1.0f);
            ImGui::SliderFloat(u8"压强上限 (倍数)", &p_max_ratio, 1.0f, 10.0f);
            ImGui::Text(u8"实际范围: %.0f - %.0f Pa", params.p_inf * p_min_ratio, params.p_inf * p_max_ratio);
            break;
        case FieldType::DENSITY:
            ImGui::SliderFloat(u8"密度下限 (倍数)", &rho_min_ratio, 0.1f, 1.0f);
            ImGui::SliderFloat(u8"密度上限 (倍数)", &rho_max_ratio, 1.0f, 10.0f);
            ImGui::Text(u8"实际范围: %.3f - %.3f kg/m³", params.rho_inf * rho_min_ratio, params.rho_inf * rho_max_ratio);
            break;
        case FieldType::VELOCITY_MAG:
            ImGui::SliderFloat(u8"速度上限 (倍数)", &v_max_ratio, 0.5f, 3.0f);
            ImGui::Text(u8"实际范围: 0 - %.1f m/s", params.u_inf * v_max_ratio);
            break;
        case FieldType::MACH:
            ImGui::SliderFloat(u8"马赫数上限 (倍数)", &mach_max_ratio, 0.5f, 3.0f);
            ImGui::Text(u8"实际范围: 0 - %.2f", params.mach * mach_max_ratio);
            break;
        }
        ImGui::Separator();

        // 速度矢量显示（仅在速度大小可视化模式下可用）
        if (currentField == FieldType::VELOCITY_MAG) {
            bool showVectors = renderer.getShowVectors();
            if (ImGui::Checkbox(u8"显示速度矢量", &showVectors))
            {
                renderer.setShowVectors(showVectors);
            }
            
            // 如果显示矢量开启，显示密度控制滑块
            if (showVectors) {
                int vectorDensity = renderer.getVectorDensity();
                if (ImGui::SliderInt(u8"矢量箭头间隔", &vectorDensity, 5, 50, u8"%d 格"))
                {
                    renderer.setVectorDensity(vectorDensity);
                }
                ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), u8"（值越小箭头越密集）");
            }
        } else {
            // 非速度可视化模式时，自动关闭矢量显示
            renderer.setShowVectors(false);
            ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), u8"（速度矢量仅在[速度大小]模式下可用）");
        }

        // Statistics
        float maxT = solver.getMaxTemperature();
        float maxMa = solver.getMaxMach();
        ImGui::Text(u8"最高温度: %.1f K", maxT);
        ImGui::Text(u8"最大马赫数: %.2f", maxMa);
    }

    ImGui::Separator();

    // Colorbar
    if (ImGui::CollapsingHeader(u8"色标", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImDrawList *drawList = ImGui::GetWindowDrawList();
        ImVec2 pos = ImGui::GetCursorScreenPos();
        float barWidth = 200.0f;
        float barHeight = 20.0f;

        // Draw colorbar
        for (int i = 0; i < (int)barWidth; i++)
        {
            float t = (float)i / barWidth;
            float r, g, b;
            
            // 使用解耦的颜色映射函数
            getColormapColor(static_cast<ColormapType>(currentColormap), t, r, g, b);
            
            ImU32 color = IM_COL32(r * 255, g * 255, b * 255, 255);
            drawList->AddRectFilled(
                ImVec2(pos.x + i, pos.y),
                ImVec2(pos.x + i + 1, pos.y + barHeight),
                color);
        }

        ImGui::Dummy(ImVec2(barWidth, barHeight + 5));

        // Labels
        const char *unit = "";
        float minVal = params.T_min;
        float maxVal = params.T_max;

        switch (currentField)
        {
        case FieldType::TEMPERATURE:
            unit = "K";
            minVal = params.T_min;
            maxVal = params.T_max;
            break;
        case FieldType::PRESSURE:
            unit = "Pa";
            minVal = params.p_inf * 0.5f;
            maxVal = params.p_inf * 5.0f;
            break;
        case FieldType::DENSITY:
            unit = "kg/m^3";
            minVal = params.rho_inf * 0.5f;
            maxVal = params.rho_inf * 5.0f;
            break;
        case FieldType::VELOCITY_MAG:
            unit = "m/s";
            minVal = 0;
            maxVal = params.u_inf * 1.5f;
            break;
        case FieldType::MACH:
            unit = "";
            minVal = 0;
            maxVal = params.mach * 1.5f;
            break;
        }

        ImGui::Text("%.1f %s", minVal, unit);
        ImGui::SameLine(barWidth - 50);
        ImGui::Text("%.1f %s", maxVal, unit);
    }

    ImGui::Separator();
    ImGui::End();
}
#pragma endregion


#pragma region 可视化函数
// 使用传统CPU拷贝方式更新可视化
void updateVisualizationCPU(){
    float *fieldData = nullptr;
    float minVal, maxVal;

    switch (currentField)
    {
    case FieldType::TEMPERATURE:
        solver.getTemperatureField(h_temperature.data());
        fieldData = h_temperature.data();
        minVal = params.T_min;
        maxVal = params.T_max;
        break;

    case FieldType::PRESSURE:
        solver.getPressureField(h_pressure.data());
        fieldData = h_pressure.data();
        minVal = params.p_inf * p_min_ratio;
        maxVal = params.p_inf * p_max_ratio;
        break;

    case FieldType::DENSITY:
        solver.getDensityField(h_density.data());
        fieldData = h_density.data();
        minVal = params.rho_inf * rho_min_ratio;
        maxVal = params.rho_inf * rho_max_ratio;
        break;

    case FieldType::VELOCITY_MAG:
    {
        solver.getVelocityField(h_u.data(), h_v.data());
        // 计算局部速率
        for (int i = 0; i < params.nx * params.ny; i++)
        {
            h_temperature[i] = sqrtf(h_u[i] * h_u[i] + h_v[i] * h_v[i]);
        }
        fieldData = h_temperature.data();
        minVal = 0.0f;
        maxVal = params.u_inf * v_max_ratio;
        
        // 更新速度场数据用于矢量可视化
        renderer.updateVelocityField(h_u.data(), h_v.data(), params.nx, params.ny, params.u_inf);
        break;
    }

    case FieldType::MACH:
    {
        solver.getVelocityField(h_u.data(), h_v.data());
        solver.getTemperatureField(h_temperature.data());
        // 计算局部马赫数
        for (int i = 0; i < params.nx * params.ny; i++)
        {
            float speed = sqrtf(h_u[i] * h_u[i] + h_v[i] * h_v[i]);
            float c = sqrtf(GAMMA * R_GAS * h_temperature[i]);
            h_pressure[i] = speed / (c + 1e-10f);
        }
        fieldData = h_pressure.data();
        minVal = 0.0f;
        maxVal = params.mach * mach_max_ratio;
        break;
    }
    }

    renderer.updateField(fieldData, params.nx, params.ny, minVal, maxVal, currentField);
}

// 使用CUDA-OpenGL互操作方式更新可视化（零拷贝）
void updateVisualizationInterop(){
    float minVal, maxVal;
    
    // 计算值域范围
    switch (currentField)
    {
    case FieldType::TEMPERATURE:
        minVal = params.T_min;
        maxVal = params.T_max;
        break;
    case FieldType::PRESSURE:
        minVal = params.p_inf * p_min_ratio;
        maxVal = params.p_inf * p_max_ratio;
        break;
    case FieldType::DENSITY:
        minVal = params.rho_inf * rho_min_ratio;
        maxVal = params.rho_inf * rho_max_ratio;
        break;
    case FieldType::VELOCITY_MAG:
        minVal = 0.0f;
        maxVal = params.u_inf * v_max_ratio;
        break;
    case FieldType::MACH:
        minVal = 0.0f;
        maxVal = params.mach * mach_max_ratio;
        break;
    }
    
    // 设置渲染器的场值范围
    renderer.setFieldRange(minVal, maxVal, currentField);
    
    // 映射PBO获取设备指针
    float* devPtr = renderer.mapFieldTexture();
    if (!devPtr) {
        // 映射失败，回退到CPU方式
        updateVisualizationCPU();
        return;
    }
    
    // 根据当前显示的物理量，直接在GPU上拷贝/计算数据到PBO
    switch (currentField)
    {
    case FieldType::TEMPERATURE:
        solver.copyTemperatureToDevice(devPtr);
        break;
    case FieldType::PRESSURE:
        solver.copyPressureToDevice(devPtr);
        break;
    case FieldType::DENSITY:
        solver.copyDensityToDevice(devPtr);
        break;
    case FieldType::VELOCITY_MAG:
        solver.copyVelocityMagnitudeToDevice(devPtr);
        // 速度矢量可视化仍需要CPU数据（箭头绘制在CPU端完成）
        if (renderer.getShowVectors()) {
            solver.getVelocityField(h_u.data(), h_v.data());
            renderer.updateVelocityField(h_u.data(), h_v.data(), params.nx, params.ny, params.u_inf);
        }
        break;
    case FieldType::MACH:
        solver.copyMachToDevice(devPtr);
        break;
    }
    
    // 取消映射，数据自动传输到纹理
    renderer.unmapFieldTexture();
}

// 统一的可视化更新函数
void updateVisualization(){
    if (enableCudaInterop && cudaInteropInitialized) {
        updateVisualizationInterop();
    } else {
        updateVisualizationCPU();
    }
}
#pragma endregion


int main(int argc, char* argv[]){
    // 设置控制台编码为 UTF-8，支持中文输出
    #ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
    #endif
    //  设置工作目录为程序所在目录，方便加载资源文件
    std::filesystem::path exePath = std::filesystem::path(argv[0]).parent_path();
    std::filesystem::current_path(exePath);

    // 初始化GLFW
    if(!glfwInit()){
        std::cerr << "[错误] 程序在初始化GLFW阶段失败并退出" << std::endl;
        system("pause");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);


    GLFWwindow* window = glfwCreateWindow(windowWidth,windowHeight,"FVM空气动力学模拟器",nullptr,nullptr);
    if(!window){
        std::cerr << "[错误] 程序在创建窗口阶段失败并退出";
        system("pause");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // 启用垂直同步

    // 设置回调函数
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetKeyCallback(window, keyCallback);

    // 初始化GLAD
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cerr << "[错误] 程序在初始化GLAD阶段失败并退出";
        system("pause");
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // 初始化ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); 
    (void)io;   //用于避免未使用警告
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // 启用键盘控制
    ImGui::StyleColorsDark();                                 // 设置深色主题
    ImGui_ImplGlfw_InitForOpenGL(window,true);                // ImGui接管GLFW输入
    ImGui_ImplOpenGL3_Init("#version 430 core");    // 指定GLSL版本，编写Shader时会用到
    // 设置中文字体
    setupImGuiFont(io, DEFAULT_FONT_PATH, DEFAULT_FONT_SIZE);


    // 初始化渲染器
    if (!renderer.initialize(windowWidth, windowHeight))
    {
        std::cerr << "[错误] 程序在初始化Renderer阶段失败并退出\n";
        system("pause");
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // 初始化仿真器
    if (!initializeSimulation())
    {
        std::cerr << "[错误] 程序在初始化Solver阶段失败并退出\n";
        system("pause");
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // 初始化时间和帧率
    auto lastTime = std::chrono::high_resolution_clock::now();
    int frameCount = 0;

    // 主渲染循环
    while(!glfwWindowShouldClose(window)){
        auto frameStart = std::chrono::high_resolution_clock::now();
        glfwPollEvents();

       
   
        // 进行一步仿真
        if (!params.paused)
        {
            auto simStart = std::chrono::high_resolution_clock::now();

            for (int i = 0; i < stepsPerFrame; i++)
            {
                // 计算 CFL 限制
                float dt = solver.computeStableTimeStep(params);
                params.dt = dt;
                solver.step(params);
            }

            auto simEnd = std::chrono::high_resolution_clock::now();
            simTimePerStep = std::chrono::duration<float>(simEnd - simStart).count() / stepsPerFrame;
        }

        // 更新可视化数据（主机缓存）
        updateVisualization();

        // 绘制可视化数据
        renderer.render(params);

        // 最后绘制ImGui，防止控制面板被遮挡
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        renderUI();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);

        // 计算帧率
        frameCount++;
        auto currentTime = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(currentTime - lastTime).count();

        if (elapsed >= 0.5f)
        {
            fps = frameCount / elapsed;
            frameCount = 0;
            lastTime = currentTime;
        }
    }

    // 例行清理代码
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    renderer.cleanup();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}