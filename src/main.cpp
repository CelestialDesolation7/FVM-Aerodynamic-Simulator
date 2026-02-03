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
#define DEBUG_MODE

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
SimParams simParams;
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

#pragma endregion

#pragma region 窗口回调函数
void framebufferSizeCallback(GLFWwindow *window, int width, int height)
{
    windowWidth = width;
    windowHeight = height;
    renderer.resize(width, height);
}

#pragma endregion

#pragma region 工具函数
void APIENTRY glDebugOutput(GLenum source, GLenum type, unsigned int id, GLenum severity, 
                            GLsizei length, const char *message, const void *userParam)
{
    // 忽略一些不重要的通知
    // 这几个数字对应的意思是：
    // 131169 - 核心剖析
    // 131185 - 性能剖析
    // 131218 - 着色器编译
    // 131204 - 着色器重定义
    if(id == 131169 || id == 131185 || id == 131218 || id == 131204) return; 

    std::cout << "---------------" << std::endl;
    std::cout << "Debug message (" << id << "): " <<  message << std::endl;
    std::cout << "---------------" << std::endl;
}

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

#pragma region ImGui控制面板渲染
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
        ImGui::Text(u8"仿真时间: %.6f 秒", simParams.t_current);
        ImGui::Text(u8"迭代步数: %d", simParams.step);
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
        }
        
        ImGui::Separator();

        if(ImGui::CollapsingHeader(u8"仿真控制")){
            if (ImGui::Button(simParams.paused ? u8"开始（space）" : u8"暂停（space）")){
                simParams.paused = !simParams.paused;
            }
            ImGui::SameLine();
            if (ImGui::Button(u8"重置（R）")){
                solver.reset(simParams);
                simParams.t_current = 0.0f;
                simParams.step = 0;
            }
            ImGui::SameLine();
            if (ImGui::Button(u8"单步（N）") && simParams.paused){
                if(simParams.paused){
                    solver.step(simParams);
                    simParams.t_current += simParams.dt;
                    simParams.step += 1;
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
            ImGui::Text(u8"当前网格分辨率：%d x %d", simParams.nx, simParams.ny);
            if (nx_ui != simParams.nx || ny_ui != simParams.ny){
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), u8" -> %d x %d", nx_ui, ny_ui);
            }

            if (ImGui::Button(u8"应用网格尺寸"))
            {
                simParams.nx = nx_ui;
                simParams.ny = ny_ui;
                simParams.computeDerived();
                //initializeSimulation();
                // 重置时间记录
                simParams.t_current = 0.0f;
                simParams.step = 0;
            }

            ImGui::Text(u8"dx = %.4f m, dy = %.4f m", simParams.dx, simParams.dy);
            ImGui::Text(u8"计算域: %.1f x %.1f m", simParams.domain_width, simParams.domain_height);
            ImGui::Text(u8"总网格数: %d", simParams.nx * simParams.ny);
        }

        ImGui::Separator();

    if (ImGui::CollapsingHeader(u8"来流条件", ImGuiTreeNodeFlags_DefaultOpen))
    {
        bool changed = false;

        changed |= ImGui::SliderFloat(u8"马赫数", &simParams.mach, 0.01f, 10.0f);
        changed |= ImGui::SliderFloat(u8"来流温度 (K)", &simParams.T_inf, 200.0f, 400.0f);
        changed |= ImGui::SliderFloat(u8"来流压强 (Pa)", &simParams.p_inf, 10000.0f, 101325.0f);

        if (changed)
        {
            simParams.computeDerived();
        }

        ImGui::Text(u8"来流密度 = %.4f kg/m^3", simParams.rho_inf);
        ImGui::Text(u8"来流速度 = %.1f m/s", simParams.u_inf);
        ImGui::Text(u8"声速 = %.1f m/s", simParams.c_inf);

        ImGui::SliderFloat(u8"CFL数", &simParams.cfl, 0.1f, 0.9f);
        ImGui::Text(u8"时间步长 = %.2e s", simParams.dt);
    }

    ImGui::Separator();

    if (ImGui::CollapsingHeader(u8"粘性设置 (Navier-Stokes)"))
    {
        bool viscosityChanged = false;

        if (ImGui::Checkbox(u8"启用粘性模拟", &simParams.enable_viscosity))
        {
            viscosityChanged = true;
        }

        if (simParams.enable_viscosity)
        {
            ImGui::SliderFloat(u8"扩散CFL数", &simParams.cfl_visc, 0.1f, 0.5f);

            ImGui::Separator();

            ImGui::Text(u8"壁面边界条件:");

            if (ImGui::Checkbox(u8"绝热壁面", &simParams.adiabatic_wall))
            {
                viscosityChanged = true;
            }

            if (!simParams.adiabatic_wall)
            {
                if (ImGui::SliderFloat(u8"壁面温度 (K)", &simParams.T_wall, 200.0f, 1000.0f))
                {
                    viscosityChanged = true;
                }
            }

            ImGui::Separator();

            // 计算来流粘性，用Sutherland公式，只代表刚飞进来的气体
            float mu_inf = MU_REF * powf(simParams.T_inf / T_REF, 1.5f) *
                           (T_REF + S_SUTHERLAND) / (simParams.T_inf + S_SUTHERLAND);
            // 雷诺值，惯性力与粘性力之比
            float Re = simParams.rho_inf * simParams.u_inf * (2.0f * simParams.obstacle_r) / mu_inf;
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
            solver.reset(simParams);
        }
    }

    ImGui::Separator();

    // Obstacle settings
    if (ImGui::CollapsingHeader(u8"障碍物设置", ImGuiTreeNodeFlags_DefaultOpen))
    {
        bool changed = false;

        // Quick shape buttons
        ImGui::Text(u8"障碍物形状:");
        ImGui::SameLine();
        if (ImGui::Button(u8"圆形"))
        {
            simParams.obstacle_shape = 0;
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(u8"五角星"))
        {
            simParams.obstacle_shape = 1;
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(u8"菱形"))
        {
            simParams.obstacle_shape = 2;
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(u8"胶囊形"))
        {
            simParams.obstacle_shape = 3;
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button(u8"三角形"))
        {
            simParams.obstacle_shape = 4;
            changed = true;
        }

        ImGui::Separator();

        changed |= ImGui::SliderFloat(u8"中心 X 坐标", &simParams.obstacle_x, 0.5f, simParams.domain_width * 0.5f);
        changed |= ImGui::SliderFloat(u8"大小 (半径)", &simParams.obstacle_r, 0.1f, 1.5f);

        // 障碍物旋转角度
        float rotationDeg = simParams.obstacle_rotation * 180.0f / 3.14159265f;
        if (ImGui::SliderFloat(u8"旋转角度 (度)", &rotationDeg, -180.0f, 180.0f))
        {
            simParams.obstacle_rotation = rotationDeg * 3.14159265f / 180.0f;
            changed = true;
        }
        if (ImGui::InputFloat(u8"精确旋转角度 (度)", &rotationDeg))
        {
            simParams.obstacle_rotation = rotationDeg * 3.14159265f / 180.0f;
            changed = true;
        }

        if (changed)
        {
            simParams.obstacle_y = simParams.domain_height / 2.0f;
            solver.reset(simParams);
            //solver.getCellTypes(h_cellTypes.data());
            //renderer.updateCellTypes(h_cellTypes.data(), simParams.nx, simParams.ny);
        }

        // Shape description
        ImGui::Separator();
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

        ImGui::SliderFloat(u8"温度下限 (K)", &simParams.T_min, 100.0f, 500.0f);
        ImGui::SliderFloat(u8"温度上限 (K)", &simParams.T_max, 300.0f, 2000.0f);

        // Statistics
        float maxT = solver.getMaxTemperature();
        float maxMa = solver.getMaxMach();
        ImGui::Text(u8"最高温度: %.1f K", maxT);
        ImGui::Text(u8"最大马赫数: %.2f", maxMa);
    }

    ImGui::Separator();
    ImGui::End();
    }
#pragma endregion


int main(int argc, char* argv[]){

    // --------------------------
    // 设置控制台编码为 UTF-8，支持中文输出
    #ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
    #endif
    //  设置工作目录为程序所在目录，方便加载资源文件
    std::filesystem::path exePath = std::filesystem::path(argv[0]).parent_path();
    std::filesystem::current_path(exePath);
    // --------------------------
    // 初始化GLFW
    if(!glfwInit()){
        std::cerr << "[错误] 程序在初始化GLFW阶段失败并退出" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    #ifdef DEBUG_MODE
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT,GLFW_TRUE);
    #endif

    GLFWwindow* window = glfwCreateWindow(windowWidth,windowHeight,"程序窗口",nullptr,nullptr);
    if(!window){
        std::cerr << "[错误] 程序在创建窗口阶段失败并退出";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // 启用垂直同步

    // --------------------------
    // 初始化GLAD
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cerr << "[错误] 程序在初始化GLAD阶段失败并退出";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // --------------------------
    // 初始化调试系统
    int flags;
    glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
    if(flags & GL_CONTEXT_FLAG_DEBUG_BIT){
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(glDebugOutput, nullptr);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
        std::cout << "[信息] OpenGL 上下文已处于调试模式" << std::endl;
    }

    //--------------------------
    // 初始化ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); 
    (void)io;   //用于避免未使用警告
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // 启用键盘控制
    ImGui::StyleColorsDark();                                 // 设置深色主题
    ImGui_ImplGlfw_InitForOpenGL(window,true);                // ImGui接管GLFW输入
    ImGui_ImplOpenGL3_Init("#version 430 core");    // 指定GLSL版本，编写Shader时会用到

    //--------------------------
    // 设置中文字体
    setupImGuiFont(io, DEFAULT_FONT_PATH, DEFAULT_FONT_SIZE);


    while(!glfwWindowShouldClose(window)){
        glfwPollEvents();

        // 开始ImGui新帧
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 这里编写你的ImGui界面代码
        renderUI();

        // 渲染ImGui
        ImGui::Render();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f); // 灰青色背景
        glClear(GL_COLOR_BUFFER_BIT); // 清屏

        // 最后绘制ImGui，防止控制面板被遮挡
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}