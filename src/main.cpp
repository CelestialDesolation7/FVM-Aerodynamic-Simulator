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

#ifdef _WIN32
#include <windows.h>
#endif
// clang-format on

#pragma region 常量和全局变量
#define DEBUG_MODE

int windowWidth = 1600;
int windowHeight = 900;

// 使用 Windows 系统自带的微软雅黑字体，支持中文显示
const std::string DEFAULT_FONT_PATH = "assets/fonts/msyh.ttc";
const float DEFAULT_FONT_SIZE = 18.0f;
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
        ImGui::Begin("控制面板");
        ImGui::Text("这里是控制面板内容");
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