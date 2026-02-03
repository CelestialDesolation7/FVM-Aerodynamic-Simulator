# pragma once

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

    void resize(int width, int height);
    void setColormap(ColormapType type);
};