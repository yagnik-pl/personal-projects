#pragma once
#include <optional>
#include "../models/Screen.h"

class IScreenRepository {
public:
    virtual ~IScreenRepository() = default;
    virtual Screen createScreen(const Screen& screen) = 0;
    virtual std::optional<Screen> getScreen(int id) = 0;
};
