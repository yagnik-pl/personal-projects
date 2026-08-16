#pragma once
#include <string>
#include <optional>

struct Movie {
    std::optional<int> id;
    std::string title;
    int duration; // duration in minutes
    std::string language;
};
