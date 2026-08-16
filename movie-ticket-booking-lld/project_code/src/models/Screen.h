#pragma once
#include <string>
#include <optional>

struct Screen {
    std::optional<int> id;
    std::string name;
    int theatre_id;
};
