#pragma once
#include <string>
#include <optional>

struct User {
    std::optional<int> id;
    std::string name;
    std::string email;
    std::string phone;
};
