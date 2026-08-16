#pragma once
#include <string>
#include <optional>

struct Show {
    std::optional<int> id;
    int movie_id;
    int screen_id;
    std::string start_time;
    std::string end_time;
};
