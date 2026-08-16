#pragma once
#include <optional>

struct Seat {
    std::optional<int> id;
    int screen_id;
    int row_no;
    int col_no;
};
