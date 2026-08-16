#pragma once
#include <optional>
#include <vector>
#include "../models/Seat.h"

class ISeatRepository {
public:
    virtual ~ISeatRepository() = default;
    virtual Seat createSeat(const Seat& seat) = 0;
    virtual std::optional<Seat> getSeat(int id) = 0;
    virtual std::vector<Seat> getSeatsByScreen(int screenId) = 0;
};
