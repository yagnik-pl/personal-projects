#pragma once
#include "ISeatRepository.h"
#include "../core/Database.h"
#include <memory>

class SeatRepository : public ISeatRepository {
public:
    explicit SeatRepository(std::shared_ptr<Database> db);
    Seat createSeat(const Seat& seat) override;
    std::optional<Seat> getSeat(int id) override;
    std::vector<Seat> getSeatsByScreen(int screenId) override;

private:
    std::shared_ptr<Database> db_;
};
