#pragma once
#include "IShowSeatRepository.h"
#include "../core/Database.h"
#include <memory>

class ShowSeatRepository : public IShowSeatRepository {
public:
    explicit ShowSeatRepository(std::shared_ptr<Database> db);
    ShowSeat createShowSeat(const ShowSeat& showSeat) override;
    std::vector<ShowSeat> createShowSeats(const std::vector<ShowSeat>& showSeats) override;
    std::optional<ShowSeat> getShowSeat(int showId, int seatId) override;
    std::vector<ShowSeat> getShowSeatsByShow(int showId) override;
    bool lockSeats(int showId, const std::vector<int>& seatIds, int bookingId, int lockDurationSeconds) override;
    bool confirmSeats(int bookingId) override;
    bool releaseSeatsForBooking(int bookingId) override;
    std::vector<int> releaseExpiredSeats() override;

private:
    std::shared_ptr<Database> db_;
};
