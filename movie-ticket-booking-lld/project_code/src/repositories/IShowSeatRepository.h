#pragma once
#include <optional>
#include <vector>
#include <string>
#include "../models/ShowSeat.h"

class IShowSeatRepository {
public:
    virtual ~IShowSeatRepository() = default;
    virtual ShowSeat createShowSeat(const ShowSeat& showSeat) = 0;
    virtual std::vector<ShowSeat> createShowSeats(const std::vector<ShowSeat>& showSeats) = 0;
    virtual std::optional<ShowSeat> getShowSeat(int showId, int seatId) = 0;
    virtual std::vector<ShowSeat> getShowSeatsByShow(int showId) = 0;
    virtual bool lockSeats(int showId, const std::vector<int>& seatIds, int bookingId, int lockDurationSeconds) = 0;
    virtual bool confirmSeats(int bookingId) = 0;
    virtual bool releaseSeatsForBooking(int bookingId) = 0;
    virtual std::vector<int> releaseExpiredSeats() = 0;
};
