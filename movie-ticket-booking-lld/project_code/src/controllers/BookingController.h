#pragma once
#include <crow.h>
#include <memory>
#include "../services/IBookingService.h"

class BookingController {
public:
    explicit BookingController(std::shared_ptr<IBookingService> bookingService);

    void registerRoutes(crow::SimpleApp& app);

    crow::response handleCreateBooking(const crow::request& req);
    crow::response handleGetBooking(const crow::request& req, int bookingId);

private:
    std::shared_ptr<IBookingService> bookingService_;
};
