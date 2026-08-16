#include "BookingController.h"
#include <vector>

BookingController::BookingController(std::shared_ptr<IBookingService> bookingService)
    : bookingService_(bookingService) {}

void BookingController::registerRoutes(crow::SimpleApp& app) {
    CROW_ROUTE(app, "/api/bookings")
    .methods(crow::HTTPMethod::POST)
    ([this](const crow::request& req) {
        return this->handleCreateBooking(req);
    });

    CROW_ROUTE(app, "/api/bookings/<int>")
    .methods(crow::HTTPMethod::GET)
    ([this](const crow::request& req, int bookingId) {
        return this->handleGetBooking(req, bookingId);
    });
}

crow::response BookingController::handleCreateBooking(const crow::request& req) {
    auto body = crow::json::load(req.body);
    if (!body) {
        crow::json::wvalue res;
        res["error"] = "Invalid JSON body";
        return crow::response(400, res);
    }

    if (!body.has("show_id") || !body.has("user_id") || !body.has("seat_ids")) {
        crow::json::wvalue res;
        res["error"] = "Missing required fields: show_id, user_id, seat_ids";
        return crow::response(400, res);
    }

    if (body["show_id"].t() != crow::json::type::Number ||
        body["user_id"].t() != crow::json::type::Number ||
        body["seat_ids"].t() != crow::json::type::List) {
        crow::json::wvalue res;
        res["error"] = "Invalid field types: show_id and user_id must be integers, seat_ids must be a list";
        return crow::response(400, res);
    }

    int showId = static_cast<int>(body["show_id"].i());
    int userId = static_cast<int>(body["user_id"].i());

    std::vector<int> seatIds;
    for (const auto& item : body["seat_ids"]) {
        if (item.t() != crow::json::type::Number) {
            crow::json::wvalue res;
            res["error"] = "Each seat_id in seat_ids must be an integer";
            return crow::response(400, res);
        }
        seatIds.push_back(static_cast<int>(item.i()));
    }

    BookingResult result = bookingService_->createBooking(userId, showId, seatIds);

    if (result.success) {
        crow::json::wvalue res;
        if (result.bookingId.has_value()) {
            res["booking_id"] = *result.bookingId;
        }
        res["status"] = result.status;
        res["total_price"] = result.totalPrice;
        return crow::response(201, res);
    } else {
        crow::json::wvalue res;
        res["error"] = result.errorMessage;
        return crow::response(result.httpStatusCode, res);
    }
}

crow::response BookingController::handleGetBooking(const crow::request& req, int bookingId) {
    if (bookingId <= 0) {
        crow::json::wvalue res;
        res["error"] = "Booking not found";
        return crow::response(404, res);
    }

    auto bookingOpt = bookingService_->getBooking(bookingId);
    if (!bookingOpt.has_value()) {
        crow::json::wvalue res;
        res["error"] = "Booking not found";
        return crow::response(404, res);
    }

    const auto& b = *bookingOpt;
    crow::json::wvalue res;
    res["booking_id"] = b.id.value_or(bookingId);
    res["show_id"] = b.show_id;
    res["user_id"] = b.user_id;
    res["status"] = b.status;
    res["total_price"] = b.amount;
    if (b.created_at.has_value()) {
        res["created_at"] = *b.created_at;
    }
    return crow::response(200, res);
}
