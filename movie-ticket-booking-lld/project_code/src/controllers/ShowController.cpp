#include "ShowController.h"

ShowController::ShowController(std::shared_ptr<IShowService> showService)
    : showService_(showService) {}

void ShowController::registerRoutes(crow::SimpleApp& app) {
    CROW_ROUTE(app, "/api/shows")
    .methods(crow::HTTPMethod::GET)
    ([this](const crow::request& req) {
        return this->handleGetAllShows(req);
    });

    CROW_ROUTE(app, "/api/shows/<int>/seats")
    .methods(crow::HTTPMethod::GET)
    ([this](const crow::request& req, int showId) {
        return this->handleGetShowSeats(req, showId);
    });
}

crow::response ShowController::handleGetAllShows(const crow::request& req) {
    auto shows = showService_->getAllShows();
    crow::json::wvalue res = crow::json::wvalue::list();
    for (size_t i = 0; i < shows.size(); ++i) {
        res[i]["id"] = shows[i].id.value_or(0);
        res[i]["movie_id"] = shows[i].movie_id;
        res[i]["screen_id"] = shows[i].screen_id;
        res[i]["start_time"] = shows[i].start_time;
        res[i]["end_time"] = shows[i].end_time;
    }
    return crow::response(200, res);
}

crow::response ShowController::handleGetShowSeats(const crow::request& req, int showId) {
    if (showId <= 0) {
        crow::json::wvalue res;
        res["error"] = "Show not found";
        return crow::response(404, res);
    }

    auto showOpt = showService_->getShow(showId);
    if (!showOpt.has_value()) {
        crow::json::wvalue res;
        res["error"] = "Show not found";
        return crow::response(404, res);
    }

    auto seats = showService_->getShowSeats(showId);
    crow::json::wvalue res = crow::json::wvalue::list();
    for (size_t i = 0; i < seats.size(); ++i) {
        res[i]["id"] = seats[i].id.value_or(0);
        res[i]["show_id"] = seats[i].show_id;
        res[i]["seat_id"] = seats[i].seat_id;
        res[i]["status"] = seats[i].status;
        if (seats[i].lock_expiry_time.has_value()) {
            res[i]["lock_expiry_time"] = *seats[i].lock_expiry_time;
        }
        if (seats[i].booking_id.has_value()) {
            res[i]["booking_id"] = *seats[i].booking_id;
        }
    }
    return crow::response(200, res);
}
