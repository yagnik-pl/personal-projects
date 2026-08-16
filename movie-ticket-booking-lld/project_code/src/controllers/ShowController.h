#pragma once
#include <crow.h>
#include <memory>
#include "../services/IShowService.h"

class ShowController {
public:
    explicit ShowController(std::shared_ptr<IShowService> showService);

    void registerRoutes(crow::SimpleApp& app);

    crow::response handleGetAllShows(const crow::request& req);
    crow::response handleGetShowSeats(const crow::request& req, int showId);

private:
    std::shared_ptr<IShowService> showService_;
};
