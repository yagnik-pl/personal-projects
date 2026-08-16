#pragma once
#include <crow.h>
#include <memory>
#include "../core/Database.h"

class ResetController {
public:
    explicit ResetController(std::shared_ptr<Database> db);

    void registerRoutes(crow::SimpleApp& app);

    crow::response handleReset(const crow::request& req);

private:
    std::shared_ptr<Database> db_;
};
