#include "ResetController.h"
#include <string>

ResetController::ResetController(std::shared_ptr<Database> db)
    : db_(db) {}

void ResetController::registerRoutes(crow::SimpleApp& app) {
    CROW_ROUTE(app, "/api/test/reset")
    .methods(crow::HTTPMethod::POST)
    ([this](const crow::request& req) {
        return this->handleReset(req);
    });
}

crow::response ResetController::handleReset(const crow::request& req) {
    if (!db_) {
        crow::json::wvalue res;
        res["status"] = "ERROR";
        res["error"] = "Database connection pool not configured";
        return crow::response(500, res);
    }

    try {
        auto conn = db_->getConnection();
        pqxx::work w(*conn);

        // 1. Truncate all 10 tables with CASCADE and RESTART IDENTITY
        w.exec(
            "TRUNCATE TABLE payments, show_seats, bookings, seats, shows, "
            "movies, screens, theatres, cities, users RESTART IDENTITY CASCADE"
        );

        // 2. Seed 105 users (id = 1..105)
        std::string userSql = "INSERT INTO users (id, name, email, phone) VALUES ";
        for (int i = 1; i <= 105; ++i) {
            if (i > 1) userSql += ", ";
            userSql += "(" + std::to_string(i) + ", 'User " + std::to_string(i) +
                       "', 'user" + std::to_string(i) + "@example.com', '1234567890')";
        }
        w.exec(userSql);
        w.exec("SELECT setval('users_id_seq', 105)");

        // 3. Seed 1 city (id = 1)
        w.exec("INSERT INTO cities (id, name) VALUES (1, 'Metropolis')");
        w.exec("SELECT setval('cities_id_seq', 1)");

        // 4. Seed 1 theatre (id = 1)
        w.exec("INSERT INTO theatres (id, name, city_id) VALUES (1, 'Grand Cinema', 1)");
        w.exec("SELECT setval('theatres_id_seq', 1)");

        // 5. Seed 1 screen (id = 1)
        w.exec("INSERT INTO screens (id, name, theatre_id) VALUES (1, 'Screen 1', 1)");
        w.exec("SELECT setval('screens_id_seq', 1)");

        // 6. Seed 1 movie (id = 1)
        w.exec("INSERT INTO movies (id, title, duration, language) VALUES (1, 'Inception', 148, 'English')");
        w.exec("SELECT setval('movies_id_seq', 1)");

        // 7. Seed 1 show (id = 1)
        w.exec(
            "INSERT INTO shows (id, movie_id, screen_id, start_time, end_time) "
            "VALUES (1, 1, 1, NOW() + INTERVAL '1 hour', NOW() + INTERVAL '3 hours')"
        );
        w.exec("SELECT setval('shows_id_seq', 1)");

        // 8. Seed 20 seats (id = 1..20, screen_id = 1)
        std::string seatSql = "INSERT INTO seats (id, screen_id, row_no, col_no) VALUES ";
        for (int i = 1; i <= 20; ++i) {
            if (i > 1) seatSql += ", ";
            int row = (i <= 10) ? 1 : 2;
            int col = (i <= 10) ? i : (i - 10);
            seatSql += "(" + std::to_string(i) + ", 1, " + std::to_string(row) + ", " + std::to_string(col) + ")";
        }
        w.exec(seatSql);
        w.exec("SELECT setval('seats_id_seq', 20)");

        // 9. Seed 20 show_seats (id = 1..20, show_id = 1, seat_id = 1..20, status = AVAILABLE)
        std::string showSeatSql = "INSERT INTO show_seats (id, show_id, seat_id, status) VALUES ";
        for (int i = 1; i <= 20; ++i) {
            if (i > 1) showSeatSql += ", ";
            showSeatSql += "(" + std::to_string(i) + ", 1, " + std::to_string(i) + ", 'AVAILABLE'::seat_status)";
        }
        w.exec(showSeatSql);
        w.exec("SELECT setval('show_seats_id_seq', 20)");

        // 10. Reset sequence starting points for bookings and payments
        w.exec("SELECT setval('bookings_id_seq', 1, false)");
        w.exec("SELECT setval('payments_id_seq', 1, false)");

        w.commit();

        crow::json::wvalue res;
        res["status"] = "OK";
        res["message"] = "Database reset and seeded successfully";
        return crow::response(200, res);
    } catch (const std::exception& e) {
        crow::json::wvalue res;
        res["status"] = "ERROR";
        res["error"] = e.what();
        return crow::response(500, res);
    }
}
