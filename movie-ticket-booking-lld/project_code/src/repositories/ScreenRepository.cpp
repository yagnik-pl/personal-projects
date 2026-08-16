#include "ScreenRepository.h"
#include <pqxx/pqxx>

ScreenRepository::ScreenRepository(std::shared_ptr<Database> db) : db_(db) {}

Screen ScreenRepository::createScreen(const Screen& screen) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "INSERT INTO screens (name, theatre_id) VALUES ($1, $2) RETURNING id",
        screen.name, screen.theatre_id
    );
    w.commit();

    Screen createdScreen = screen;
    createdScreen.id = r[0][0].as<int>();
    return createdScreen;
}

std::optional<Screen> ScreenRepository::getScreen(int id) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "SELECT id, name, theatre_id FROM screens WHERE id = $1",
        id
    );

    if (r.empty()) {
        return std::nullopt;
    }

    Screen screen;
    screen.id = r[0]["id"].as<int>();
    screen.name = r[0]["name"].as<std::string>();
    screen.theatre_id = r[0]["theatre_id"].as<int>();

    return screen;
}
