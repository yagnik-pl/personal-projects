#include "TheatreRepository.h"
#include <pqxx/pqxx>

TheatreRepository::TheatreRepository(std::shared_ptr<Database> db) : db_(db) {}

Theatre TheatreRepository::createTheatre(const Theatre& theatre) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "INSERT INTO theatres (name, city_id) VALUES ($1, $2) RETURNING id",
        theatre.name, theatre.city_id
    );
    w.commit();

    Theatre createdTheatre = theatre;
    createdTheatre.id = r[0][0].as<int>();
    return createdTheatre;
}

std::optional<Theatre> TheatreRepository::getTheatre(int id) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "SELECT id, name, city_id FROM theatres WHERE id = $1",
        id
    );

    if (r.empty()) {
        return std::nullopt;
    }

    Theatre theatre;
    theatre.id = r[0]["id"].as<int>();
    theatre.name = r[0]["name"].as<std::string>();
    theatre.city_id = r[0]["city_id"].as<int>();

    return theatre;
}
