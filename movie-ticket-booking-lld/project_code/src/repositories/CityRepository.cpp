#include "CityRepository.h"
#include <pqxx/pqxx>

CityRepository::CityRepository(std::shared_ptr<Database> db) : db_(db) {}

City CityRepository::createCity(const City& city) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "INSERT INTO cities (name) VALUES ($1) RETURNING id",
        city.name
    );
    w.commit();

    City createdCity = city;
    createdCity.id = r[0][0].as<int>();
    return createdCity;
}

std::optional<City> CityRepository::getCity(int id) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "SELECT id, name FROM cities WHERE id = $1",
        id
    );

    if (r.empty()) {
        return std::nullopt;
    }

    City city;
    city.id = r[0]["id"].as<int>();
    city.name = r[0]["name"].as<std::string>();

    return city;
}
