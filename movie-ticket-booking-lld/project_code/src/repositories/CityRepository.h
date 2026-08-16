#pragma once
#include "ICityRepository.h"
#include "../core/Database.h"
#include <memory>

class CityRepository : public ICityRepository {
public:
    explicit CityRepository(std::shared_ptr<Database> db);
    City createCity(const City& city) override;
    std::optional<City> getCity(int id) override;
private:
    std::shared_ptr<Database> db_;
};
