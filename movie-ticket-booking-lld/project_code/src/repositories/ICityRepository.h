#pragma once
#include <optional>
#include "../models/City.h"

class ICityRepository {
public:
    virtual ~ICityRepository() = default;
    virtual City createCity(const City& city) = 0;
    virtual std::optional<City> getCity(int id) = 0;
};
