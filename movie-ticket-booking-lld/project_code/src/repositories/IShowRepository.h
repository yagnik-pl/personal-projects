#pragma once
#include <optional>
#include <vector>
#include "../models/Show.h"

class IShowRepository {
public:
    virtual ~IShowRepository() = default;
    virtual Show createShow(const Show& show) = 0;
    virtual std::optional<Show> getShow(int id) = 0;
    virtual std::vector<Show> getAllShows() = 0;
    virtual std::vector<Show> getShowsByMovie(int movieId) = 0;
    virtual std::vector<Show> getShowsByScreen(int screenId) = 0;
};
