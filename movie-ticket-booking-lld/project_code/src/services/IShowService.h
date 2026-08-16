#pragma once
#include <vector>
#include <optional>
#include "../models/Show.h"
#include "../models/ShowSeat.h"

class IShowService {
public:
    virtual ~IShowService() = default;

    virtual std::vector<Show> getAllShows() = 0;
    virtual std::optional<Show> getShow(int showId) = 0;
    virtual std::vector<ShowSeat> getShowSeats(int showId) = 0;
    virtual std::vector<Show> getShowsByMovie(int movieId) = 0;
};
