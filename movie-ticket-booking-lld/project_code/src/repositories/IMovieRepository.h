#pragma once
#include <optional>
#include <vector>
#include "../models/Movie.h"

class IMovieRepository {
public:
    virtual ~IMovieRepository() = default;
    virtual Movie createMovie(const Movie& movie) = 0;
    virtual std::optional<Movie> getMovie(int id) = 0;
    virtual std::vector<Movie> getAllMovies() = 0;
};
