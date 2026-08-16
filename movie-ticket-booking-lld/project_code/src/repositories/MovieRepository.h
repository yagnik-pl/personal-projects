#pragma once
#include "IMovieRepository.h"
#include "../core/Database.h"
#include <memory>

class MovieRepository : public IMovieRepository {
public:
    explicit MovieRepository(std::shared_ptr<Database> db);
    Movie createMovie(const Movie& movie) override;
    std::optional<Movie> getMovie(int id) override;
    std::vector<Movie> getAllMovies() override;

private:
    std::shared_ptr<Database> db_;
};
