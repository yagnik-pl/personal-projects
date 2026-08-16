#include "MovieRepository.h"
#include <pqxx/pqxx>

MovieRepository::MovieRepository(std::shared_ptr<Database> db) : db_(db) {}

Movie MovieRepository::createMovie(const Movie& movie) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "INSERT INTO movies (title, duration, language) VALUES ($1, $2, $3) RETURNING id",
        movie.title, movie.duration, movie.language
    );
    w.commit();

    Movie created = movie;
    created.id = r[0][0].as<int>();
    return created;
}

std::optional<Movie> MovieRepository::getMovie(int id) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "SELECT id, title, duration, language FROM movies WHERE id = $1",
        id
    );

    if (r.empty()) {
        return std::nullopt;
    }

    Movie movie;
    movie.id = r[0]["id"].as<int>();
    movie.title = r[0]["title"].as<std::string>();
    movie.duration = r[0]["duration"].as<int>();
    movie.language = r[0]["language"].as<std::string>();

    return movie;
}

std::vector<Movie> MovieRepository::getAllMovies() {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "SELECT id, title, duration, language FROM movies ORDER BY id ASC"
    );

    std::vector<Movie> movies;
    movies.reserve(r.size());

    for (const auto& row : r) {
        Movie movie;
        movie.id = row["id"].as<int>();
        movie.title = row["title"].as<std::string>();
        movie.duration = row["duration"].as<int>();
        movie.language = row["language"].as<std::string>();
        movies.push_back(movie);
    }

    return movies;
}
