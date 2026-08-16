#include "ShowRepository.h"
#include <pqxx/pqxx>

ShowRepository::ShowRepository(std::shared_ptr<Database> db) : db_(db) {}

Show ShowRepository::createShow(const Show& show) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "INSERT INTO shows (movie_id, screen_id, start_time, end_time) "
        "VALUES ($1, $2, $3, $4) "
        "RETURNING id, start_time::text, end_time::text",
        show.movie_id, show.screen_id, show.start_time, show.end_time
    );
    w.commit();

    Show created = show;
    created.id = r[0]["id"].as<int>();
    created.start_time = r[0]["start_time"].as<std::string>();
    created.end_time = r[0]["end_time"].as<std::string>();
    return created;
}

std::optional<Show> ShowRepository::getShow(int id) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "SELECT id, movie_id, screen_id, start_time::text, end_time::text FROM shows WHERE id = $1",
        id
    );

    if (r.empty()) {
        return std::nullopt;
    }

    Show show;
    show.id = r[0]["id"].as<int>();
    show.movie_id = r[0]["movie_id"].as<int>();
    show.screen_id = r[0]["screen_id"].as<int>();
    show.start_time = r[0]["start_time"].as<std::string>();
    show.end_time = r[0]["end_time"].as<std::string>();

    return show;
}

std::vector<Show> ShowRepository::getAllShows() {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "SELECT id, movie_id, screen_id, start_time::text, end_time::text FROM shows ORDER BY id ASC"
    );

    std::vector<Show> shows;
    shows.reserve(r.size());

    for (const auto& row : r) {
        Show show;
        show.id = row["id"].as<int>();
        show.movie_id = row["movie_id"].as<int>();
        show.screen_id = row["screen_id"].as<int>();
        show.start_time = row["start_time"].as<std::string>();
        show.end_time = row["end_time"].as<std::string>();
        shows.push_back(show);
    }

    return shows;
}

std::vector<Show> ShowRepository::getShowsByMovie(int movieId) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "SELECT id, movie_id, screen_id, start_time::text, end_time::text FROM shows WHERE movie_id = $1 ORDER BY id ASC",
        movieId
    );

    std::vector<Show> shows;
    shows.reserve(r.size());

    for (const auto& row : r) {
        Show show;
        show.id = row["id"].as<int>();
        show.movie_id = row["movie_id"].as<int>();
        show.screen_id = row["screen_id"].as<int>();
        show.start_time = row["start_time"].as<std::string>();
        show.end_time = row["end_time"].as<std::string>();
        shows.push_back(show);
    }

    return shows;
}

std::vector<Show> ShowRepository::getShowsByScreen(int screenId) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "SELECT id, movie_id, screen_id, start_time::text, end_time::text FROM shows WHERE screen_id = $1 ORDER BY id ASC",
        screenId
    );

    std::vector<Show> shows;
    shows.reserve(r.size());

    for (const auto& row : r) {
        Show show;
        show.id = row["id"].as<int>();
        show.movie_id = row["movie_id"].as<int>();
        show.screen_id = row["screen_id"].as<int>();
        show.start_time = row["start_time"].as<std::string>();
        show.end_time = row["end_time"].as<std::string>();
        shows.push_back(show);
    }

    return shows;
}
