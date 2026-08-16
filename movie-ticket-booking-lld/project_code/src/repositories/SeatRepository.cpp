#include "SeatRepository.h"
#include <pqxx/pqxx>

SeatRepository::SeatRepository(std::shared_ptr<Database> db) : db_(db) {}

Seat SeatRepository::createSeat(const Seat& seat) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "INSERT INTO seats (screen_id, row_no, col_no) VALUES ($1, $2, $3) RETURNING id",
        seat.screen_id, seat.row_no, seat.col_no
    );
    w.commit();

    Seat created = seat;
    created.id = r[0][0].as<int>();
    return created;
}

std::optional<Seat> SeatRepository::getSeat(int id) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "SELECT id, screen_id, row_no, col_no FROM seats WHERE id = $1",
        id
    );

    if (r.empty()) {
        return std::nullopt;
    }

    Seat seat;
    seat.id = r[0]["id"].as<int>();
    seat.screen_id = r[0]["screen_id"].as<int>();
    seat.row_no = r[0]["row_no"].as<int>();
    seat.col_no = r[0]["col_no"].as<int>();

    return seat;
}

std::vector<Seat> SeatRepository::getSeatsByScreen(int screenId) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "SELECT id, screen_id, row_no, col_no FROM seats WHERE screen_id = $1 ORDER BY row_no ASC, col_no ASC, id ASC",
        screenId
    );

    std::vector<Seat> seats;
    seats.reserve(r.size());

    for (const auto& row : r) {
        Seat seat;
        seat.id = row["id"].as<int>();
        seat.screen_id = row["screen_id"].as<int>();
        seat.row_no = row["row_no"].as<int>();
        seat.col_no = row["col_no"].as<int>();
        seats.push_back(seat);
    }

    return seats;
}
