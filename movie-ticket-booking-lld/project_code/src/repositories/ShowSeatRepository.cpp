#include "ShowSeatRepository.h"
#include <pqxx/pqxx>
#include <algorithm>
#include <unordered_set>
#include <string>

namespace {
std::string toPostgresIntArray(const std::vector<int>& ids) {
    std::string result = "{";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) result += ",";
        result += std::to_string(ids[i]);
    }
    result += "}";
    return result;
}
} // anonymous namespace

ShowSeatRepository::ShowSeatRepository(std::shared_ptr<Database> db) : db_(db) {}

ShowSeat ShowSeatRepository::createShowSeat(const ShowSeat& showSeat) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    std::string status = showSeat.status.empty() ? "AVAILABLE" : showSeat.status;
    pqxx::result r = w.exec_params(
        "INSERT INTO show_seats (show_id, seat_id, status) "
        "VALUES ($1, $2, $3::seat_status) "
        "RETURNING id",
        showSeat.show_id, showSeat.seat_id, status
    );
    w.commit();

    ShowSeat created = showSeat;
    created.id = r[0][0].as<int>();
    created.status = status;
    return created;
}

std::vector<ShowSeat> ShowSeatRepository::createShowSeats(const std::vector<ShowSeat>& showSeats) {
    if (showSeats.empty()) {
        return {};
    }

    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    std::vector<ShowSeat> createdList;
    createdList.reserve(showSeats.size());

    for (const auto& ss : showSeats) {
        std::string status = ss.status.empty() ? "AVAILABLE" : ss.status;
        pqxx::result r = w.exec_params(
            "INSERT INTO show_seats (show_id, seat_id, status) "
            "VALUES ($1, $2, $3::seat_status) "
            "RETURNING id",
            ss.show_id, ss.seat_id, status
        );
        ShowSeat created = ss;
        created.id = r[0][0].as<int>();
        created.status = status;
        createdList.push_back(created);
    }
    w.commit();
    return createdList;
}

std::optional<ShowSeat> ShowSeatRepository::getShowSeat(int showId, int seatId) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "SELECT id, show_id, seat_id, "
        "CASE WHEN status = 'LOCKED'::seat_status AND lock_expiry_time IS NOT NULL AND lock_expiry_time < NOW() THEN 'AVAILABLE' ELSE status::text END AS status, "
        "lock_expiry_time::text, booking_id "
        "FROM show_seats WHERE show_id = $1 AND seat_id = $2",
        showId, seatId
    );

    if (r.empty()) {
        return std::nullopt;
    }

    ShowSeat showSeat;
    showSeat.id = r[0]["id"].as<int>();
    showSeat.show_id = r[0]["show_id"].as<int>();
    showSeat.seat_id = r[0]["seat_id"].as<int>();
    showSeat.status = r[0]["status"].as<std::string>();
    if (!r[0]["lock_expiry_time"].is_null()) {
        showSeat.lock_expiry_time = r[0]["lock_expiry_time"].as<std::string>();
    }
    if (!r[0]["booking_id"].is_null()) {
        showSeat.booking_id = r[0]["booking_id"].as<int>();
    }

    return showSeat;
}

std::vector<ShowSeat> ShowSeatRepository::getShowSeatsByShow(int showId) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "SELECT id, show_id, seat_id, "
        "CASE WHEN status = 'LOCKED'::seat_status AND lock_expiry_time IS NOT NULL AND lock_expiry_time < NOW() THEN 'AVAILABLE' ELSE status::text END AS status, "
        "lock_expiry_time::text, booking_id "
        "FROM show_seats WHERE show_id = $1 ORDER BY seat_id ASC",
        showId
    );

    std::vector<ShowSeat> showSeats;
    showSeats.reserve(r.size());

    for (const auto& row : r) {
        ShowSeat ss;
        ss.id = row["id"].as<int>();
        ss.show_id = row["show_id"].as<int>();
        ss.seat_id = row["seat_id"].as<int>();
        ss.status = row["status"].as<std::string>();
        if (!row["lock_expiry_time"].is_null()) {
            ss.lock_expiry_time = row["lock_expiry_time"].as<std::string>();
        }
        if (!row["booking_id"].is_null()) {
            ss.booking_id = row["booking_id"].as<int>();
        }
        showSeats.push_back(ss);
    }

    return showSeats;
}

bool ShowSeatRepository::lockSeats(int showId, const std::vector<int>& seatIds, int bookingId, int lockDurationSeconds) {
    if (seatIds.empty() || lockDurationSeconds <= 0) {
        return false;
    }

    // Sort seat IDs in ascending order to prevent deadlocks across concurrent transactions
    std::vector<int> sortedSeatIds = seatIds;
    std::sort(sortedSeatIds.begin(), sortedSeatIds.end());
    sortedSeatIds.erase(std::unique(sortedSeatIds.begin(), sortedSeatIds.end()), sortedSeatIds.end());

    // Reject if duplicate seat IDs were requested
    if (sortedSeatIds.size() != seatIds.size()) {
        return false;
    }

    auto conn = db_->getConnection();
    try {
        pqxx::work w(*conn);
        std::string arrayParam = toPostgresIntArray(sortedSeatIds);

        // Pessimistic row-level locking on sorted seat rows
        pqxx::result r = w.exec_params(
            "SELECT seat_id, status::text, lock_expiry_time::text, "
            "(lock_expiry_time IS NOT NULL AND lock_expiry_time < NOW()) AS is_expired "
            "FROM show_seats "
            "WHERE show_id = $1 AND seat_id = ANY($2::int[]) "
            "ORDER BY seat_id ASC FOR UPDATE",
            showId, arrayParam
        );

        // Ensure all requested seats exist for this show
        if (r.size() != sortedSeatIds.size()) {
            return false;
        }

        // Validate each seat's availability
        for (const auto& row : r) {
            std::string status = row["status"].as<std::string>();
            if (status == "AVAILABLE") {
                continue;
            } else if (status == "LOCKED") {
                bool isExpired = row["is_expired"].as<bool>();
                if (isExpired) {
                    continue; // Lock has expired, can be re-locked
                } else {
                    return false; // Active lock held
                }
            } else {
                return false; // Already BOOKED or invalid state
            }
        }

        // Atomically update seat status to LOCKED with lock expiration timestamp
        w.exec_params(
            "UPDATE show_seats "
            "SET status = 'LOCKED'::seat_status, booking_id = $1, "
            "lock_expiry_time = NOW() + ($2 || ' seconds')::INTERVAL "
            "WHERE show_id = $3 AND seat_id = ANY($4::int[])",
            bookingId, lockDurationSeconds, showId, arrayParam
        );

        w.commit();
        return true;
    } catch (const std::exception& e) {
        // Automatic rollback upon exception / lock conflict
        return false;
    }
}

bool ShowSeatRepository::confirmSeats(int bookingId) {
    auto conn = db_->getConnection();
    try {
        pqxx::work w(*conn);
        pqxx::result r = w.exec_params(
            "UPDATE show_seats SET status = 'BOOKED'::seat_status, lock_expiry_time = NULL "
            "WHERE booking_id = $1",
            bookingId
        );
        w.commit();
        return r.affected_rows() > 0;
    } catch (const std::exception& e) {
        return false;
    }
}

bool ShowSeatRepository::releaseSeatsForBooking(int bookingId) {
    auto conn = db_->getConnection();
    try {
        pqxx::work w(*conn);
        pqxx::result r = w.exec_params(
            "UPDATE show_seats SET status = 'AVAILABLE'::seat_status, booking_id = NULL, lock_expiry_time = NULL "
            "WHERE booking_id = $1",
            bookingId
        );
        w.commit();
        return r.affected_rows() > 0;
    } catch (const std::exception& e) {
        return false;
    }
}

std::vector<int> ShowSeatRepository::releaseExpiredSeats() {
    auto conn = db_->getConnection();
    try {
        pqxx::work w(*conn);
        pqxx::result r = w.exec(
            "UPDATE show_seats "
            "SET status = 'AVAILABLE'::seat_status, booking_id = NULL, lock_expiry_time = NULL "
            "WHERE status = 'LOCKED'::seat_status AND lock_expiry_time IS NOT NULL AND lock_expiry_time < NOW() "
            "RETURNING booking_id"
        );
        w.commit();

        std::vector<int> expiredBookingIds;
        std::unordered_set<int> uniqueBookings;
        for (const auto& row : r) {
            if (!row["booking_id"].is_null()) {
                int bId = row["booking_id"].as<int>();
                if (uniqueBookings.insert(bId).second) {
                    expiredBookingIds.push_back(bId);
                }
            }
        }
        return expiredBookingIds;
    } catch (const std::exception& e) {
        return {};
    }
}
