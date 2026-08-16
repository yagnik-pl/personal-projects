#include <gtest/gtest.h>
#include "services/ShowService.h"
#include "repositories/IShowRepository.h"
#include "repositories/IShowSeatRepository.h"
#include "models/Enums.h"
#include <map>

class FakeShowRepositoryForShowService : public IShowRepository {
public:
    Show createShow(const Show& show) override {
        Show s = show;
        s.id = nextId_++;
        shows_[*s.id] = s;
        return s;
    }

    std::optional<Show> getShow(int id) override {
        auto it = shows_.find(id);
        if (it != shows_.end()) return it->second;
        return std::nullopt;
    }

    std::vector<Show> getAllShows() override {
        std::vector<Show> res;
        for (const auto& [id, s] : shows_) res.push_back(s);
        return res;
    }

    std::vector<Show> getShowsByMovie(int movieId) override {
        std::vector<Show> res;
        for (const auto& [id, s] : shows_) {
            if (s.movie_id == movieId) res.push_back(s);
        }
        return res;
    }

    std::vector<Show> getShowsByScreen(int screenId) override {
        std::vector<Show> res;
        for (const auto& [id, s] : shows_) {
            if (s.screen_id == screenId) res.push_back(s);
        }
        return res;
    }

    void addShow(const Show& s) {
        if (s.id.has_value()) {
            shows_[*s.id] = s;
        }
    }

private:
    int nextId_{1};
    std::map<int, Show> shows_;
};

class FakeShowSeatRepositoryForShowService : public IShowSeatRepository {
public:
    ShowSeat createShowSeat(const ShowSeat& showSeat) override { return showSeat; }
    std::vector<ShowSeat> createShowSeats(const std::vector<ShowSeat>& showSeats) override { return showSeats; }
    std::optional<ShowSeat> getShowSeat(int showId, int seatId) override { return std::nullopt; }

    std::vector<ShowSeat> getShowSeatsByShow(int showId) override {
        auto it = showSeatsMap_.find(showId);
        if (it != showSeatsMap_.end()) return it->second;
        return {};
    }

    bool lockSeats(int showId, const std::vector<int>& seatIds, int bookingId, int lockDurationSeconds) override { return true; }
    bool confirmSeats(int bookingId) override { return true; }
    bool releaseSeatsForBooking(int bookingId) override { return true; }
    std::vector<int> releaseExpiredSeats() override { return {}; }

    void setShowSeats(int showId, const std::vector<ShowSeat>& seats) {
        showSeatsMap_[showId] = seats;
    }

private:
    std::map<int, std::vector<ShowSeat>> showSeatsMap_;
};

class ShowServiceTest : public ::testing::Test {
protected:
    void SetUp() override {
        showRepo_ = std::make_shared<FakeShowRepositoryForShowService>();
        showSeatRepo_ = std::make_shared<FakeShowSeatRepositoryForShowService>();
        service_ = std::make_unique<ShowService>(showRepo_, showSeatRepo_);

        Show s1;
        s1.id = 1;
        s1.movie_id = 10;
        s1.screen_id = 100;
        s1.start_time = "2026-08-15 10:00:00";
        s1.end_time = "2026-08-15 12:30:00";
        showRepo_->addShow(s1);

        Show s2;
        s2.id = 2;
        s2.movie_id = 10;
        s2.screen_id = 101;
        s2.start_time = "2026-08-15 14:00:00";
        s2.end_time = "2026-08-15 16:30:00";
        showRepo_->addShow(s2);

        ShowSeat ss1;
        ss1.id = 1;
        ss1.show_id = 1;
        ss1.seat_id = 1;
        ss1.status = SeatStatus::AVAILABLE;

        ShowSeat ss2;
        ss2.id = 2;
        ss2.show_id = 1;
        ss2.seat_id = 2;
        ss2.status = SeatStatus::BOOKED;

        showSeatRepo_->setShowSeats(1, {ss1, ss2});
    }

    std::shared_ptr<FakeShowRepositoryForShowService> showRepo_;
    std::shared_ptr<FakeShowSeatRepositoryForShowService> showSeatRepo_;
    std::unique_ptr<ShowService> service_;
};

TEST_F(ShowServiceTest, GetAllShows) {
    auto shows = service_->getAllShows();
    EXPECT_EQ(shows.size(), 2);
}

TEST_F(ShowServiceTest, GetShowById) {
    auto show = service_->getShow(1);
    ASSERT_TRUE(show.has_value());
    EXPECT_EQ(show->id, 1);
    EXPECT_EQ(show->movie_id, 10);

    auto notFound = service_->getShow(999);
    EXPECT_FALSE(notFound.has_value());

    auto invalid = service_->getShow(-1);
    EXPECT_FALSE(invalid.has_value());
}

TEST_F(ShowServiceTest, GetShowSeats) {
    auto seats = service_->getShowSeats(1);
    EXPECT_EQ(seats.size(), 2);
    EXPECT_EQ(seats[0].status, SeatStatus::AVAILABLE);
    EXPECT_EQ(seats[1].status, SeatStatus::BOOKED);

    // Non-existent show
    auto emptySeats = service_->getShowSeats(999);
    EXPECT_TRUE(emptySeats.empty());
}

TEST_F(ShowServiceTest, GetShowsByMovie) {
    auto shows = service_->getShowsByMovie(10);
    EXPECT_EQ(shows.size(), 2);

    auto noShows = service_->getShowsByMovie(999);
    EXPECT_TRUE(noShows.empty());
}
