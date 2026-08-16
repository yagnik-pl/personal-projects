#include <gtest/gtest.h>
#include "../../src/core/Database.h"
#include "../../src/repositories/UserRepository.h"
#include "../../src/repositories/CityRepository.h"
#include "../../src/repositories/TheatreRepository.h"
#include "../../src/repositories/ScreenRepository.h"
#include "../../src/repositories/MovieRepository.h"
#include "../../src/repositories/ShowRepository.h"
#include "../../src/repositories/SeatRepository.h"
#include "../../src/repositories/BookingRepository.h"
#include "../../src/repositories/ShowSeatRepository.h"
#include "../../src/repositories/PaymentRepository.h"
#include "../../src/models/Enums.h"
#include <memory>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

class DomainRepositoriesTest : public ::testing::Test {
protected:
    void SetUp() override {
        const std::string conn_str = "host=localhost port=5432 dbname=postgres user=postgres password=postgres";
        try {
            db = std::make_shared<Database>(conn_str, 15);
            // Clean up tables before testing to ensure clean state
            auto conn = db->getConnection();
            pqxx::work w(*conn);
            w.exec("TRUNCATE TABLE users, cities, theatres, screens, movies, shows, seats, bookings, show_seats, payments CASCADE");
            w.commit();
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Database connection failed, skipping tests. Error: " << e.what();
        }
    }

    void TearDown() override {
        if (db) {
            try {
                auto conn = db->getConnection();
                pqxx::work w(*conn);
                w.exec("TRUNCATE TABLE users, cities, theatres, screens, movies, shows, seats, bookings, show_seats, payments CASCADE");
                w.commit();
            } catch (...) {}
        }
    }

    std::shared_ptr<Database> db;
};

TEST_F(DomainRepositoriesTest, MovieCreateAndGet) {
    MovieRepository movieRepo(db);

    Movie movie;
    movie.title = "Inception";
    movie.duration = 148;
    movie.language = "English";

    Movie created = movieRepo.createMovie(movie);
    EXPECT_TRUE(created.id.has_value());
    EXPECT_EQ(created.title, "Inception");
    EXPECT_EQ(created.duration, 148);
    EXPECT_EQ(created.language, "English");

    auto fetched = movieRepo.getMovie(created.id.value());
    ASSERT_TRUE(fetched.has_value());
    EXPECT_EQ(fetched->id, created.id);
    EXPECT_EQ(fetched->title, "Inception");
    EXPECT_EQ(fetched->duration, 148);
    EXPECT_EQ(fetched->language, "English");

    auto allMovies = movieRepo.getAllMovies();
    EXPECT_GE(allMovies.size(), 1u);
}

TEST_F(DomainRepositoriesTest, ShowCreateAndGet) {
    CityRepository cityRepo(db);
    City city;
    city.name = "Test City";
    City createdCity = cityRepo.createCity(city);

    TheatreRepository theatreRepo(db);
    Theatre theatre;
    theatre.name = "Test Theatre";
    theatre.city_id = createdCity.id.value();
    Theatre createdTheatre = theatreRepo.createTheatre(theatre);

    ScreenRepository screenRepo(db);
    Screen screen;
    screen.name = "Screen 1";
    screen.theatre_id = createdTheatre.id.value();
    Screen createdScreen = screenRepo.createScreen(screen);

    MovieRepository movieRepo(db);
    Movie movie;
    movie.title = "Interstellar";
    movie.duration = 169;
    movie.language = "English";
    Movie createdMovie = movieRepo.createMovie(movie);

    ShowRepository showRepo(db);
    Show show;
    show.movie_id = createdMovie.id.value();
    show.screen_id = createdScreen.id.value();
    show.start_time = "2026-08-15 18:00:00";
    show.end_time = "2026-08-15 21:00:00";

    Show created = showRepo.createShow(show);
    EXPECT_TRUE(created.id.has_value());
    EXPECT_EQ(created.movie_id, createdMovie.id.value());
    EXPECT_EQ(created.screen_id, createdScreen.id.value());

    auto fetched = showRepo.getShow(created.id.value());
    ASSERT_TRUE(fetched.has_value());
    EXPECT_EQ(fetched->id, created.id);

    auto showsByMovie = showRepo.getShowsByMovie(createdMovie.id.value());
    EXPECT_EQ(showsByMovie.size(), 1u);

    auto showsByScreen = showRepo.getShowsByScreen(createdScreen.id.value());
    EXPECT_EQ(showsByScreen.size(), 1u);

    auto allShows = showRepo.getAllShows();
    EXPECT_GE(allShows.size(), 1u);
}

TEST_F(DomainRepositoriesTest, SeatCreateAndGet) {
    CityRepository cityRepo(db);
    City city;
    city.name = "Seat City";
    City createdCity = cityRepo.createCity(city);

    TheatreRepository theatreRepo(db);
    Theatre theatre;
    theatre.name = "Seat Theatre";
    theatre.city_id = createdCity.id.value();
    Theatre createdTheatre = theatreRepo.createTheatre(theatre);

    ScreenRepository screenRepo(db);
    Screen screen;
    screen.name = "Screen 1";
    screen.theatre_id = createdTheatre.id.value();
    Screen createdScreen = screenRepo.createScreen(screen);

    SeatRepository seatRepo(db);
    Seat seat;
    seat.screen_id = createdScreen.id.value();
    seat.row_no = 1;
    seat.col_no = 1;

    Seat created = seatRepo.createSeat(seat);
    EXPECT_TRUE(created.id.has_value());
    EXPECT_EQ(created.row_no, 1);
    EXPECT_EQ(created.col_no, 1);

    auto fetched = seatRepo.getSeat(created.id.value());
    ASSERT_TRUE(fetched.has_value());
    EXPECT_EQ(fetched->screen_id, createdScreen.id.value());
    EXPECT_EQ(fetched->row_no, 1);
    EXPECT_EQ(fetched->col_no, 1);

    Seat seat2;
    seat2.screen_id = createdScreen.id.value();
    seat2.row_no = 1;
    seat2.col_no = 2;
    Seat created2 = seatRepo.createSeat(seat2);
    EXPECT_TRUE(created2.id.has_value());

    auto screenSeats = seatRepo.getSeatsByScreen(createdScreen.id.value());
    EXPECT_EQ(screenSeats.size(), 2u);
}

TEST_F(DomainRepositoriesTest, BookingCreateAndStatusUpdate) {
    UserRepository userRepo(db);
    User user;
    user.name = "John Doe";
    user.email = "john@example.com";
    user.phone = "1234567890";
    User createdUser = userRepo.createUser(user);

    CityRepository cityRepo(db);
    City city;
    city.name = "Booking City";
    City createdCity = cityRepo.createCity(city);

    TheatreRepository theatreRepo(db);
    Theatre theatre;
    theatre.name = "Booking Theatre";
    theatre.city_id = createdCity.id.value();
    Theatre createdTheatre = theatreRepo.createTheatre(theatre);

    ScreenRepository screenRepo(db);
    Screen screen;
    screen.name = "Screen 1";
    screen.theatre_id = createdTheatre.id.value();
    Screen createdScreen = screenRepo.createScreen(screen);

    MovieRepository movieRepo(db);
    Movie movie;
    movie.title = "The Matrix";
    movie.duration = 136;
    movie.language = "English";
    Movie createdMovie = movieRepo.createMovie(movie);

    ShowRepository showRepo(db);
    Show show;
    show.movie_id = createdMovie.id.value();
    show.screen_id = createdScreen.id.value();
    show.start_time = "2026-08-15 20:00:00";
    show.end_time = "2026-08-15 22:30:00";
    Show createdShow = showRepo.createShow(show);

    BookingRepository bookingRepo(db);
    Booking booking;
    booking.user_id = createdUser.id.value();
    booking.show_id = createdShow.id.value();
    booking.status = BookingStatus::CREATED;
    booking.amount = 50.0;

    Booking created = bookingRepo.createBooking(booking);
    EXPECT_TRUE(created.id.has_value());
    EXPECT_EQ(created.status, BookingStatus::CREATED);
    EXPECT_DOUBLE_EQ(created.amount, 50.0);

    auto fetched = bookingRepo.getBooking(created.id.value());
    ASSERT_TRUE(fetched.has_value());
    EXPECT_EQ(fetched->status, BookingStatus::CREATED);

    bool updated = bookingRepo.updateBookingStatus(created.id.value(), BookingStatus::SEATS_LOCKED);
    EXPECT_TRUE(updated);

    auto fetchedAfterUpdate = bookingRepo.getBooking(created.id.value());
    ASSERT_TRUE(fetchedAfterUpdate.has_value());
    EXPECT_EQ(fetchedAfterUpdate->status, BookingStatus::SEATS_LOCKED);

    bool amountUpdated = bookingRepo.updateBookingAmount(created.id.value(), 75.0);
    EXPECT_TRUE(amountUpdated);

    auto fetchedAfterAmount = bookingRepo.getBooking(created.id.value());
    ASSERT_TRUE(fetchedAfterAmount.has_value());
    EXPECT_DOUBLE_EQ(fetchedAfterAmount->amount, 75.0);

    auto userBookings = bookingRepo.getBookingsByUser(createdUser.id.value());
    EXPECT_EQ(userBookings.size(), 1u);

    auto showBookings = bookingRepo.getBookingsByShow(createdShow.id.value());
    EXPECT_EQ(showBookings.size(), 1u);
}

TEST_F(DomainRepositoriesTest, ShowSeatLockingAndConcurrency) {
    UserRepository userRepo(db);
    User u1; u1.name = "User 1"; u1.email = "u1@example.com"; u1.phone = "111";
    User user1 = userRepo.createUser(u1);

    User u2; u2.name = "User 2"; u2.email = "u2@example.com"; u2.phone = "222";
    User user2 = userRepo.createUser(u2);

    CityRepository cityRepo(db);
    City city; city.name = "ShowSeat City";
    City createdCity = cityRepo.createCity(city);

    TheatreRepository theatreRepo(db);
    Theatre theatre; theatre.name = "ShowSeat Theatre"; theatre.city_id = createdCity.id.value();
    Theatre createdTheatre = theatreRepo.createTheatre(theatre);

    ScreenRepository screenRepo(db);
    Screen screen; screen.name = "Screen 1"; screen.theatre_id = createdTheatre.id.value();
    Screen createdScreen = screenRepo.createScreen(screen);

    SeatRepository seatRepo(db);
    Seat s1; s1.screen_id = createdScreen.id.value(); s1.row_no = 1; s1.col_no = 1;
    Seat seat1 = seatRepo.createSeat(s1);

    Seat s2; s2.screen_id = createdScreen.id.value(); s2.row_no = 1; s2.col_no = 2;
    Seat seat2 = seatRepo.createSeat(s2);

    MovieRepository movieRepo(db);
    Movie movie; movie.title = "Avatar"; movie.duration = 162; movie.language = "English";
    Movie createdMovie = movieRepo.createMovie(movie);

    ShowRepository showRepo(db);
    Show show;
    show.movie_id = createdMovie.id.value();
    show.screen_id = createdScreen.id.value();
    show.start_time = "2026-08-15 14:00:00";
    show.end_time = "2026-08-15 17:00:00";
    Show createdShow = showRepo.createShow(show);

    ShowSeatRepository showSeatRepo(db);
    ShowSeat ssA; ssA.show_id = createdShow.id.value(); ssA.seat_id = seat1.id.value(); ssA.status = SeatStatus::AVAILABLE;
    showSeatRepo.createShowSeat(ssA);

    ShowSeat ssB; ssB.show_id = createdShow.id.value(); ssB.seat_id = seat2.id.value(); ssB.status = SeatStatus::AVAILABLE;
    showSeatRepo.createShowSeat(ssB);

    BookingRepository bookingRepo(db);
    Booking b1; b1.user_id = user1.id.value(); b1.show_id = createdShow.id.value(); b1.status = BookingStatus::CREATED; b1.amount = 25.0;
    Booking booking1 = bookingRepo.createBooking(b1);

    Booking b2; b2.user_id = user2.id.value(); b2.show_id = createdShow.id.value(); b2.status = BookingStatus::CREATED; b2.amount = 25.0;
    Booking booking2 = bookingRepo.createBooking(b2);

    // 1. Lock seat 1 for booking 1 with 300 second lock
    bool lock1Success = showSeatRepo.lockSeats(createdShow.id.value(), {seat1.id.value()}, booking1.id.value(), 300);
    EXPECT_TRUE(lock1Success);

    // 2. Lock seat 1 again for booking 2 -> must FAIL (conflict)
    bool lock2Success = showSeatRepo.lockSeats(createdShow.id.value(), {seat1.id.value()}, booking2.id.value(), 300);
    EXPECT_FALSE(lock2Success);

    // 3. Lock non-existent seat -> must FAIL
    bool invalidLock = showSeatRepo.lockSeats(createdShow.id.value(), {99999}, booking2.id.value(), 300);
    EXPECT_FALSE(invalidLock);

    // 4. Confirm booking 1 seats
    bool confirmSuccess = showSeatRepo.confirmSeats(booking1.id.value());
    EXPECT_TRUE(confirmSuccess);

    auto ss1 = showSeatRepo.getShowSeat(createdShow.id.value(), seat1.id.value());
    ASSERT_TRUE(ss1.has_value());
    EXPECT_EQ(ss1->status, SeatStatus::BOOKED);

    // 5. Try locking confirmed/booked seat -> must FAIL
    bool lockBooked = showSeatRepo.lockSeats(createdShow.id.value(), {seat1.id.value()}, booking2.id.value(), 300);
    EXPECT_FALSE(lockBooked);

    // 6. Release seats for booking 1
    bool releaseSuccess = showSeatRepo.releaseSeatsForBooking(booking1.id.value());
    EXPECT_TRUE(releaseSuccess);

    auto ss1AfterRelease = showSeatRepo.getShowSeat(createdShow.id.value(), seat1.id.value());
    ASSERT_TRUE(ss1AfterRelease.has_value());
    EXPECT_EQ(ss1AfterRelease->status, SeatStatus::AVAILABLE);
}

TEST_F(DomainRepositoriesTest, ShowSeatExpirationAndReLock) {
    UserRepository userRepo(db);
    User u1; u1.name = "User Expiry 1"; u1.email = "ue1@example.com"; u1.phone = "111";
    User user1 = userRepo.createUser(u1);

    User u2; u2.name = "User Expiry 2"; u2.email = "ue2@example.com"; u2.phone = "222";
    User user2 = userRepo.createUser(u2);

    CityRepository cityRepo(db);
    City city; city.name = "Expiry City";
    City createdCity = cityRepo.createCity(city);

    TheatreRepository theatreRepo(db);
    Theatre theatre; theatre.name = "Expiry Theatre"; theatre.city_id = createdCity.id.value();
    Theatre createdTheatre = theatreRepo.createTheatre(theatre);

    ScreenRepository screenRepo(db);
    Screen screen; screen.name = "Screen 1"; screen.theatre_id = createdTheatre.id.value();
    Screen createdScreen = screenRepo.createScreen(screen);

    SeatRepository seatRepo(db);
    Seat s; s.screen_id = createdScreen.id.value(); s.row_no = 1; s.col_no = 1;
    Seat seat = seatRepo.createSeat(s);

    MovieRepository movieRepo(db);
    Movie movie; movie.title = "Tenet"; movie.duration = 150; movie.language = "English";
    Movie createdMovie = movieRepo.createMovie(movie);

    ShowRepository showRepo(db);
    Show show;
    show.movie_id = createdMovie.id.value();
    show.screen_id = createdScreen.id.value();
    show.start_time = "2026-08-15 14:00:00";
    show.end_time = "2026-08-15 17:00:00";
    Show createdShow = showRepo.createShow(show);

    ShowSeatRepository showSeatRepo(db);
    ShowSeat ss; ss.show_id = createdShow.id.value(); ss.seat_id = seat.id.value(); ss.status = SeatStatus::AVAILABLE;
    showSeatRepo.createShowSeat(ss);

    BookingRepository bookingRepo(db);
    Booking b1; b1.user_id = user1.id.value(); b1.show_id = createdShow.id.value(); b1.status = BookingStatus::CREATED; b1.amount = 25.0;
    Booking booking1 = bookingRepo.createBooking(b1);

    Booking b2; b2.user_id = user2.id.value(); b2.show_id = createdShow.id.value(); b2.status = BookingStatus::CREATED; b2.amount = 25.0;
    Booking booking2 = bookingRepo.createBooking(b2);

    // Lock with 1 second duration
    bool lock1Success = showSeatRepo.lockSeats(createdShow.id.value(), {seat.id.value()}, booking1.id.value(), 1);
    EXPECT_TRUE(lock1Success);

    // Wait for 1.5 seconds for lock to expire
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));

    // Release expired seats via repository method
    auto expiredBookings = showSeatRepo.releaseExpiredSeats();
    EXPECT_GE(expiredBookings.size(), 1u);
    EXPECT_EQ(expiredBookings[0], booking1.id.value());

    // Now User 2 should be able to lock the seat
    bool lock2Success = showSeatRepo.lockSeats(createdShow.id.value(), {seat.id.value()}, booking2.id.value(), 300);
    EXPECT_TRUE(lock2Success);
}

TEST_F(DomainRepositoriesTest, MultiThreadedConcurrentSeatLocking) {
    UserRepository userRepo(db);
    CityRepository cityRepo(db);
    City city; city.name = "Concurrent City";
    City createdCity = cityRepo.createCity(city);

    TheatreRepository theatreRepo(db);
    Theatre theatre; theatre.name = "Concurrent Theatre"; theatre.city_id = createdCity.id.value();
    Theatre createdTheatre = theatreRepo.createTheatre(theatre);

    ScreenRepository screenRepo(db);
    Screen screen; screen.name = "Screen 1"; screen.theatre_id = createdTheatre.id.value();
    Screen createdScreen = screenRepo.createScreen(screen);

    SeatRepository seatRepo(db);
    Seat s; s.screen_id = createdScreen.id.value(); s.row_no = 1; s.col_no = 1;
    Seat seat = seatRepo.createSeat(s);

    MovieRepository movieRepo(db);
    Movie movie; movie.title = "Gladiator"; movie.duration = 155; movie.language = "English";
    Movie createdMovie = movieRepo.createMovie(movie);

    ShowRepository showRepo(db);
    Show show;
    show.movie_id = createdMovie.id.value();
    show.screen_id = createdScreen.id.value();
    show.start_time = "2026-08-15 14:00:00";
    show.end_time = "2026-08-15 17:00:00";
    Show createdShow = showRepo.createShow(show);

    ShowSeatRepository showSeatRepo(db);
    ShowSeat ss; ss.show_id = createdShow.id.value(); ss.seat_id = seat.id.value(); ss.status = SeatStatus::AVAILABLE;
    showSeatRepo.createShowSeat(ss);

    BookingRepository bookingRepo(db);

    const int numThreads = 10;
    std::vector<int> userIds;
    std::vector<int> bookingIds;
    for (int i = 0; i < numThreads; ++i) {
        User u;
        u.name = "User " + std::to_string(i);
        u.email = "u" + std::to_string(i) + "@conc.com";
        u.phone = "123";
        User createdU = userRepo.createUser(u);
        userIds.push_back(createdU.id.value());

        Booking b;
        b.user_id = createdU.id.value();
        b.show_id = createdShow.id.value();
        b.status = BookingStatus::CREATED;
        b.amount = 25.0;
        Booking createdB = bookingRepo.createBooking(b);
        bookingIds.push_back(createdB.id.value());
    }

    std::atomic<int> successCount{0};
    std::atomic<int> failureCount{0};

    std::vector<std::thread> threads;
    threads.reserve(numThreads);

    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&, i]() {
            ShowSeatRepository repo(db);
            bool success = repo.lockSeats(createdShow.id.value(), {seat.id.value()}, bookingIds[i], 300);
            if (success) {
                successCount++;
            } else {
                failureCount++;
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Exactly 1 thread must successfully acquire the lock
    EXPECT_EQ(successCount.load(), 1);
    EXPECT_EQ(failureCount.load(), numThreads - 1);
}

TEST_F(DomainRepositoriesTest, PaymentCreateAndGet) {
    UserRepository userRepo(db);
    User u; u.name = "Pay User"; u.email = "pay@example.com"; u.phone = "123";
    User user = userRepo.createUser(u);

    CityRepository cityRepo(db);
    City city; city.name = "Pay City";
    City createdCity = cityRepo.createCity(city);

    TheatreRepository theatreRepo(db);
    Theatre theatre; theatre.name = "Pay Theatre"; theatre.city_id = createdCity.id.value();
    Theatre createdTheatre = theatreRepo.createTheatre(theatre);

    ScreenRepository screenRepo(db);
    Screen screen; screen.name = "Screen 1"; screen.theatre_id = createdTheatre.id.value();
    Screen createdScreen = screenRepo.createScreen(screen);

    MovieRepository movieRepo(db);
    Movie movie; movie.title = "Dune"; movie.duration = 155; movie.language = "English";
    Movie createdMovie = movieRepo.createMovie(movie);

    ShowRepository showRepo(db);
    Show show;
    show.movie_id = createdMovie.id.value();
    show.screen_id = createdScreen.id.value();
    show.start_time = "2026-08-15 14:00:00";
    show.end_time = "2026-08-15 17:00:00";
    Show createdShow = showRepo.createShow(show);

    BookingRepository bookingRepo(db);
    Booking b;
    b.user_id = user.id.value();
    b.show_id = createdShow.id.value();
    b.status = BookingStatus::CREATED;
    b.amount = 50.0;
    Booking booking = bookingRepo.createBooking(b);

    PaymentRepository paymentRepo(db);
    Payment payment;
    payment.booking_id = booking.id.value();
    payment.status = PaymentStatus::PENDING;
    payment.transaction_id = "txn_initial_123";

    Payment created = paymentRepo.createPayment(payment);
    EXPECT_TRUE(created.id.has_value());
    EXPECT_EQ(created.status, PaymentStatus::PENDING);
    EXPECT_EQ(created.transaction_id, "txn_initial_123");

    auto fetched = paymentRepo.getPayment(created.id.value());
    ASSERT_TRUE(fetched.has_value());
    EXPECT_EQ(fetched->status, PaymentStatus::PENDING);

    auto fetchedByBooking = paymentRepo.getPaymentByBookingId(booking.id.value());
    ASSERT_TRUE(fetchedByBooking.has_value());
    EXPECT_EQ(fetchedByBooking->id, created.id);

    bool updateSuccess = paymentRepo.updatePaymentStatus(created.id.value(), PaymentStatus::SUCCESS, "txn_final_999");
    EXPECT_TRUE(updateSuccess);

    auto fetchedAfterUpdate = paymentRepo.getPayment(created.id.value());
    ASSERT_TRUE(fetchedAfterUpdate.has_value());
    EXPECT_EQ(fetchedAfterUpdate->status, PaymentStatus::SUCCESS);
    EXPECT_EQ(fetchedAfterUpdate->transaction_id, "txn_final_999");
}
