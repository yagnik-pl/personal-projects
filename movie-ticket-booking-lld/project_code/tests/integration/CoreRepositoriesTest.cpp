#include <gtest/gtest.h>
#include "../../src/core/Database.h"
#include "../../src/repositories/UserRepository.h"
#include "../../src/repositories/CityRepository.h"
#include "../../src/repositories/TheatreRepository.h"
#include "../../src/repositories/ScreenRepository.h"
#include <memory>

class CoreRepositoriesTest : public ::testing::Test {
protected:
    void SetUp() override {
        const std::string conn_str = "host=localhost port=5432 dbname=postgres user=postgres password=postgres";
        try {
            db = std::make_shared<Database>(conn_str, 5);
            // Clean up tables before testing to ensure a clean state
            auto conn = db->getConnection();
            pqxx::work w(*conn);
            w.exec("TRUNCATE TABLE users, cities, theatres, screens CASCADE");
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
                w.exec("TRUNCATE TABLE users, cities, theatres, screens CASCADE");
                w.commit();
            } catch (...) {}
        }
    }

    std::shared_ptr<Database> db;
};

TEST_F(CoreRepositoriesTest, UserCreateAndGet) {
    UserRepository userRepo(db);
    User user;
    user.name = "Test User";
    user.email = "test@example.com";
    user.phone = "1234567890";

    User createdUser = userRepo.createUser(user);
    EXPECT_TRUE(createdUser.id.has_value());
    EXPECT_EQ(createdUser.name, "Test User");

    auto fetchedUserOpt = userRepo.getUser(createdUser.id.value());
    ASSERT_TRUE(fetchedUserOpt.has_value());
    EXPECT_EQ(fetchedUserOpt->name, "Test User");
    EXPECT_EQ(fetchedUserOpt->email, "test@example.com");
    EXPECT_EQ(fetchedUserOpt->phone, "1234567890");
}

TEST_F(CoreRepositoriesTest, CityCreateAndGet) {
    CityRepository cityRepo(db);
    City city;
    city.name = "Test City";

    City createdCity = cityRepo.createCity(city);
    EXPECT_TRUE(createdCity.id.has_value());

    auto fetchedCityOpt = cityRepo.getCity(createdCity.id.value());
    ASSERT_TRUE(fetchedCityOpt.has_value());
    EXPECT_EQ(fetchedCityOpt->name, "Test City");
}

TEST_F(CoreRepositoriesTest, TheatreCreateAndGet) {
    CityRepository cityRepo(db);
    City city;
    city.name = "Theatre City";
    City createdCity = cityRepo.createCity(city);

    TheatreRepository theatreRepo(db);
    Theatre theatre;
    theatre.name = "Test Theatre";
    theatre.city_id = createdCity.id.value();

    Theatre createdTheatre = theatreRepo.createTheatre(theatre);
    EXPECT_TRUE(createdTheatre.id.has_value());

    auto fetchedTheatreOpt = theatreRepo.getTheatre(createdTheatre.id.value());
    ASSERT_TRUE(fetchedTheatreOpt.has_value());
    EXPECT_EQ(fetchedTheatreOpt->name, "Test Theatre");
    EXPECT_EQ(fetchedTheatreOpt->city_id, createdCity.id.value());
}

TEST_F(CoreRepositoriesTest, ScreenCreateAndGet) {
    CityRepository cityRepo(db);
    City city;
    city.name = "Screen City";
    City createdCity = cityRepo.createCity(city);

    TheatreRepository theatreRepo(db);
    Theatre theatre;
    theatre.name = "Screen Theatre";
    theatre.city_id = createdCity.id.value();
    Theatre createdTheatre = theatreRepo.createTheatre(theatre);

    ScreenRepository screenRepo(db);
    Screen screen;
    screen.name = "Screen 1";
    screen.theatre_id = createdTheatre.id.value();

    Screen createdScreen = screenRepo.createScreen(screen);
    EXPECT_TRUE(createdScreen.id.has_value());

    auto fetchedScreenOpt = screenRepo.getScreen(createdScreen.id.value());
    ASSERT_TRUE(fetchedScreenOpt.has_value());
    EXPECT_EQ(fetchedScreenOpt->name, "Screen 1");
    EXPECT_EQ(fetchedScreenOpt->theatre_id, createdTheatre.id.value());
}

TEST_F(CoreRepositoriesTest, UserUniqueEmailConstraint) {
    UserRepository userRepo(db);
    User user;
    user.name = "Test User 1";
    user.email = "duplicate@example.com";
    user.phone = "1234567890";
    
    // First creation should succeed
    EXPECT_NO_THROW(userRepo.createUser(user));
    
    // Second creation with the same email should throw a pqxx::sql_error
    User user2;
    user2.name = "Test User 2";
    user2.email = "duplicate@example.com";
    user2.phone = "0987654321";
    
    EXPECT_THROW(userRepo.createUser(user2), pqxx::sql_error);
}

TEST_F(CoreRepositoriesTest, TheatreForeignKeyConstraint) {
    TheatreRepository theatreRepo(db);
    Theatre theatre;
    theatre.name = "Invalid City Theatre";
    theatre.city_id = 9999; // Non-existent city ID
    
    EXPECT_THROW(theatreRepo.createTheatre(theatre), pqxx::sql_error);
}

TEST_F(CoreRepositoriesTest, ScreenForeignKeyConstraint) {
    ScreenRepository screenRepo(db);
    Screen screen;
    screen.name = "Invalid Theatre Screen";
    screen.theatre_id = 9999; // Non-existent theatre ID
    
    EXPECT_THROW(screenRepo.createScreen(screen), pqxx::sql_error);
}
