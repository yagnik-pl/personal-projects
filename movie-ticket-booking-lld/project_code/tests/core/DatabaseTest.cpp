#include <gtest/gtest.h>
#include "core/Database.h"
#include <thread>
#include <vector>
#include <atomic>

// Note: This test requires a running PostgreSQL instance if we don't mock.
// We'll use a valid dummy connection string but expect an exception if the DB is down.
// However, if the environment has a DB, it will succeed. 
// For robust testing without a guaranteed DB, we catch connection errors.

TEST(DatabaseTest, PoolInitialization) {
    const std::string conn_str = "host=localhost port=5432 dbname=postgres user=postgres password=postgres";
    try {
        Database db(conn_str, 2);
        auto conn1 = db.getConnection();
        auto conn2 = db.getConnection();
        
        EXPECT_NE(conn1.get(), nullptr);
        EXPECT_NE(conn2.get(), nullptr);
    } catch (const pqxx::broken_connection& e) {
        // If no DB is available locally, we skip the test rather than fail it
        GTEST_SKIP() << "Skipping test because DB is not available: " << e.what();
    } catch (const std::exception& e) {
        FAIL() << "Unexpected exception: " << e.what();
    }
}

TEST(DatabaseTest, ThreadSafety) {
    const std::string conn_str = "host=localhost port=5432 dbname=postgres user=postgres password=postgres";
    try {
        Database db(conn_str, 5);
        std::atomic<int> successful_gets{0};
        
        auto worker = [&]() {
            for (int i = 0; i < 10; ++i) {
                auto conn = db.getConnection();
                if (conn.get() != nullptr) {
                    successful_gets++;
                }
                // connection is automatically returned when conn goes out of scope
            }
        };

        std::vector<std::thread> threads;
        for (int i = 0; i < 10; ++i) {
            threads.emplace_back(worker);
        }

        for (auto& t : threads) {
            t.join();
        }

        EXPECT_EQ(successful_gets, 100);
    } catch (const pqxx::broken_connection& e) {
        GTEST_SKIP() << "Skipping test because DB is not available: " << e.what();
    } catch (const std::exception& e) {
        FAIL() << "Unexpected exception: " << e.what();
    }
}
