#pragma once

#include <pqxx/pqxx>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>

class Database;

// RAII wrapper for a database connection returned from the pool
class ConnectionProxy {
public:
    ConnectionProxy(Database* db, std::unique_ptr<pqxx::connection> conn);
    ~ConnectionProxy();

    // Move semantics only
    ConnectionProxy(ConnectionProxy&& other) noexcept;
    ConnectionProxy& operator=(ConnectionProxy&& other) noexcept;

    ConnectionProxy(const ConnectionProxy&) = delete;
    ConnectionProxy& operator=(const ConnectionProxy&) = delete;

    pqxx::connection* operator->();
    pqxx::connection& operator*();
    pqxx::connection* get();

private:
    Database* db_;
    std::unique_ptr<pqxx::connection> conn_;
};

class Database {
public:
    // Initialize the pool with a connection string and pool size
    Database(const std::string& connection_string, size_t pool_size);
    ~Database();

    // Get a connection from the pool, blocking if none are available
    ConnectionProxy getConnection();

    // Return a connection to the pool
    void releaseConnection(std::unique_ptr<pqxx::connection> conn);

private:
    std::string connection_string_;
    std::queue<std::unique_ptr<pqxx::connection>> pool_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_;
};
