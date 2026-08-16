#include "core/Database.h"
#include <stdexcept>

// --- ConnectionProxy ---

ConnectionProxy::ConnectionProxy(Database* db, std::unique_ptr<pqxx::connection> conn)
    : db_(db), conn_(std::move(conn)) {}

ConnectionProxy::~ConnectionProxy() {
    if (conn_) {
        db_->releaseConnection(std::move(conn_));
    }
}

ConnectionProxy::ConnectionProxy(ConnectionProxy&& other) noexcept
    : db_(other.db_), conn_(std::move(other.conn_)) {
    other.db_ = nullptr;
}

ConnectionProxy& ConnectionProxy::operator=(ConnectionProxy&& other) noexcept {
    if (this != &other) {
        if (conn_) {
            db_->releaseConnection(std::move(conn_));
        }
        db_ = other.db_;
        conn_ = std::move(other.conn_);
        other.db_ = nullptr;
    }
    return *this;
}

pqxx::connection* ConnectionProxy::operator->() {
    return conn_.get();
}

pqxx::connection& ConnectionProxy::operator*() {
    return *conn_;
}

pqxx::connection* ConnectionProxy::get() {
    return conn_.get();
}

// --- Database ---

Database::Database(const std::string& connection_string, size_t pool_size)
    : connection_string_(connection_string), stop_(false) {
    if (pool_size == 0) {
        throw std::invalid_argument("Pool size must be greater than 0");
    }

    for (size_t i = 0; i < pool_size; ++i) {
        pool_.push(std::make_unique<pqxx::connection>(connection_string_));
    }
}

Database::~Database() {
    std::unique_lock<std::mutex> lock(mutex_);
    stop_ = true;
    cv_.notify_all();
    
    while (!pool_.empty()) {
        pool_.pop();
    }
}

ConnectionProxy Database::getConnection() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return stop_ || !pool_.empty(); });

    if (stop_ && pool_.empty()) {
        throw std::runtime_error("Database connection pool is stopped.");
    }

    auto conn = std::move(pool_.front());
    pool_.pop();

    return ConnectionProxy(this, std::move(conn));
}

void Database::releaseConnection(std::unique_ptr<pqxx::connection> conn) {
    std::unique_lock<std::mutex> lock(mutex_);
    pool_.push(std::move(conn));
    lock.unlock();
    cv_.notify_one();
}
