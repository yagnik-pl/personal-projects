#include "UserRepository.h"
#include <pqxx/pqxx>

UserRepository::UserRepository(std::shared_ptr<Database> db) : db_(db) {}

User UserRepository::createUser(const User& user) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "INSERT INTO users (name, email, phone) VALUES ($1, $2, $3) RETURNING id",
        user.name, user.email, user.phone
    );
    w.commit();

    User createdUser = user;
    createdUser.id = r[0][0].as<int>();
    return createdUser;
}

std::optional<User> UserRepository::getUser(int id) {
    auto conn = db_->getConnection();
    pqxx::work w(*conn);
    pqxx::result r = w.exec_params(
        "SELECT id, name, email, phone FROM users WHERE id = $1",
        id
    );

    if (r.empty()) {
        return std::nullopt;
    }

    User user;
    user.id = r[0]["id"].as<int>();
    user.name = r[0]["name"].as<std::string>();
    user.email = r[0]["email"].as<std::string>();
    user.phone = r[0]["phone"].as<std::string>();

    return user;
}
