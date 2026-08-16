#pragma once
#include "IUserRepository.h"
#include "../core/Database.h"
#include <memory>

class UserRepository : public IUserRepository {
public:
    explicit UserRepository(std::shared_ptr<Database> db);
    User createUser(const User& user) override;
    std::optional<User> getUser(int id) override;
private:
    std::shared_ptr<Database> db_;
};
