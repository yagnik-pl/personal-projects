#pragma once
#include <optional>
#include "../models/User.h"

class IUserRepository {
public:
    virtual ~IUserRepository() = default;
    virtual User createUser(const User& user) = 0;
    virtual std::optional<User> getUser(int id) = 0;
};
