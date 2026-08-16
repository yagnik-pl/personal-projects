#pragma once
#include "IScreenRepository.h"
#include "../core/Database.h"
#include <memory>

class ScreenRepository : public IScreenRepository {
public:
    explicit ScreenRepository(std::shared_ptr<Database> db);
    Screen createScreen(const Screen& screen) override;
    std::optional<Screen> getScreen(int id) override;
private:
    std::shared_ptr<Database> db_;
};
