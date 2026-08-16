#pragma once
#include "ITheatreRepository.h"
#include "../core/Database.h"
#include <memory>

class TheatreRepository : public ITheatreRepository {
public:
    explicit TheatreRepository(std::shared_ptr<Database> db);
    Theatre createTheatre(const Theatre& theatre) override;
    std::optional<Theatre> getTheatre(int id) override;
private:
    std::shared_ptr<Database> db_;
};
