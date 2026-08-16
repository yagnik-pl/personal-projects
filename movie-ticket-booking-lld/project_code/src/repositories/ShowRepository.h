#pragma once
#include "IShowRepository.h"
#include "../core/Database.h"
#include <memory>

class ShowRepository : public IShowRepository {
public:
    explicit ShowRepository(std::shared_ptr<Database> db);
    Show createShow(const Show& show) override;
    std::optional<Show> getShow(int id) override;
    std::vector<Show> getAllShows() override;
    std::vector<Show> getShowsByMovie(int movieId) override;
    std::vector<Show> getShowsByScreen(int screenId) override;

private:
    std::shared_ptr<Database> db_;
};
