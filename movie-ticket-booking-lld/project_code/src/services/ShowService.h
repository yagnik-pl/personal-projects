#pragma once
#include "IShowService.h"
#include "../repositories/IShowRepository.h"
#include "../repositories/IShowSeatRepository.h"
#include <memory>

class ShowService : public IShowService {
public:
    ShowService(
        std::shared_ptr<IShowRepository> showRepo,
        std::shared_ptr<IShowSeatRepository> showSeatRepo
    );

    std::vector<Show> getAllShows() override;
    std::optional<Show> getShow(int showId) override;
    std::vector<ShowSeat> getShowSeats(int showId) override;
    std::vector<Show> getShowsByMovie(int movieId) override;

private:
    std::shared_ptr<IShowRepository> showRepo_;
    std::shared_ptr<IShowSeatRepository> showSeatRepo_;
};
