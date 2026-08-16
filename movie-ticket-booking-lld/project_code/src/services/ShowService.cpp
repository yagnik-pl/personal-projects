#include "ShowService.h"

ShowService::ShowService(
    std::shared_ptr<IShowRepository> showRepo,
    std::shared_ptr<IShowSeatRepository> showSeatRepo
) : showRepo_(showRepo), showSeatRepo_(showSeatRepo) {}

std::vector<Show> ShowService::getAllShows() {
    if (!showRepo_) {
        return {};
    }
    return showRepo_->getAllShows();
}

std::optional<Show> ShowService::getShow(int showId) {
    if (!showRepo_ || showId <= 0) {
        return std::nullopt;
    }
    return showRepo_->getShow(showId);
}

std::vector<ShowSeat> ShowService::getShowSeats(int showId) {
    if (!showSeatRepo_ || showId <= 0) {
        return {};
    }
    // Check if show exists
    if (showRepo_) {
        auto showOpt = showRepo_->getShow(showId);
        if (!showOpt.has_value()) {
            return {};
        }
    }
    // ShowSeatRepository dynamically resolves expired locks to AVAILABLE
    return showSeatRepo_->getShowSeatsByShow(showId);
}

std::vector<Show> ShowService::getShowsByMovie(int movieId) {
    if (!showRepo_ || movieId <= 0) {
        return {};
    }
    return showRepo_->getShowsByMovie(movieId);
}
