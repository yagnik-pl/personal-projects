#pragma once
#include <optional>
#include "../models/Theatre.h"

class ITheatreRepository {
public:
    virtual ~ITheatreRepository() = default;
    virtual Theatre createTheatre(const Theatre& theatre) = 0;
    virtual std::optional<Theatre> getTheatre(int id) = 0;
};
