-- init.sql
-- 10 Tables for Movie Ticket Booking

CREATE TYPE seat_status AS ENUM ('AVAILABLE', 'LOCKED', 'BOOKED');
CREATE TYPE booking_status AS ENUM ('CREATED', 'SEATS_LOCKED', 'PAYMENT_PENDING', 'CONFIRMED', 'CANCELLED');
CREATE TYPE payment_status AS ENUM ('PENDING', 'SUCCESS', 'FAILED');

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(20) NOT NULL
);

CREATE TABLE cities (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL
);

CREATE TABLE theatres (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    city_id INT NOT NULL REFERENCES cities(id) ON DELETE CASCADE
);

CREATE TABLE screens (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    theatre_id INT NOT NULL REFERENCES theatres(id) ON DELETE CASCADE
);

CREATE TABLE movies (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    duration INT NOT NULL, -- duration in minutes
    language VARCHAR(50) NOT NULL
);

CREATE TABLE shows (
    id SERIAL PRIMARY KEY,
    movie_id INT NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    screen_id INT NOT NULL REFERENCES screens(id) ON DELETE CASCADE,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL
);

CREATE TABLE seats (
    id SERIAL PRIMARY KEY,
    screen_id INT NOT NULL REFERENCES screens(id) ON DELETE CASCADE,
    row_no INT NOT NULL,
    col_no INT NOT NULL,
    UNIQUE (screen_id, row_no, col_no)
);

CREATE TABLE bookings (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    show_id INT NOT NULL REFERENCES shows(id) ON DELETE CASCADE,
    status booking_status NOT NULL DEFAULT 'CREATED',
    amount DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE show_seats (
    id SERIAL PRIMARY KEY,
    show_id INT NOT NULL REFERENCES shows(id) ON DELETE CASCADE,
    seat_id INT NOT NULL REFERENCES seats(id) ON DELETE CASCADE,
    status seat_status NOT NULL DEFAULT 'AVAILABLE',
    lock_expiry_time TIMESTAMP,
    booking_id INT REFERENCES bookings(id) ON DELETE SET NULL,
    UNIQUE (show_id, seat_id)
);

CREATE TABLE payments (
    id SERIAL PRIMARY KEY,
    booking_id INT NOT NULL REFERENCES bookings(id) ON DELETE CASCADE,
    status payment_status NOT NULL DEFAULT 'PENDING',
    transaction_id VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
