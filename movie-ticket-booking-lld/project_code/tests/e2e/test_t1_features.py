import pytest

BASE_URL = "http://localhost:8080/api"

# Feature 1: Book Ticket Lifecycle (5 tests)

def test_f1_1_get_shows(api_session):
    resp = api_session.get(f"{BASE_URL}/shows")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["id"] == 1
    assert "movie_id" in data[0]
    assert "screen_id" in data[0]

def test_f1_2_get_show_seats(api_session):
    resp = api_session.get(f"{BASE_URL}/shows/1/seats")
    assert resp.status_code == 200
    seats = resp.json()
    assert isinstance(seats, list)
    assert len(seats) == 20
    for s in seats:
        assert s["status"] == "AVAILABLE"
        assert s["show_id"] == 1
        assert 1 <= s["seat_id"] <= 20

def test_f1_3_book_single_seat_success(api_session):
    resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1,
        "user_id": 1,
        "seat_ids": [1]
    })
    assert resp.status_code == 201
    data = resp.json()
    assert "booking_id" in data
    assert data["status"] == "SEATS_LOCKED"
    assert data["total_price"] == 25.0

def test_f1_4_book_multiple_seats_success(api_session):
    resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1,
        "user_id": 1,
        "seat_ids": [1, 2]
    })
    assert resp.status_code == 201
    data = resp.json()
    assert "booking_id" in data
    assert data["status"] == "SEATS_LOCKED"
    assert data["total_price"] == 50.0

def test_f1_5_complete_booking_lifecycle(api_session):
    # 1. Create booking
    book_resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1,
        "user_id": 1,
        "seat_ids": [1, 2]
    })
    assert book_resp.status_code == 201
    booking_id = book_resp.json()["booking_id"]

    # 2. Make payment
    pay_resp = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": booking_id,
        "amount": 50.0,
        "fail_mock": False
    })
    assert pay_resp.status_code == 200
    assert pay_resp.json()["status"] == "SUCCESS"

    # 3. Verify confirmed booking status
    status_resp = api_session.get(f"{BASE_URL}/bookings/{booking_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "CONFIRMED"
    assert status_resp.json()["total_price"] == 50.0


# Feature 2: Handle Payment Failures (5 tests)

def test_f2_1_payment_failure_mock(api_session):
    book_resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1,
        "user_id": 2,
        "seat_ids": [3]
    })
    assert book_resp.status_code == 201
    booking_id = book_resp.json()["booking_id"]

    pay_resp = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": booking_id,
        "amount": 25.0,
        "fail_mock": True
    })
    assert pay_resp.status_code == 400
    assert pay_resp.json()["status"] == "FAILED"

def test_f2_2_booking_status_not_confirmed_on_payment_failure(api_session):
    book_resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1,
        "user_id": 2,
        "seat_ids": [3]
    })
    assert book_resp.status_code == 201
    booking_id = book_resp.json()["booking_id"]

    api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": booking_id,
        "amount": 25.0,
        "fail_mock": True
    })

    status_resp = api_session.get(f"{BASE_URL}/bookings/{booking_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] != "CONFIRMED"

def test_f2_3_payment_retry_success_after_failure(api_session):
    book_resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1,
        "user_id": 3,
        "seat_ids": [4]
    })
    assert book_resp.status_code == 201
    booking_id = book_resp.json()["booking_id"]

    # First attempt fails
    fail_resp = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": booking_id,
        "amount": 25.0,
        "fail_mock": True
    })
    assert fail_resp.status_code == 400

    # Second attempt succeeds
    succ_resp = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": booking_id,
        "amount": 25.0,
        "fail_mock": False
    })
    assert succ_resp.status_code == 200
    assert succ_resp.json()["status"] == "SUCCESS"

    status_resp = api_session.get(f"{BASE_URL}/bookings/{booking_id}")
    assert status_resp.json()["status"] == "CONFIRMED"

def test_f2_4_payment_incorrect_amount_rejected(api_session):
    book_resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1,
        "user_id": 4,
        "seat_ids": [5, 6]
    })
    assert book_resp.status_code == 201
    booking_id = book_resp.json()["booking_id"]

    # Total is 50.0, paying 25.0 should fail
    pay_resp = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": booking_id,
        "amount": 25.0,
        "fail_mock": False
    })
    assert pay_resp.status_code == 400
    assert pay_resp.json()["status"] == "FAILED"

def test_f2_5_payment_success_response_fields(api_session):
    book_resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1,
        "user_id": 5,
        "seat_ids": [7]
    })
    assert book_resp.status_code == 201
    booking_id = book_resp.json()["booking_id"]

    pay_resp = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": booking_id,
        "amount": 25.0,
        "fail_mock": False
    })
    assert pay_resp.status_code == 200
    data = pay_resp.json()
    assert "payment_id" in data
    assert data["status"] == "SUCCESS"
    assert "transaction_id" in data


# Feature 3: Concurrency / Double Booking Prevention (5 tests)

def test_f3_1_concurrency_same_seat_double_booking(api_session):
    resp1 = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 1, "seat_ids": [8]
    })
    resp2 = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 2, "seat_ids": [8]
    })
    statuses = [resp1.status_code, resp2.status_code]
    assert 201 in statuses
    assert 409 in statuses

def test_f3_2_concurrency_overlapping_seats(api_session):
    resp1 = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 3, "seat_ids": [9, 10]
    })
    resp2 = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 4, "seat_ids": [10, 11]
    })
    statuses = [resp1.status_code, resp2.status_code]
    assert 201 in statuses
    assert 409 in statuses

def test_f3_3_booked_seat_cannot_be_booked(api_session):
    # Book & confirm seat 12
    resp1 = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 5, "seat_ids": [12]
    })
    assert resp1.status_code == 201
    b_id = resp1.json()["booking_id"]

    pay_resp = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": b_id, "amount": 25.0, "fail_mock": False
    })
    assert pay_resp.status_code == 200

    # User 6 tries to book the same seat 12
    resp2 = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 6, "seat_ids": [12]
    })
    assert resp2.status_code == 409

def test_f3_4_locked_seat_cannot_be_booked(api_session):
    # User 7 locks seat 13
    resp1 = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 7, "seat_ids": [13]
    })
    assert resp1.status_code == 201

    # User 8 tries to lock seat 13 while still locked
    resp2 = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 8, "seat_ids": [13]
    })
    assert resp2.status_code == 409

def test_f3_5_concurrent_disjoint_seats_both_succeed(api_session):
    resp1 = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 9, "seat_ids": [14, 15]
    })
    resp2 = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 10, "seat_ids": [16, 17]
    })
    assert resp1.status_code == 201
    assert resp2.status_code == 201


# Feature 4: Seat Lock Status & Lifecycle (5 tests)

def test_f4_1_seats_show_locked_status(api_session):
    resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 1, "seat_ids": [18]
    })
    assert resp.status_code == 201

    seats_resp = api_session.get(f"{BASE_URL}/shows/1/seats")
    assert seats_resp.status_code == 200
    seats = {s["seat_id"]: s["status"] for s in seats_resp.json()}
    assert seats[18] == "LOCKED"

def test_f4_2_seats_show_booked_status(api_session):
    book_resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 1, "seat_ids": [19]
    })
    assert book_resp.status_code == 201
    b_id = book_resp.json()["booking_id"]

    pay_resp = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": b_id, "amount": 25.0, "fail_mock": False
    })
    assert pay_resp.status_code == 200

    seats_resp = api_session.get(f"{BASE_URL}/shows/1/seats")
    assert seats_resp.status_code == 200
    seats = {s["seat_id"]: s["status"] for s in seats_resp.json()}
    assert seats[19] == "BOOKED"

def test_f4_3_booking_get_status_seats_locked(api_session):
    resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 1, "seat_ids": [20]
    })
    assert resp.status_code == 201
    b_id = resp.json()["booking_id"]

    status_resp = api_session.get(f"{BASE_URL}/bookings/{b_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "SEATS_LOCKED"
    assert status_resp.json()["booking_id"] == b_id

def test_f4_4_seats_available_after_test_reset(api_session):
    # Book some seats
    api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 1, "seat_ids": [1, 2, 3]
    })
    # Reset database
    reset_resp = api_session.post(f"{BASE_URL}/test/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json()["status"] == "OK"

    # Verify all seats are available again
    seats_resp = api_session.get(f"{BASE_URL}/shows/1/seats")
    assert seats_resp.status_code == 200
    for s in seats_resp.json():
        assert s["status"] == "AVAILABLE"

def test_f4_5_multiple_shows_independence(api_session):
    resp = api_session.get(f"{BASE_URL}/shows")
    assert resp.status_code == 200
    shows = resp.json()
    assert len(shows) >= 1
    show_id = shows[0]["id"]
    seats_resp = api_session.get(f"{BASE_URL}/shows/{show_id}/seats")
    assert seats_resp.status_code == 200
    assert len(seats_resp.json()) == 20
