import pytest

BASE_URL = "http://localhost:8080/api"

# Feature 1 Boundaries (5 tests)

def test_t2_1_book_zero_seats(api_session):
    resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 1, "seat_ids": []
    })
    assert resp.status_code == 400

def test_t2_2_book_invalid_show_id_negative(api_session):
    resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": -1, "user_id": 1, "seat_ids": [1]
    })
    assert resp.status_code in [400, 404]

def test_t2_3_book_nonexistent_show_id(api_session):
    resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 99999, "user_id": 1, "seat_ids": [1]
    })
    assert resp.status_code == 404

def test_t2_4_book_invalid_user_id_negative(api_session):
    resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": -1, "seat_ids": [1]
    })
    assert resp.status_code in [400, 404]

def test_t2_5_book_nonexistent_user_id(api_session):
    resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 999999, "seat_ids": [1]
    })
    assert resp.status_code == 404


# Feature 2 Boundaries (5 tests)

def test_t2_6_payment_negative_amount(api_session):
    book_resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 1, "seat_ids": [1]
    })
    assert book_resp.status_code == 201
    b_id = book_resp.json()["booking_id"]

    pay_resp = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": b_id, "amount": -25.0, "fail_mock": False
    })
    assert pay_resp.status_code == 400

def test_t2_7_payment_zero_amount(api_session):
    book_resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 1, "seat_ids": [1]
    })
    assert book_resp.status_code == 201
    b_id = book_resp.json()["booking_id"]

    pay_resp = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": b_id, "amount": 0.0, "fail_mock": False
    })
    assert pay_resp.status_code == 400

def test_t2_8_payment_nonexistent_booking(api_session):
    resp = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": 999999, "amount": 100.0, "fail_mock": False
    })
    assert resp.status_code == 404

def test_t2_9_payment_negative_booking_id(api_session):
    resp = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": -1, "amount": 25.0, "fail_mock": False
    })
    assert resp.status_code in [400, 404]

def test_t2_10_payment_excessive_amount(api_session):
    book_resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 1, "seat_ids": [1]
    })
    assert book_resp.status_code == 201
    b_id = book_resp.json()["booking_id"]

    pay_resp = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": b_id, "amount": 1000.0, "fail_mock": False
    })
    assert pay_resp.status_code == 400


# Feature 3 Boundaries (5 tests)

def test_t2_11_duplicate_seats_in_booking_request(api_session):
    resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 1, "seat_ids": [1, 1]
    })
    assert resp.status_code == 400

def test_t2_12_book_too_many_seats(api_session):
    resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 1, "seat_ids": list(range(1, 100))
    })
    assert resp.status_code in [400, 409]

def test_t2_13_seat_id_out_of_range_negative(api_session):
    resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 1, "seat_ids": [-5]
    })
    assert resp.status_code in [400, 404, 409]

def test_t2_14_seat_id_out_of_range_high(api_session):
    resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 1, "seat_ids": [9999]
    })
    assert resp.status_code in [400, 404, 409]

def test_t2_15_all_available_seats_booked_simultaneously(api_session):
    all_seats = list(range(1, 21))
    resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 1, "seat_ids": all_seats
    })
    assert resp.status_code == 201
    assert resp.json()["total_price"] == 500.0

    # Next booking must fail as all seats are locked
    subsequent = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 2, "seat_ids": [1]
    })
    assert subsequent.status_code == 409


# Feature 4 Boundaries (5 tests)

def test_t2_16_get_nonexistent_booking_id(api_session):
    resp = api_session.get(f"{BASE_URL}/bookings/999999")
    assert resp.status_code == 404

def test_t2_17_get_negative_booking_id(api_session):
    resp = api_session.get(f"{BASE_URL}/bookings/-1")
    assert resp.status_code == 404

def test_t2_18_get_nonexistent_show_seats(api_session):
    resp = api_session.get(f"{BASE_URL}/shows/999999/seats")
    assert resp.status_code == 404

def test_t2_19_get_negative_show_seats(api_session):
    resp = api_session.get(f"{BASE_URL}/shows/-1/seats")
    assert resp.status_code == 404

def test_t2_20_test_reset_idempotency(api_session):
    resp1 = api_session.post(f"{BASE_URL}/test/reset")
    assert resp1.status_code == 200
    resp2 = api_session.post(f"{BASE_URL}/test/reset")
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "OK"
