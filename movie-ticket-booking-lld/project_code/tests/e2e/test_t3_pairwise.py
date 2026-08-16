import pytest

BASE_URL = "http://localhost:8080/api"

def test_t3_1_pairwise_single_seat_success_payment(api_session):
    resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 1, "seat_ids": [1]
    })
    assert resp.status_code == 201
    b_id = resp.json()["booking_id"]

    pay_resp = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": b_id, "amount": 25.0, "fail_mock": False
    })
    assert pay_resp.status_code == 200
    assert pay_resp.json()["status"] == "SUCCESS"

    status_resp = api_session.get(f"{BASE_URL}/bookings/{b_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "CONFIRMED"

def test_t3_2_pairwise_single_seat_fail_payment(api_session):
    resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 2, "seat_ids": [2]
    })
    assert resp.status_code == 201
    b_id = resp.json()["booking_id"]

    pay_resp = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": b_id, "amount": 25.0, "fail_mock": True
    })
    assert pay_resp.status_code == 400
    assert pay_resp.json()["status"] == "FAILED"

    status_resp = api_session.get(f"{BASE_URL}/bookings/{b_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] != "CONFIRMED"

def test_t3_3_pairwise_multi_seat_success_payment(api_session):
    resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 3, "seat_ids": [3, 4, 5]
    })
    assert resp.status_code == 201
    b_id = resp.json()["booking_id"]
    assert resp.json()["total_price"] == 75.0

    pay_resp = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": b_id, "amount": 75.0, "fail_mock": False
    })
    assert pay_resp.status_code == 200
    assert pay_resp.json()["status"] == "SUCCESS"

    status_resp = api_session.get(f"{BASE_URL}/bookings/{b_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "CONFIRMED"

def test_t3_4_pairwise_multi_seat_fail_payment(api_session):
    resp = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 4, "seat_ids": [6, 7]
    })
    assert resp.status_code == 201
    b_id = resp.json()["booking_id"]

    pay_resp = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": b_id, "amount": 50.0, "fail_mock": True
    })
    assert pay_resp.status_code == 400

    status_resp = api_session.get(f"{BASE_URL}/bookings/{b_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] != "CONFIRMED"

def test_t3_5_pairwise_concurrent_users_independent_bookings(api_session):
    resp1 = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 5, "seat_ids": [8, 9]
    })
    resp2 = api_session.post(f"{BASE_URL}/bookings", json={
        "show_id": 1, "user_id": 6, "seat_ids": [10, 11]
    })
    assert resp1.status_code == 201
    assert resp2.status_code == 201

    b1_id = resp1.json()["booking_id"]
    b2_id = resp2.json()["booking_id"]

    pay1 = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": b1_id, "amount": 50.0, "fail_mock": False
    })
    pay2 = api_session.post(f"{BASE_URL}/payments", json={
        "booking_id": b2_id, "amount": 50.0, "fail_mock": False
    })
    assert pay1.status_code == 200
    assert pay2.status_code == 200
