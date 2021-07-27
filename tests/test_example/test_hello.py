"""Tests for hello function."""
# import pytest

from gdemandfcast.ai import execute

# from gdemandfcast.example import hello


# @pytest.mark.parametrize(
# ("name", "expected"),
# [
# ("Jeanette", "Hello Jeanette!"),
# ("Raven", "Hello Raven!"),
# ("Maxine", "Hello Maxine!"),
# ("Matteo", "Hello Matteo!")
# ],
# )


# def test_hello(name, expected):
# """Example test with parametrization."""
# assert hello(name) == expected


def test_execute():

    train = "./gdemandfcast/data/Test Data.xlsx"
    test = "./gdemandfcast/data/Test Data.xlsx"
    df = execute(train, test, 3).frm()
    assert print(df)
