"""Tests for hello function."""
from gdemandfcast.ai import execute


def test_execute():

    train = "./gdemandfcast/data/Test Data.xlsx"
    test = "./gdemandfcast/data/Test Data.xlsx"
    df = execute(train, test, 3).frm()
    print(df)
