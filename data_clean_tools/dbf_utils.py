# -*- coding: utf-8 -*-

"""
    Author:     Bin Liang
    Version:    1.0
    Date:       29/03/2018
    DBF utils
"""
from dbfread import DBF, FieldParser, InvalidValue
import pandas as pd


class MyFieldParser(FieldParser):
    def parse(self, field, data):
        try:
            return FieldParser.parse(self, field, data)
        except ValueError:
            return InvalidValue(data)


def read_dbf(dbf_filepath):
    """
        convert dbf file to csv file
    """
    table = DBF(dbf_filepath, parserclass=MyFieldParser)
    data_df = pd.DataFrame(iter(table))
    return data_df

