import pandas as pd
import pycountry


def convert_country(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        df["country"] = df["country"].apply(
            lambda country: pycountry.countries.search_fuzzy(country.replace("_", " "))[0].name
        )
    except:
        pass
    return df
