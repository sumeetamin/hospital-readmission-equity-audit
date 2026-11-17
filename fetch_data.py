"""Download UCI dataset 296 and extract its encounter table."""
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us-hospitals-for-years-1999-2008.zip"
ARCHIVE = Path("data/diabetes_uci.zip")
OUT = Path("data/diabetic_data.csv")


def main():
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(URL, ARCHIVE)
    with ZipFile(ARCHIVE) as archive:
        with archive.open("diabetic_data.csv") as src, OUT.open("wb") as dst:
            dst.write(src.read())
    print(f"Extracted {OUT} from UCI dataset 296.")


if __name__ == "__main__":
    main()

