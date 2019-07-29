import sys
import requests
import urllib.request
import time
from bs4 import BeautifulSoup

ref_name = sys.argv[1] if len(sys.argv) >= 2 else 'methods'

target_file = "./training-data/utils/querywords/js_mdn_scraped_" + ref_name + ".wl"

url = (
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/"
    + ref_name.capitalize()
    + "_Index"
)
response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

code_tags = [a.find("code") for a in soup.findAll("a")]
code_tags = [code for code in code_tags if code is not None]
names = [code.text.replace(".", "").replace("()", "") for code in code_tags]

with open(target_file, "w") as f:
    for item in names:
        f.write("%s\n" % item)

time.sleep(1)
