import requests
from bs4 import BeautifulSoup
import os
import re

BASE_URL = "http://shakespeare.mit.edu/Poetry/sonnets.html"
SONNET_BASE = "http://shakespeare.mit.edu/Poetry/"
OUTPUT_FILE = "data/shakespeare_sonnets.txt"

os.makedirs("data", exist_ok=True)

def get_sonnet_links():
    response = requests.get(BASE_URL)
    soup = BeautifulSoup(response.text, "html.parser")
    links = [SONNET_BASE + a["href"] for a in soup.find_all("a", href=True) if a["href"].startswith("sonnet.")]
    return links

def fetch_and_save_sonnets():
    links = get_sonnet_links()
    all_lines = []
    for link in links:
        resp = requests.get(link)
        page = BeautifulSoup(resp.text, "html.parser")
        raw_text = page.get_text(separator="\n")
        lines = [line.strip() for line in raw_text.split("\n")]

        cleaned_lines = []
        for line in lines:
            # Skip empty lines
            if not line:
                continue
            # Remove "Sonnet I", "Sonnet II", etc.
            if re.match(r"^Sonnet\s+[IVXLCDM]+$", line):
                continue
            # Remove uppercase headings like "FROM ...", "BY ...", etc.
            if re.match(r"^[A-Z\s]{3,}$", line):
                continue
            # Keep only poetic lines
            cleaned_lines.append(line)

        all_lines.extend(cleaned_lines + [""])  # Add empty line between sonnets
        print(f"Crawled and cleaned: {link}")

    # Save cleaned text directly
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines))

if __name__ == "__main__":
    fetch_and_save_sonnets()
    print(f"✅ Done! Cleaned sonnets saved to {OUTPUT_FILE}")
