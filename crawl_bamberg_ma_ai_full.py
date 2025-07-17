import os
import re
import json
import requests
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from datetime import datetime
from tqdm import tqdm

# Only scrape pages with the following prefixes
PREFIXES = [
    "https://www.uni-bamberg.de/ma-ai/",
    "https://www.uni-bamberg.de/en/ma-ai/"
]

# Storage directories
date_str = datetime.today().strftime("%Y-%m-%d")
os.makedirs("cleaned_json", exist_ok=True)
os.makedirs("pdfs", exist_ok=True)

visited = set()
to_visit = set(PREFIXES)

def fetch_html(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"❌ Failed to fetch {url}: {e}")
        return None

def clean_and_save_text(url, html):
    downloaded = trafilatura.extract(html, favor_precision=True, include_comments=False, include_tables=False, no_fallback=True)
    if downloaded and len(downloaded.strip()) > 200:
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.text.strip() if title_tag else url.split("/")[-2]
        safe_title = re.sub(r'[\\/*?:"<>|]', "", title).strip().replace(" ", "_")
        data = {
            "url": url,
            "title": title,
            "date": date_str,
            "content": downloaded.strip()
        }
        filename = f"cleaned_json/{safe_title}_{date_str}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved: {filename}")
    else:
        print(f"⚠️ Content too short or failed to clean: {url}")

def extract_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    pdf_links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(base_url, href)
        if href.lower().endswith(".pdf"):
            pdf_links.add(full_url)
        else:
            if any(full_url.startswith(prefix) for prefix in PREFIXES):
                links.add(full_url.split("#")[0])
    return links, pdf_links

def download_pdf(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        filename = os.path.basename(urlparse(url).path)
        filename = re.sub(r'[\\/*?:"<>|]', "", filename)
        with open(os.path.join("pdfs", filename), "wb") as f:
            f.write(response.content)
        print(f"📥 Downloaded PDF: {filename}")
    except Exception as e:
        print(f"❌ Failed to download PDF {url}: {e}")

if __name__ == "__main__":
    while to_visit:
        current_url = to_visit.pop()
        if current_url in visited:
            continue
        visited.add(current_url)

        html = fetch_html(current_url)
        if html:
            clean_and_save_text(current_url, html)
            links, pdf_links = extract_links(html, current_url)
            to_visit.update(links - visited)
            for pdf_url in pdf_links:
                download_pdf(pdf_url)

    print("✅ Completed scraping texts and PDFs from Bamberg MA-AI program pages and subpages.")
