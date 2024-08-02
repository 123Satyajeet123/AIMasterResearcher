# tools/scrape_tool.py
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
from langchain_core.tools import Tool
from pydantic import BaseModel, Field

class ScrapeWebsiteInput(BaseModel):
    url: str = Field(..., description="The URL to scrape")

def scrape_website(url: str):
    print(f"Scraping website: {url}")
    service = Service(ChromeDriverManager().install())
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    time.sleep(5)
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    driver.quit()
    
    # Extract main content (this is a simple approach and might need refinement)

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    word_count = len(text.split())
    
    return {
        "content": text,
        "word_count": word_count,
        "url": url
    }


# print("Scrape tool loaded")
# result = scrape_website("https://ai.meta.com/blog/meta-llama-3-1/")
# print(result)


scrape_tool = Tool(
    name="ScrapeWebsite",
    func=scrape_website,
    description="Extracts the main content from a given URL.",
    args_schema=ScrapeWebsiteInput
)