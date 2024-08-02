# from typing import Type

# from utils.websiteInput import ScrapeWebsiteInput
# from utils.scrape import scrape

# from pydantic import BaseModel, Field

# class ScrapeWebsiteTool(BaseModel):
#     name = "Scrape Website"
#     description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT MAKE A LOT OF REQUESTS TO THE SAME WEBSITE, IT MAY CAUSE A BAN"

#     args_schema: Type[BaseModel] = ScrapeWebsiteInput

#     def _run(self, objective: str, url: str):
#         return scrape(objective, url)
    
#     def _error_message(self, url: str):
#         print(f"Error scraping {url}")
#         return NotImplementedError("Error")