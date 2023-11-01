
import os

search_query = input("Enter your search query: ")
os.system(f"start https://duckduckgo.com/?q={search_query}")