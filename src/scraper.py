import asyncio
import os
import aiohttp # Use aiohttp for async requests
import json
from pathlib import Path # Import Path
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from crawl4ai import AsyncWebCrawler
from typing import Optional, List, Dict, Set, Tuple

# Define target file extensions
TARGET_EXTENSIONS = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.zip', '.rar']

# Use pathlib for cross-platform paths
BASE_DIR = Path.cwd() # Or choose a more specific base if needed
DOWNLOAD_DIR = BASE_DIR / 'downloads'
SITEMAP_PATH = BASE_DIR / 'sitemap.json' # Path object for the sitemap file

async def download_file(session: aiohttp.ClientSession, url: str, target_dir: Path): # Use Path type hint
    """Downloads a single file asynchronously from a URL using aiohttp."""
    try:
        # Ensure the target directory exists using pathlib
        target_dir.mkdir(parents=True, exist_ok=True)

        # Get the filename from the URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename: # Handle cases where path ends with /
            # Try to get filename from Content-Disposition header later if possible
            filename = f"downloaded_file_{os.urandom(4).hex()}{os.path.splitext(parsed_url.path)[1]}" # Keep extension if present

        filepath = target_dir / filename # Use pathlib for joining paths

        print(f"Attempting download: {url} -> {filepath}")
        async with session.get(url, allow_redirects=True) as response:
            if response.status == 200:
                # Optional: Try to get a better filename from headers
                content_disposition = response.headers.get('Content-Disposition')
                if content_disposition:
                    import re
                    fname_match = re.search(r'filename=['"?]?([^"'\n;]+)', content_disposition)
                    if fname_match:
                        header_filename = fname_match.group(1)
                        # Basic sanitization
                        header_filename = header_filename.strip().replace('/', '_').replace('\\', '_')
                        if header_filename:
                            filepath = target_dir / header_filename # Use pathlib here too
                            print(f"Using filename from header: {header_filename}")

                with open(filepath, 'wb') as f:
                    while True:
                        chunk = await response.content.read(8192) # Read in chunks
                        if not chunk:
                            break
                        f.write(chunk)
                print(f"Successfully downloaded {filepath.name}") # Use pathlib's name attribute
                return True
            else:
                print(f"Error downloading {url}: Status {response.status}")
                return False

    except aiohttp.ClientError as e:
        print(f"Network error downloading {url}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while downloading {url}: {e}")
    return False

async def _crawl_internal_links(session: aiohttp.ClientSession, url: str, base_domain: str, visited_urls: Set[str], current_depth: int, max_depth: int) -> Set[str]:
    """Recursively crawls internal links up to a max depth."""
    if current_depth > max_depth or url in visited_urls:
        return set()

    print(f"Crawling (depth {current_depth}): {url}")
    visited_urls.add(url)
    internal_links = {url} # Include the current URL

    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200 and 'text/html' in response.headers.get('Content-Type', ''):
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'html.parser')

                tasks = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    absolute_url = urljoin(url, href)
                    parsed_absolute_url = urlparse(absolute_url)

                    # Check if it's an internal link (same domain) and not an anchor
                    if parsed_absolute_url.netloc == base_domain and not parsed_absolute_url.fragment:
                        # Normalize URL (optional, remove query params/fragments if desired)
                        normalized_url = f"{parsed_absolute_url.scheme}://{parsed_absolute_url.netloc}{parsed_absolute_url.path}"
                        if normalized_url not in visited_urls:
                            # Create a task for recursive crawl
                            task = asyncio.create_task(
                                _crawl_internal_links(session, normalized_url, base_domain, visited_urls, current_depth + 1, max_depth)
                            )
                            tasks.append(task)

                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, set):
                            internal_links.update(result)
                        elif isinstance(result, Exception):
                            print(f"Error during sub-crawl: {result}")
            else:
                print(f"Skipping non-HTML or error page: {url} (Status: {response.status})")

    except aiohttp.ClientError as e:
        print(f"Network error crawling {url}: {e}")
    except asyncio.TimeoutError:
        print(f"Timeout crawling {url}")
    except Exception as e:
        print(f"Error crawling {url}: {e}")

    return internal_links

async def generate_sitemap(start_url: str, max_depth: int = 1) -> List[str]:
    """Generates a list of internal URLs found by crawling."""
    parsed_start_url = urlparse(start_url)
    base_domain = parsed_start_url.netloc
    if not base_domain:
        print(f"Invalid start URL for sitemap generation: {start_url}")
        return []

    visited_urls: Set[str] = set()
    sitemap_links: Set[str] = set()

    print(f"\nGenerating sitemap starting from: {start_url} (max depth: {max_depth})")
    async with aiohttp.ClientSession() as session:
        sitemap_links = await _crawl_internal_links(session, start_url, base_domain, visited_urls, 0, max_depth)

    print(f"Sitemap generation complete. Found {len(sitemap_links)} unique internal URLs.")
    return sorted(list(sitemap_links))

def save_sitemap(sitemap_data: List[str], filepath: Path = SITEMAP_PATH): # Use Path type hint and default
    """Saves the sitemap data to a JSON file using pathlib."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sitemap_data, f, indent=4, ensure_ascii=False)
        print(f"Sitemap successfully saved to {filepath}")
    except IOError as e:
        print(f"Error saving sitemap to {filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving sitemap: {e}")


async def scrape_website(url: str, sitemap_depth: int = 1) -> Tuple[Optional[str], List[str]]:
    """Scrapes the text content of the given URL using Crawl4AI and downloads specified file types asynchronously."""

    print(f"Attempting to scrape text content from: {url}")
    scraped_markdown = None
    try:
        # Use Crawl4AI for text content extraction
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result and result.markdown:
                print("Text scraping successful.")
                scraped_markdown = result.markdown
            else:
                print(f"Failed to scrape text content from {url}. Result: {result}")
    except Exception as e:
        print(f"An error occurred during text scraping {url}: {e}")

    print(f"\nAttempting to find and download files from: {url}")
    download_tasks = []
    try:
        # Use aiohttp to fetch the page for link analysis
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    html_content = await response.text()
                    soup = BeautifulSoup(html_content, 'html.parser')

                    links_found = 0
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        # Resolve relative URLs
                        absolute_url = urljoin(url, href)

                        # Check if the URL has a target extension
                        parsed_link_url = urlparse(absolute_url)
                        _, ext = os.path.splitext(parsed_link_url.path)

                        if ext.lower() in TARGET_EXTENSIONS:
                            # Basic check to avoid downloading the page itself if it ends with e.g. .html but is listed
                            if parsed_link_url.scheme in ['http', 'https']:
                                links_found += 1
                                print(f"Found potential file link: {absolute_url}")
                                # Create a download task
                                task = asyncio.create_task(download_file(session, absolute_url, DOWNLOAD_DIR))
                                download_tasks.append(task)
                            else:
                                print(f"Skipping non-http(s) link: {absolute_url}")

                    if links_found == 0:
                        print("No links matching target file extensions found.")
                    else:
                        print(f"\nStarting download of {len(download_tasks)} files...")
                        results = await asyncio.gather(*download_tasks, return_exceptions=True)
                        print("\nFile downloads attempted.")
                        # Optionally check results for errors
                        success_count = sum(1 for r in results if isinstance(r, bool) and r)
                        fail_count = len(results) - success_count
                        print(f"Downloads completed: {success_count} successful, {fail_count} failed/skipped.")
                else:
                    print(f"Error fetching page for file download analysis {url}: Status {response.status}")

    except aiohttp.ClientError as e:
        print(f"Network error fetching page for file analysis {url}: {e}")
    except asyncio.TimeoutError:
        print(f"Timeout fetching page for file analysis {url}")
    except Exception as e:
        print(f"An error occurred during file download analysis {url}: {e}")

    # --- Sitemap Generation --- 
    sitemap_data = []
    try:
        sitemap_data = await generate_sitemap(url, max_depth=sitemap_depth)
    except Exception as e:
        print(f"An error occurred during sitemap generation: {e}")

    return scraped_markdown, sitemap_data

# Example usage (for testing purposes)
# async def main():
#     # Replace with a URL known to have downloadable files of target types
#     test_url = "https://www.python.org/downloads/" # Example: Look for .zip/.tgz
#     # test_url = "https://file-examples.com/index.php/sample-documents-download/sample-pdf-download/" # Example: PDF
#     content = await scrape_website(test_url)
#     if content:
#         print("\n--- Scraped Content (Markdown Summary) ---")
#         print(content[:200] + "...") # Print first 200 chars
#     else:
#         print("\nFailed to retrieve text content.")
#     print(f"\nFiles (if any) were attempted to be downloaded to the '{DOWNLOAD_DIR}' directory.")
#
if __name__ == "__main__":
    async def main_test():
        # test_url = "https://www.python.org/"
        test_url = "https://docs.python.org/3/library/asyncio.html" # Example with internal links
        content, sitemap = await scrape_website(test_url, sitemap_depth=1)
        if content:
            print("\n--- Scraped Content (Markdown Summary) ---")
            print(content[:200] + "...")
        else:
            print("\nFailed to retrieve text content.")

        if sitemap:
            print("\n--- Generated Sitemap --- ")
            # print(sitemap)
            save_sitemap(sitemap, "test_sitemap.json")
        else:
            print("\nFailed to generate sitemap.")

        print(f"\nFiles (if any) were attempted to be downloaded to the '{DOWNLOAD_DIR}' directory.")

    asyncio.run(main_test())