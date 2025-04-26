try:
    import click
except ImportError:
    raise ImportError("Could not import click. Please install it using: pip install click")
import asyncio
import os
import subprocess
import sys
from pathlib import Path # Import Path
from playwright.sync_api import Error as PlaywrightError
# Remove find_dotenv and set_key from here, they are handled in ai_clients now
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, IntPrompt

from scraper import scrape_website, save_sitemap, SITEMAP_FILENAME
from ai_clients import (
    get_gemini_client,
    get_groq_client,
    get_openai_client,
    get_ollama_client,
    extract_with_gemini,
    extract_with_groq,
    extract_with_openai,
    extract_with_ollama,
    get_api_key,
    save_api_key_to_env # Import the new save function
)

# Initialize Rich Console
console = Console()

# .env file handling is now managed within ai_clients.py using appdirs

def check_playwright_installation():
    """Checks if Playwright browsers seem to be installed."""
    try:
        # Try running a command that requires browsers
        # Using 'pw:api' context check as a proxy for browser installation
        # This isn't perfect but avoids running a full browser launch
        process = subprocess.run(['playwright', 'install', '--dry-run'], capture_output=True, text=True, check=False)
        # A more direct check might be better if Playwright offers one.
        # For now, we assume if 'install --dry-run' runs without specific browser errors, it's likely okay.
        # A more robust check might involve trying to launch a browser briefly.
        if "Please run the following command" in process.stderr:
             # This suggests browsers might be missing
             return False
        # Check if playwright command exists at all
        subprocess.run(['playwright', '--version'], check=True, capture_output=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, PlaywrightError):
        return False
    except Exception as e:
        # Catch unexpected errors during check
        console.print(f"[yellow]Warning:[/yellow] Could not reliably check Playwright installation: {e}")
        return True # Assume installed if check fails unexpectedly

@click.command()
@click.option('--prompt', default='Extract the main content and summarize it.', help='The prompt for the AI to extract information.')
@click.option('--sitemap-depth', type=int, default=1, help='Depth to crawl for sitemap generation (0=only initial page, 1=initial + links, etc.).')
def main(prompt, sitemap_depth):
    """A CLI tool to scrape a website and extract information using Gemini, Groq, Ollama, or OpenAI."""
    # --- Check Playwright Installation --- 
    if not check_playwright_installation():
        console.print(Panel(
            "[bold yellow]Playwright browsers not detected![/bold yellow]\n\n" 
            "This tool requires Playwright browsers to function correctly.\n"
            "Please run the following command in your terminal to install them:\n\n"
            "  [bold cyan]playwright install --with-deps[/bold cyan]\n\n"
            "After installation, please re-run this tool.",
            title="[bold red]Action Required[/bold red]",
            border_style="red",
            expand=False
        ))
        sys.exit(1) # Exit if playwright check fails
    """A CLI tool to scrape a website and extract information using Gemini, Groq, Ollama, or OpenAI."""
    console.print(Panel("[bold cyan]Welcome to the AI Web Scraper CLI![/bold cyan]\nThis tool scrapes web pages and extracts information using an AI model.", title="AI Scraper", border_style="blue"))

    # --- API Selection --- 
    console.print("\nPlease choose an AI provider:")
    console.print("1. Gemini")
    console.print("2. Groq")
    console.print("3. Ollama")
    console.print("4. OpenAI")
    # Use IntPrompt for integer input with choices
    api_choice = IntPrompt.ask("Enter the number of your choice", choices=['1', '2', '3', '4'], show_choices=False)

    api_map = {1: 'gemini', 2: 'groq', 3: 'ollama', 4: 'openai'}
    api = api_map[api_choice]

    # --- API Key Handling --- 
    api_key = None
    client = None
    env_var_name = f"{api.upper()}_API_KEY"

    if api != 'ollama': # Ollama might not need a key in the same way
        api_key = get_api_key(api)
        if not api_key:
            # Use Prompt.ask for password input
            api_key = Prompt.ask(f'Enter your {api.capitalize()} API key', password=True)
            # Save the API key to the user config .env file using the new function
            save_api_key_to_env(api, api_key)
            # The key is loaded into the environment by save_api_key_to_env, no need to use it directly here
    else:
        # Handle Ollama connection check - get_ollama_client does basic check
        pass # No key needed typically, client setup handles connection

    # --- URL Input --- 
    url = Prompt.ask('Enter the URL to scrape')
    console.print(f"\n:robot: Using [bold green]{api.capitalize()}[/bold green] API to scrape: [link={url}]{url}[/link]")

    # Scrape the website with Progress
    scraped_content = None
    sitemap_data = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Scraping website & generating sitemap...", total=None)
        # Pass sitemap_depth to the scraper function
        scraped_content, sitemap_data = asyncio.run(scrape_website(url, sitemap_depth=sitemap_depth))

    if not scraped_content:
        console.print("[bold red]Error:[/bold red] Failed to scrape the website. Exiting.")
        return

    console.print(":white_check_mark: Scraping complete.")

    extracted_data = None
    client = None

    # --- Initialize Client --- 
    # (Keep client initialization logic as is, just update print statements)
    try:
        if api == 'gemini':
            client = get_gemini_client(api_key)
        elif api == 'groq':
            client = get_groq_client(api_key)
        elif api == 'openai':
            client = get_openai_client(api_key)
        elif api == 'ollama':
            client = get_ollama_client()

        if not client and api != 'ollama': # Ollama client check is slightly different
             raise ValueError(f"Failed to initialize {api.capitalize()} client. Check API key.")
        elif not client and api == 'ollama':
             raise ConnectionError("Failed to initialize Ollama client. Is the Ollama service running?")

    except (ValueError, ConnectionError) as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return
    except Exception as e: # Catch other potential init errors
        console.print(f"[bold red]Error:[/bold red] An unexpected error occurred during client initialization: {e}")
        return

    # --- Model Selection (if applicable) & Extraction --- 
    model_name = None
    if api == 'openai':
        # Example models: gpt-4, gpt-3.5-turbo
        model_name = Prompt.ask('Enter the OpenAI model name to use', default='gpt-3.5-turbo')
    elif api == 'ollama':
        # Example models: llama3, mistral
        model_name = Prompt.ask('Enter the Ollama model name to use', default='llama3')
        # You could potentially list available Ollama models here if the client supports it easily

    # AI Extraction with Progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description=f"Extracting information using {api.capitalize()}...", total=None)
        try:
            if api == 'gemini':
                extracted_data = extract_with_gemini(client, scraped_content, prompt)
            elif api == 'groq':
                extracted_data = extract_with_groq(client, scraped_content, prompt)
            elif api == 'openai':
                extracted_data = extract_with_openai(client, scraped_content, prompt, model=model_name)
            elif api == 'ollama':
                extracted_data = extract_with_ollama(client, scraped_content, prompt, model=model_name)
        except Exception as e:
            console.print(f"\n[bold red]Error during AI extraction:[/bold red] {e}")
            extracted_data = None # Ensure extracted_data is None on error

    if extracted_data:
        console.print(Panel(extracted_data, title="Extracted Information", border_style="green", expand=False))
    else:
        console.print("\n[bold red]Error:[/bold red] Failed to extract information using the AI.")

    # --- Save Sitemap --- 
    if sitemap_data:
        save_sitemap(sitemap_data, SITEMAP_FILENAME)
        console.print(f":world_map: Sitemap saved to [bold cyan]{SITEMAP_FILENAME}[/bold cyan]")
    else:
        console.print("[yellow]Warning:[/yellow] Sitemap could not be generated or was empty.")

if __name__ == '__main__':
    main()