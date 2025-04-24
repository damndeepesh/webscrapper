from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read the requirements from requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='ai_web_scraper_cli',
    version='0.1.0',
    author='Deepesh', # Replace with your name
    author_email='damndeepesh@tutanota.com', # Replace with your email
    description='A CLI tool to scrape websites and extract info using Gemini or Groq.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/damndeepesh/ai-web-scraper-cli', # Replace with your repo URL
    project_urls={
        'Bug Tracker': 'https://github.com/damndeepesh/ai-web-scraper-cli/issues', # Replace
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Choose your license
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Internet :: WWW/HTTP :: Indexing/Search',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing',
    ],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.7', # Specify your minimum Python version
    install_requires=required,
    entry_points={
        'console_scripts': [
            'ai-scrape=cli:main', # This creates the command 'ai-scrape'
        ],
    },
)