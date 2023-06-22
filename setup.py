from distutils.core import setup
from setuptools import find_packages

with open("README.md", "r", encoding = "utf-8") as readme:
    long_description = readme.read()

setup(
    name="datasetGPT",
    version="0.0.3",
    description="Generate textual and conversational datasets with LLMs.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    author="Radostin Cholakov",
    author_email="radicho123@gmail.com",
    url="https://github.com/radi-cho/datasetGPT",
    # download_url="https://github.com/radi-cho/datasetGPT/archive/v0.0.1.tar.gz",
    keywords=["dataset", "llm", "langchain", "openai"],
    package_dir={"": "src"},
    packages = find_packages(where="src"),
    install_requires=[
        "langchain>=0.0.113",
        "click>=8.1"
    ],
    entry_points={
        "console_scripts": [
            "datasetGPT=datasetGPT:datasetGPT"
        ],
    },
)
