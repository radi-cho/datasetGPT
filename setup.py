from distutils.core import setup

setup(
    name="datasetGPT",
    packages=["datasetGPT"],
    version="0.0.1",
    description="Generate textual and conversational datasets with LLMs.",
    author="Radostin Cholakov",
    author_email="radicho123@gmail.com",
    url="https://github.com/radi-cho/datasetGPT",
    # download_url="https://github.com/radi-cho/datasetGPT/archive/v0.0.1.tar.gz",
    keywords=["dataset", "llm", "langchain", "openai"],
    install_requires=[
        "langchain",
        "openai",
        "click"
    ],
    entry_points={
        "console_scripts": [
            "datasetGPT=datasetGPT:datasetGPT"
        ],
    },
)
