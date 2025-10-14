from setuptools import setup, find_packages

setup(
    name="music_analytics_app",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit==1.35.0",
        "prefect==2.13.7",
        "pdfplumber==0.9.0",
        "fpdf==1.7.2",
        "pandas==2.2.2",
        "matplotlib==3.8.0",
        "seaborn==0.13.2",
        "python-dotenv==1.0.1",
    ],
    author="Aniruddha Partha",
    description="Modular analytics app for music report PDF pipelines, DTC/IP analytics, and dashboard visualization.",
    url="https://github.com/your-repo-url",
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)
