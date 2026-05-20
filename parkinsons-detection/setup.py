from setuptools import setup, find_packages

setup(
    name="parkinsons-detection",
    version="1.0.0",
    description="AI-Based Early Parkinson's Detection Using Voice and Touch",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.13.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "librosa>=0.10.0",
        "flask>=2.3.0",
        "scipy>=1.11.0",
    ],
    python_requires=">=3.9",
)
