from setuptools import setup, find_packages

setup(
    name="regime_backtester",
    version="0.1.0",
    packages=find_packages(),  # will find the regime_backtester package
    install_requires=[
        "pandas>=1.3",
        "numpy>=1.20",
        "scikit-learn>=1.0",
        "matplotlib>=3.4",
        "yfinance>=0.1",
    ],
    entry_points={
        "console_scripts": [
            "regime-backtest=regime_backtester.cli:main",
        ],
    },
)
