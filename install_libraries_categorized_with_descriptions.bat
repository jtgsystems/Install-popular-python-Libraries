@echo off

REM Data Science & Scientific Computing Libraries
echo Installing NumPy...
rem NumPy provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
pip install numpy
echo Installing Pandas...
rem Pandas is a powerful and versatile library for data manipulation and analysis. It provides high-performance, easy-to-use data structures and data analysis tools.
pip install pandas
echo Installing Matplotlib...
rem Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
pip install matplotlib
echo Installing SciPy...
rem SciPy builds on NumPy, extending its capabilities with functions for scientific and technical computing, including optimization, linear algebra, integration, interpolation, signal processing, and more.
pip install scipy
echo Installing scikit-learn...
rem scikit-learn is a widely used machine learning library providing various algorithms and tools for model training, evaluation, and deployment.
pip install scikit-learn
echo Installing TensorFlow...
rem TensorFlow is a powerful open-source library for numerical computation and large-scale machine learning, particularly known for its deep learning capabilities.
pip install tensorflow
echo Installing Keras...
rem Keras is a high-level API for building and training neural networks, often used with TensorFlow or other backends. It simplifies the process of creating and experimenting with deep learning models.
pip install keras
echo Installing PyTorch...
rem PyTorch is another popular deep learning framework known for its dynamic computation graph and ease of use, particularly in research settings.
pip install torch
echo Installing Statsmodels...
rem Statsmodels provides classes and functions for the estimation of many different statistical models, as well as for conducting statistical tests, and statistical data exploration.
pip install statsmodels

REM Web Development Libraries
echo Installing requests...
rem The requests library simplifies making HTTP requests in Python, providing a user-friendly interface for interacting with web services and APIs.
pip install requests
echo Installing Beautiful Soup...
rem Beautiful Soup is a library for parsing HTML and XML documents, making it easy to extract data from web pages.
pip install beautifulsoup4
echo Installing Scrapy...
rem Scrapy is a powerful and versatile framework for web scraping, allowing you to efficiently extract data from websites.
pip install scrapy

REM Image Processing & OCR Libraries
echo Installing pytesseract...
rem pytesseract is a wrapper for Tesseract OCR, enabling optical character recognition (OCR) capabilities in your Python applications.  Requires Tesseract to be installed separately.
pip install pytesseract

REM AI & Large Language Models Libraries
echo Installing openai...
rem The openai library provides an interface for interacting with OpenAI's powerful language models and APIs.
pip install openai
echo Installing anthropic...
rem The anthropic library allows interaction with Anthropic's Claude models, another leading large language model.
pip install anthropic

echo Installation complete.
pause
