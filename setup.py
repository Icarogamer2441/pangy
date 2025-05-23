import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

VERSION = "0.1.0"

setuptools.setup(
    name="pangy-compiler", # Replace with your own username if distributing on PyPI
    version=VERSION,
    author="Your Name", # Replace with your name
    author_email="your.email@example.com", # Replace with your email
    description="A simple compiler for the Pangy programming language.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pangy",  # Replace with your project's URL
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Or your chosen license
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Topic :: Software Development :: Compilers",
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'pangy=pangy.cli:main',
        ],
    },
    # install_requires=[
    #    # Add any dependencies here, e.g., 'some_package>=1.0'
    # ],
) 