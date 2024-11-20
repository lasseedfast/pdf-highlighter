from setuptools import setup, find_packages

setup(
    name='pdf-highlighter',
    version='0.1.0',
    packages=find_packages(),
    data_files=['prompts.yaml']
    install_requires=[
        'pymupdf',
        'nltk',
        'scikit-learn',
        'python-dotenv',
        'aiofiles',
        'pyyaml',
    ],
    entry_points={
        'console_scripts': [
            # Add any command-line scripts here
        ],
    },
    author='Lasse Edfast',
    author_email='lasse@edfast.se',
    description='A tool for annotating and highlighting sentences in PDF documents using an LLM.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/lasseedfast/pdf-highlighter',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)