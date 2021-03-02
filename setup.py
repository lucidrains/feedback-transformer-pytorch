from setuptools import setup, find_packages

setup(
    name = 'feedback-transformer-pytorch',
    packages = find_packages(),
    version = '0.0.11',
    license='MIT',
    description = 'Implementation of Feedback Transformer in Pytorch',
    author = 'Phil Wang',
    author_email = 'lucidrains@gmail.com',
    url = 'https://github.com/lucidrains/feedback-transformer-pytorch',
    keywords = [
        'attention',
        'artificial intelligence',
        'transformer',
        'deep learning',
        'memory'
    ],
    install_requires=[
        'torch>=1.6',
        'einops'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)