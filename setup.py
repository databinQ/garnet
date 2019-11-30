# coding: utf-8
# @Author: garnet
# @Time: 2019/11/30 22:25

import codecs
from setuptools import setup, find_packages

with codecs.open("requirements.txt", "r") as f:
    install_requires = [t.strip() for t in f.readlines()]

setup(
    name="garnet",
    version="0.0.1",
    description="Toolkit for machine learning and NLP tasks",
    keywords=["garnet", "NLP", "machine learning", "deep learning"],
    author="No Looking Pass Dee",
    author_email="garnetreds7@163.com",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False
)
