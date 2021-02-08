from setuptools import find_packages, setup

print(find_packages())

setup(name='cc-rl', packages=find_packages(),
      version='0.1.0', python_requires='>=3.6')
