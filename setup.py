from setuptools import find_packages, setup

setup(name='cc_rl',
      version='0.1.0.dev',
      author="Alicia Fortes Machado, Aloysio Galvão Lopes, Iago Martinelli Lopes, Igor Albuquerque Silva",
      description="Reinforcement learning to improve classifier chain performance",
      long_description=open("README.md", "r", encoding="utf-8").read(),
      long_description_content_type="text/markdown",
      keywords="classifier chains deep learning gym openai reinforcement learning pytorch tensorflow",
      package_dir={"": "src"},
      packages=find_packages("src"),
      python_requires='>=3.6'
)
