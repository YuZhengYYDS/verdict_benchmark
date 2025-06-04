from setuptools import setup, find_packages

setup(
    name='verdict_benchmark',
    version='0.1',
    packages=find_packages(),  # 会自动把 utils、data（如果 data 下有 __init__.py）等当成 package
)

# Usage:
# the terminal must be like /…/verdict_benchmark/
# Then DO: <code> 
#               pip install -e .
#          <code> 