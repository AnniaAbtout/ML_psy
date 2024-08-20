from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='ML_psy',
      version="0.0.1",
      description="Psy Diagnostics based on EEG",
      license="ML_psy",
      author="ML_psy_team",
      author_email="",
      #url="https://github.com/AnniaAbtout/ML_psy",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
