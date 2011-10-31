from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
  name="map_generate",
  version="0.2",
  author="Tom Temple",
  author_email="tom.temple@vecna.com",
  install_requires = ["skimage"],
  dependency_links = ["http://github.com/scikits-image/scikits-image"],
  entry_points = {
    'console_scripts' : ['map_generate = map_generate.map_generate:main']},
  packages=['map_generate'],
  long_description=read('README')
)
