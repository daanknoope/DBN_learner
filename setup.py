from setuptools import setup, find_packages

setup(name='DBN_learner',
      version='0.1',
      description='Learning DBNs in Python with Gobnilp',
      author='Daan Knoope',
      author_email='daan@knoope.dev',
      license='MIT',
      packages=find_packages(), install_requires=['pandas', 'gobnilp'],
      python_requires='>3.6'
      )
