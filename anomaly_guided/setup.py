from setuptools import setup, find_packages

setup(
   name='agm',
   version='0.1',
   description='Anomaly Guided WSSS on OCT Retinal Images',
   author='Jiaqi Yang',
   author_email='jyang@gradcenter.cuny.edu',
   url='https://github.com/YangjiaqiDig/WSSS-AGM/tree/master/anomaly-guided',
   packages=find_packages(),
   python_requires='>=3.6',
   zip_safe=False,
)