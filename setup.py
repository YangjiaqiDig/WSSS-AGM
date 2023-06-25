from setuptools import setup

# with open("requirements.txt") as requirement_file:
#     requirements = requirement_file.read().split()


setup(
   name='project_retina_wsss',
   version='1.0.0',
   description='retinal wsss packages.',
   author='Jiaqi Yang',
   author_email='jyang@gradcenter.cuny.edu',
   url='https://github.com/YangjiaqiDig/OCT_Retinal_Project',
   packages=['gan_and_str', 'utils', 'network'],  #same as name
#    install_requires=requirements, #external packages as dependencies
   zip_safe=False,
)