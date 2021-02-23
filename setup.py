# Authors: Leon Aksman <l.aksman@ucl.ac.uk>
# License: TBC


from setuptools import setup

#parse the requirement.txt file, ignoring commented lines, placing results in install_reqs
with open('requirements.txt', 'r') as f:
    install_reqs = [
        s for s in [
            line.strip(' \n') for line in f
        ] if not s.startswith('#') and s != ''
    ]


print("Started pySuStaIn setup.py")

setup(name=					'pySuStaIn',
      version=				'0.1',
      description=			'Python implementation of the SuStaIn algorithm',
      url=					'https://github.com/ucl-pond/pySuStaIn',
      classifiers=			['Intended Audience :: Science/Research',
                   			'Programming Language :: Python',
                   			'Topic :: Scientific/Engineering',
                   			'Programming Language :: Python :: 3.7'],
      maintainer=			'Leon Aksman',
      maintainer_email=		'l.aksman@ucl.ac.uk',
      license=				'TBC',
      packages=				['pySuStaIn', 'mixture_model', 'plotting', 'sim'],
      python_requires= 		'>=3.7',
      install_requires = 	install_reqs,	#the parsed requirements from requirements.txt
      entry_points=			{},
      zip_safe=				False)

print("Finished pySuStaIn setup.py")