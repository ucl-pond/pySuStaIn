# Authors: Leon Aksman <l.aksman@ucl.ac.uk>
# License: TBC


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('pySuStaIn')
    config.add_subpackage('mixture_model')

    return config

with open('requirements.txt', 'r') as f:
    install_reqs = [
        s for s in [
            line.strip(' \n') for line in f
        ] if not s.startswith('#') and s != ''
    ]

def setup_package():
    metadata = dict(name='pySuStaIn',
                    maintainer='Leon Aksman',
                    maintainer_email='l.aksman@ucl.ac.uk',
                    description='Python implementation of the SuStaIn algorithm',
                    license='TBC',
                    url='https://github.com/ucl-pond/pySuStaIn',
                    version='0.1',
                    zip_safe=False,
                    classifiers=['Intended Audience :: Science/Research',
                                 'Programming Language :: Python',
                                 'Topic :: Scientific/Engineering',
                                 'Programming Language :: Python :: 3.5',
                                 ],
              		install_requires=install_reqs
                    )

    from numpy.distutils.core import setup

    metadata['configuration'] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()