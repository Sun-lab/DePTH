import setuptools
from DePTH import __version__

with open("DESCRIPTION", "r", encoding = "utf-8") as fh:
    long_description = fh.read()


seeds_list = []

with open("DePTH/data/ensemble_seeds_20.txt") as fp:

    Lines = fp.readlines()

    for line in Lines:

        line_split = line.split("\t")
        seed_1 = line_split[0]
        seed_2 = line_split[1]
        seed_3 = line_split[2].split("\n")[0]

        seeds_list += [[seed_1, seed_2, seed_3]]


data_file_list = ['data/ensemble_seeds_20.txt', \
                  'data/for_encoders/*.csv', \
                  'data/for_encoders/*.tsv']

for cur_seeds in seeds_list:
    for core_name in ["HLA_I_all_match", "HLA_II_all_match"]:
        full_name = core_name+"_model_"+"_".join(cur_seeds)
        data_file_list += \
             ['data/trained_models_legacy/'+core_name+'/'+full_name+'/assets/*', \
              'data/trained_models_legacy/'+core_name+'/'+full_name+'/fingerprint.pb', \
              'data/trained_models_legacy/'+core_name+'/'+full_name+'/keras_metadata.pb', \
              'data/trained_models_legacy/'+core_name+'/'+full_name+'/saved_model.pb', \
              'data/trained_models_legacy/'+core_name+'/'+full_name+'/variables/*']

for cur_seeds in seeds_list:
    for core_name in ["HLA_I", "HLA_II"]:
        full_name = "model_"+"_".join(cur_seeds)
        data_file_list += \
             ['data/trained_models/'+core_name+'/'+full_name+'/assets/*', \
              'data/trained_models/'+core_name+'/'+full_name+'/fingerprint.pb', \
              'data/trained_models/'+core_name+'/'+full_name+'/keras_metadata.pb', \
              'data/trained_models/'+core_name+'/'+full_name+'/saved_model.pb', \
              'data/trained_models/'+core_name+'/'+full_name+'/variables/*']


setuptools.setup(
    name = "DePTH",
    version = __version__,
    author="Si Liu",
    author_email="liusi2019@gmail.com",
    description = "DePTH provides neural network models for sequence-based TCR and HLA association prediction",
    long_description = long_description,
    long_description_content_type = "text/plain",
    url = "https://github.com/Sun-lab/DePTH",
    project_urls = {
        "Documentation": "https://github.com/Sun-lab/DePTH",
        "Bug Tracker": "https://github.com/Sun-lab/DePTH/issues",
    },
    license='MIT',
    entry_points={
        "console_scripts": ["DePTH=DePTH.main:main"]
        },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    #packages = setuptools.find_packages(where="DePTH"),
    #packages = ["DePTH"],
    packages = setuptools.find_packages(),
    include_package_data=True,
    package_data={'DePTH': data_file_list},
    python_requires = ">=3.9",
    install_requires=[
        'scikit-learn>=1.0.2',
        'tensorflow>=2.4.1,<2.15.1',
        'pandas>=1.4.2',
        'numpy>=1.21.5',
        ]
)
