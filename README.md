# m3gnet

This project introduces all scripts for reading VASP calculations from vasprun.xml files and retraining m3gnet machine learning interatomic potential (ML-IAP).

The core technology behind this framework is based on a graph-like representation of data:

[https://doi.org/10.1038/s43588-022-00349-3](https://doi.org/10.1038/s43588-022-00349-3)
[https://doi.org/10.1063/1.5086167](https://doi.org/10.1063/1.5086167)

This technology is suitable for different applications: from analyzing defects in solid-solutions to estudying grain boundaries or interfaces in solar devices.

## Features

- Generation of graph database based on DFT simulations
- Retraining of the ML-IAP model
- Various validations of the retrained model

Please be aware that the code is under active development, bug reports are welcomed in the GitHub issues!

## Installation

To download the repository and install the dependencies:

```bash
git clone https://github.com/CibranLopez/m3gnet.git
cd m3gnet
pip3 install -r requirements.txt
```

## Execution

An user-friendly jupyter notebook has been developed, which can be run locally with pytorch dependencies. It generates a graph-like database (from the some local folder), trains and saves the model and analyzes the results.

## Authors

This project is being developed by:

 - Cibrán López Álvarez
 - ...

## Contact, questions and contributing

If you have questions, please don't hesitate to reach out at: cibran.lopez@upc.edu
