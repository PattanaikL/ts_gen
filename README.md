# ts_gen
Generate 3D transition state geometries with GNNs (Note: python3 pytorch version and integration into [ARC](https://github.com/ReactionMechanismGenerator/ARC) coming soon!)

## Requirements
* python (version=2.7)
* tensorflow (version=1.14)
* rdkit (version=2018.09.3)

## Installation
`git clone https://github.com/PattanaikL/ts_gen`

## Usage
To train the model, call the `train.py` script with the following parameters defined. If training with your own data, ensure data is in sdf format and molecules between reactants, products, and transition states are all aligned.

`python train.py --r_file data/intra_rxns_reactants.sdf --p_file data/intra_rxns_products.sdf --ts_file data/intra_rxns_ts.sdf`

To evaluate the trained model, refer to `use_trained_model.ipynb`

## Data structures
Currently, we support sdf integration through rdkit, but all that's required is an rdkit mol. If you have data in xyz format, consider using the [code](https://github.com/jensengroup/xyz2mol) from the Jensen group to convert to rdkit.
