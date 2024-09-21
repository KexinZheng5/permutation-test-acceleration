# permutation-test-acceleration

Faster permutation based hypothesis test for neural networks with performance comparison and analysis between BANNs (Biologically Annoted Nerual Networks) and P-NET (Biologically informed deep neural network for prostate cancer classification and discovery)

## Usage

### Uploading source code of neural networks

Source code of the following models need to be added as the following directory:

| Model |  Path  | Source |
|:-----|:--------:|------:|
| BANNs   | `BANNs/_Source/BANNs` | [BANNs](https://github.com/lcrawlab/BANNs/tree/master/BANN) |
| mvBANNs   |  `BANNs/_Source/mvBANNs`  |   [GPU_Banns](https://github.com/sandraleebatista/GPU_Banns) |
| PNET   | `PNET/_Source` |    [pnet_prostate_paper](https://github.com/marakeby/pnet_prostate_paper) |

### Uploading data for each model

Data of the following models need to be added as the following directory:

| Dataset | Model |  Path  | Source |
|:----- |:-----|:--------:|------:|
| mice data | BANNs / mvBANNs   | `BANNs/_Data/mice` | [Data](https://www.dropbox.com/scl/fo/y5wb6qk36v8perehm6zar/AFBR4hdlJQy51J9e7jYYpwA/Data?dl=0&e=2&rlkey=6mjrm7fhgdiyoauu76xerytnu&subfolder_nav_tracking=1) |
| prostate cancer | PNET   | `PNET/_Source/_database` | [_database.zip](https://zenodo.org/records/5163213) |
