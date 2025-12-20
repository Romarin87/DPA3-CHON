# General reactive machine learning potentials for CHON elements

This work presents a scalable workflow for developing general and reliable reactive machine learning potentials for systems containing C, H, O, and N elements, achieving high accuracy and transferability for large-scale chemical simulations.

-----

➡️ **[Read the full paper on ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/684ffe583ba0887c33dad39b)**

-----

## How to Cite

If you use this work, please cite the following publication:

```
@article{li2025general,
  title={General reactive machine learning potentials for CHON elements},
  author={Li, Bowen and Mi, Sixuan and Xiao, Jin and Duo, Zhang and Shuwen, Zhang and Zhang, John and Wang, Han and Zhu, Tong},
  year={2025}
}
```

-----

## DPA-3 Environment Setup and Model Usage

This document outlines the steps to set up the `deepmd-kit` environment and use the DPA-3 models. This guide is based on `deepmd-kit-3.1.0`.

### 1\. Environment Setup

For detailed installation instructions, please refer to the official `deepmd-kit` v3.1.0 release page: [https://github.com/deepmodeling/deepmd-kit/releases/tag/v3.1.0](https://github.com/deepmodeling/deepmd-kit/releases/tag/v3.1.0).

#### Environment Activation and Configuration

1.  After installation, activate the environment. By default, it is located in the `/home/xxx/deepmd-kit` directory.

    ```bash
    # Note: 'xxx' should be replaced with your actual user directory.
    source activate /home/xxx/deepmd-kit
    ```

2.  The default environment does not include the `ASE` and `xtb-python` packages, which must be installed manually.

    ```bash
    # Install ase
    pip install ase

    # Install xtb-python from the conda-forge channel
    conda install xtb-python -c conda-forge
    ```

### 2\. Model Usage

There are two DPA-3 models available under the DFT tag: `DPA-3-F@DFT.pt` and `DPA-3-DF@DFT.pt`.

#### 2.1. DPA-3-F@DFT.pt (Directly Trained Model)

This model is trained directly and can be used in Python through the ASE interface.

```python
from deepmd.pt.utils.ase_calc import DPCalculator
from ase.io import read

# Initialize the calculator
calc = DPCalculator(model="DPA-3-F@DFT.pt", device='cuda')

# The remaining steps are consistent with other ASE calculators
data = read('test.xyz')
data.calc = calc

# Example calculations
energy = data.get_potential_energy()
forces = data.get_forces()
```

#### 2.2. DPA-3-DF@DFT.pt ($\\Delta$-Learning Model)

This is a $\\Delta$-Learning model, and its use requires combination with GFN2-xTB calculations. To use it with the ASE interface in Python, a helper script (`deepmd_xtb.py`) is needed.

```python
import sys

# Add the directory containing 'deepmd_xtb.py' to the Python path.
# This example assumes the script is in '/home/xxx/ase_interface'.
sys.path.append('/home/xxx/ase_interface')

from deepmd_xtb import DP_XTB

# Initialize the calculator
calc = DP_XTB(model="DPA-3-DF@DFT.pt", device='cuda')
```


"Please note: The relevant model files will be released here after our paper is formally accepted."
