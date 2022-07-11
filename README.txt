Noisier2Noise for self-supervised MRI reconstruction. This code was used to produce the results in
"A framework for self-supervised MR image reconstruction using sub-sampling via Noisier2Noise"
by Charles Millard and Mark Chiew [1].

The data and code from fastMRI [2] is required to run this code, which can be downloaded from
https://fastmri.org/ and https://github.com/facebookresearch/fastMRI/ respectively. The path of the data should be
given in config.yaml.

train_network.py trains a VarNet with configuration given in config.yaml. The model is saved in the logs folder.

test_network.py loads the model and evaluates the performance on the test set.

*** contact ***

If you have any questions/comments, please feel free to contact Charles
(Charlie) Millard at <charles.millard@ndcn.ox.ac.uk> or Mark Chiew at
<mark.chiew@ndcn.ox.ac.uk>

*** copyright and licensing ***

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

A copy of the GNU General Public License can be found in the file GNU_General_Public_License,
and is also availabe at <https://www.gnu.org/licenses/>

*** bibliography ***

[1] Millard, Charles, and Mark Chiew. "A framework for self-supervised MR image reconstruction using
    sub-sampling via Noisier2Noise." arXiv preprint arXiv:2205.10278 (2022)
[2] Zbontar, Jure, et al. "fastMRI: An open dataset and benchmarks for accelerated MRI." arXiv preprint
    arXiv:1811.08839 (2018).

