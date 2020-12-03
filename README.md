# Group-Transformer: Scale-down Transformers by Grouping Features 

Official Pytorch implementation of Group-Transformer that adapts group-wise computations rather than reduces feature dimension or network depth. Please refer to the paper, "Scale down Transformer by Grouping Features for a Lightweight Character-level Language Model (COLING-2020)", for more details. 
  
## Software requirement

* This work has been done with PyTorch 0.4.1, CUDA 9.0, python 3.6 and Ubuntu 16.04.
```
pip3 install torch==0.4.1
```

## How to run the trained model

* Download enwik8 dataset
```
sh download_enwik8.sh
```
* Train Group-Transformer
```
sh enwik_model_train.sh
```
Check the parameters and options in the file.

## Contact

Feel free to contact me if there is any question (Sungrae Park sungrae.park@navercorp.com).

## Acknowledgement

This repository contains the code originally forked from the [transformer-xl](https://github.com/kimiyoung/transformer-xl). 

## License

```
Copyright 2019-present NAVER Corp.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
