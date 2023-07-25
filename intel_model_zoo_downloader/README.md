# OpenVINO™ Model Downloader 
###Intel Model Zoo - AI Library Component

This directory contains scripts that automate certain model-related tasks
based on configuration files in the models' directories.

* `downloader.py` (model downloader) downloads model files from online sources
  and, if necessary, patches them to make them more usable with Model
  Optimizer;

# [OpenVINO™ Toolkit](https://01.org/openvinotoolkit) - Open Model Zoo repository
[![Stable release](https://img.shields.io/badge/version-2021.3-green.svg)](https://github.com/openvinotoolkit/open_model_zoo/releases/tag/2021.3)
[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/open_model_zoo/community)
[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)

This repository includes optimized deep learning models and a set of demos to expedite development of high-performance deep learning inference applications. Use these free pre-trained models instead of training your own models to speed-up the development and production deployment process.

## Repository Components:
* [Pre-Trained Models](models/intel/index.md)
* [Public Models Description](models/public/index.md)
* [Model Downloader](tools/downloader/README.md) and other automation tools
* [Demos](demos/README.md) that demonstrate models usage with Deep Learning Deployment Toolkit
* [Accuracy Checker](tools/accuracy_checker/README.md) tool for models accuracy validation

## License
Open Model Zoo is licensed under [Apache License Version 2.0](LICENSE).

## Documentation
* [OpenVINO™ Release Notes](https://software.intel.com/en-us/articles/OpenVINO-RelNotes)
* [Pre-Trained Models](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models)
* [Demos and samples](https://software.intel.com/en-us/articles/OpenVINO-IE-Samples)

## Other usage examples
* [Open Visual Cloud](https://01.org/openvisualcloud)
  * [Tutorial: Build and Run the AD Insertion Sample on public cloud or local machine](https://01.org/openvisualcloud/documents/tutorial-build-and-run-ad-insertion-sample-public-cloud-or-local-machine)
  * [GitHub Repo for Ad Insertion Sample](https://github.com/OpenVisualCloud/Ad-Insertion-Sample)
* [OpenVINO for Smart City](https://github.com/incluit/OpenVino-For-SmartCity)
* [OpenVINO Driver Behavior](https://github.com/incluit/OpenVino-Driver-Behaviour)

## How to Contribute
We welcome community contributions to the Open Model Zoo repository. If you have an idea how to improve the product, please share it with us doing the following steps:
* Make sure you can build the product and run all the demos with your patch.
* In case of a larger feature, provide a relevant demo.
* Submit a pull request at https://github.com/openvinotoolkit/open_model_zoo/pulls

You can find additional information about model contribution [here](CONTRIBUTING.md).

We will review your contribution and, if any additional fixes or modifications are needed, may give you feedback to guide you. When accepted, your pull request will be merged into the GitHub* repositories.

Open Model Zoo is licensed under Apache License, Version 2.0. By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

## Support
Please report questions, issues and suggestions using:
* [\#open_model_zoo](https://stackoverflow.com/search?q=%23open_model_zoo) tag on StackOverflow*
* [GitHub* Issues](https://github.com/openvinotoolkit/open_model_zoo/issues)
* [Forum](https://software.intel.com/en-us/forums/intel-distribution-of-openvino-toolkit)
* [Gitter](https://gitter.im/open_model_zoo/community)

---
\* Other names and brands may be claimed as the property of others.

## Prerequisites

1. Install Python (version 3.6 or higher)
2. Install the tools' dependencies with the following command:

```sh
python3 -mpip install --user -r ./requirements.in
```

For the model converter, you will also need to install the OpenVINO&trade;
toolkit and the prerequisite libraries for Model Optimizer. See the
[OpenVINO toolkit documentation](https://docs.openvinotoolkit.org/) for details.

To convert models from certain frameworks, you will also need to install
additional dependencies.

For models from Caffe2:

```sh
python3 -mpip install --user -r ./requirements-caffe2.in
```

For models from PyTorch:

```sh
python3 -mpip install --user -r ./requirements-pytorch.in
```

For models from TensorFlow:

```sh
python3 -mpip install --user -r ./requirements-tensorflow.in
```

## Model downloader usage

The basic usage is to run the script like this:

```sh
./downloader.py
```

This will download all models. The `--all` option can be replaced with
other filter options to download only a subset of models. See the "Shared options"
section.

By default, the script will download models into a directory tree rooted
in the current directory. To download into a different directory, use
the `-o`/`--output_dir` option:

```sh
./downloader.py --output_dir my/download/directory
```

You may use `--precisions` flag to specify comma separated precisions of weights
to be downloaded.

```sh
./downloader.py --name face-detection-retail-0004 
```


__________

OpenVINO is a trademark of Intel Corporation or its subsidiaries in the U.S.
and/or other countries.


Copyright &copy; 2018-2019 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
