[![Build Status](https://travis-ci.com/zhaofeng-shu33/info-detection-experiment.svg?branch=master)](https://travis-ci.com/zhaofeng-shu33/info-detection-experiment)

# Overview

This experiment focuses on comparing the detection results of info-detection with other methods on artificial and real-world dataset.

To replicate the experiment, you should use `python3` and follow steps below:
1. `pip3 install --user -r requirements.txt`
1. run `python3 demo.py` to generate the boundary curve figure.
1. run `mkdir -p build && cp parameter.json build/ && python3 schema.py` to generate result LaTeX table in build directory.
We use `sacred` library to organize our experiment. If proper environment variable is set, the result can be saved to mongo database for further reference.

Result:

![Figure](./outlier_boundary_illustration.svg)

## How to tune the parameter

To tune the hyper parameters of different methods, there are several preliminaries needed to be done:

1. Create a file called `conf.yaml` from `conf-sample.yaml`. 

2. Set up `mongodb` database, export `USE_MONGO=1` environment variables.

3. Create a new database `sacred` in `mongodb` with root user. Set up a user in `user-data` authentication database with username `admin` and password `abc`. Grant `ReadWrite` privileges to the `admin` user.

After finishing the above preliminaries. You can modify `conf.yaml` before running `evaluation.py`. Then each experiment record is written to the database. Also, [omniboard](https://github.com/vivekratnavel/omniboard) is recommended to visualize the experiment results from multiple run.

