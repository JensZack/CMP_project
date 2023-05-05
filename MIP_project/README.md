# MIP project

A finite element solver for a torus mesh

## Description

This module is intended to be able to solve Laplace's equation on a 2D torus using a bi-quadratic finite element approach.
In this module there is an auto-generating documentation and test-report functionality.

## Getting Started

To get started, you can run the main.py file to generate a torus mesh with an inner radius of 2, outer radius of 10, 15 nodes radially and 15 nodes circumferentially. This will plot the method of manufactured correct solution on top of the solution generated from the finite element mesh.

### Running Tests and Generating Documentation

* Running tests
The tests for this module can be run by running:
```
pytest test/test_solver.py
```
Once the tests have finished running, reports from the test_mms test will be saved in test/results

To add a mms test to the test_harness: add a gen_test function at the top of test/test_solvers.py with the appropriate MMS return:
```
def gen_mms_quad():
    """
    generate the MMS object for <my function>

    :return mms: MMS object
    """
    mms = MMS(
        lambda x, y: <my_function(x, y)>
        lambda x, y: <my_laplacian(x, y)>
        "function description for latex",
        "mms name"
    )
    return mms
```

Then add this function into the mms_list in test_mms, and all of the evaluation and auto-documentation of the test will be handled.

* Generating Docs

To generate a static html site with documentation for the GridTorus class and the test harness run:
```
cd doc
make html
```

To generate a pdf with documentation and test results from the test harness run:
```
cd doc
make latex

cd build/latex
make
```

This will generate mip_project.pdf in the doc/build/latex directory

