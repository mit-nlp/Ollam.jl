OLLAM: Online Learning of Linear Adaptatable Models
===================================================

<img align=right src="https://travis-ci.org/mit-nlp/Ollam.jl.svg?branch=master" alt="Build Status"/>

This package is an implementation of:

1. MIRA and PA algorithms (Krammer and Singer 2003)
2. Averaged Perceptron (Collins 2001)

It is also contains a LIBSVM wrapper for linear models (mainly for
comparison and initialization).

Prerequistes
------------

- `Stage.jl` - Needed for logging and memoization *(Note: requires manual install)*
- `LIBSVM.jl` - LibSVM binaries and julia wrapper
- `MNIST.jl` - for testing

Install
-------

This is an experimental package which is not currently registered in
the julia central repository.  You can install via:

```julia
Pkg.clone("https://github.com/saltpork/Stage.jl")
Pkg.clone("https://github.com/mit-nlp/Ollam.jl")
```

This process should install all dependent packages in addition to `Ollam`.

Usage
-----

See `test/runtests.jl` for detailed usage.

License
-------
This package was created for the DARPA XDATA program under an Apache v2 License.

