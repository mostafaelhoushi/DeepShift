# Simple Python Fixed-Point Module (SPFPM)

spfpm is a pure-Python toolkit for performing binary fixed-point arithmetic,
including trigonometric and exponential functions.

The package provides:
* Representations of values with a fixed number of fractional bits
* Optional constraints on the number of whole-number bits
* Interconversion between native Python types and fixed-point objects
* Arithmetic operations (addition, subtraction, multiplication, division)
  of fixed-point numbers
* Methods for computing powers, logarithms and exponents
* Methods for computing trigonometric functions and their inverses
* Computation of various mathematical constants, such as pi and log(2),
  to maximal precision for the chosen fixed-point resolution
* Printing fixed-point numbers as decimal numbers
* Support for numbers with thousands of bits of resolution

On a modern desktop PC, spfpm is typically capable
of hundreds of thousands of arithmetic operations per second,
i.e. over 100 kilo-FLOPS, even for a few hundred bits of resolution.

Development currently targets [Python](https://www.python.org)
versions 3.2 and later, although the library should also
work with python-2.7.
The latest version of spfpm can be found
on [GitHub](https://github.com/rwpenney/spfpm),
with earlier versions also available
on [SourceForge](https://sourceforge.net/projects/pyfixedpoint/).


## Examples

After installation there are two main classes that you need
from the FixedPoint module:

```python
from FixedPoint import FXfamily, FXnum
```

you can create fixed-point numbers with the default (64-bit) resolution
as follows:

```python
x = FXnum(22) / FXnum(7)
y = FXnum(3.1415)
print(x - y)
```

Creating numbers with a specific precision requires use of the FXfamily class:

```python
fam100 = FXfamily(100)
z = FXnum(1, fam100)
z2 = fam100(2)
```

One can then apply various computations such as:

```python
print(z.atan() * 4)
print(z2.sqrt())
```

The FXfamily class also provides access to pre-computed constants
which should be accurate to at least 1/2 of the least significant bit (LSB):

```python
print('pi = ', fam100.pi)
print('log2 = ', fam100.log2)
print('sqrt2 = ', fam100.sqrt2)
```

This produces the following printed values of those constants:

* 3.141592653589793238462643383279 ~~3~~
* 0.69314718055994530941723212145 ~~79~~
* 1.414213562373095048801688724209 ~~1~~

The struck-through digits show where the values computed
at 100-bit precision differ from the true values of these constants.
Alternatively, one could print these values in base-16:

```python
print(FXfamily(400).pi.toBinaryString(logBase=4))
```

giving a hexadecimal value of Pi as:

* 3.243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c89452821e638d01377be5466cf34e90c6cc0ac

which agrees exactly with
the [accepted result](http://hexpi.sourceforge.net/).


## Licensing

All files are released under
the [Python PSF License](https://docs.python.org/3/license.html)
and are Copyright 2006-2018 RW Penney.
