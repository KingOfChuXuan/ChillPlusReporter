# Example data

`195K.pdb` is a 4096-water configuration used as a regression case for the
CHILL+ classifier.  It is read by `examples/analyze_pdb.py` and by
`tests/test_example_data.py`.

Reference result with the default parameters (`cut_off=3.5` angstrom,
`l=3`, `cluster_cut_off=3.2` angstrom):

```text
Ic                       1109
Ih                        544
Interfacial Ice          1047
Clathrate                   5
Interfacial Clathrate      62
Other                    1329
Largest ice cluster      1561
```

The file contains oxygen atoms and additional coarse-grained sites; only atoms
named `O` (or with element oxygen) are analysed.
