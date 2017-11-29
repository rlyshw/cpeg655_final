This is the cpeg655 Final Project

Implement FFT in parallel with Cuda C.

Steps:

1. Implement parallel DFT
2. Figure out how to use butterfly to combine the DFTs into major FFT.
3. Parallel butterflies to finish parallel FFT.


Here's how to connect to the github repo on the CPEG655 machine:

run `export http_proxy=proxy.eecis.udel.edu:8888;export https_proxy=proxy.eecis.udel.edu:8888` in your terminal, then do your git operations.

OR, to have this take place automatically when you login, write 

```bash
export http_proxy=proxy.eecis.udel.edu:8888
export https_proxy=proxy.eecis.udel.edu:8888
```

to the end of your `.bash_profile`
