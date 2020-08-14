Dear all,
Last Friday I had a meeting with Amik and Gerard and showed the latest results to them. Their feedback was the following:
RTM

We are in the right direction, but still need to include more sources on the right hand side of the domain. For some reason the sources are distributed from 0 to a little less than 8 km (see figure after input [8]). Could you please extend to the end of the domain (12 km) and run again?
FWI

The divergence we observe in the last figure is due to the inadequate setup of the artificial step length, alpha, for gradient descent (see input [16]). Gerard recommended we employed L-BFGS-B to determine alpha, and take a look on the Scipy optimizer fuctions. He forwarded the following links to help us to do that.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
https://github.com/dask/dask-tutorial
https://github.com/opesci/devito/blob/master/examples/seismic/tutorials/04_dask.ipynb
Post-processing

Amik and Gerard suggested we should filter out the high-intensity contours on the top of the domain. Gerard said that simply masking out the "water" part should be enough.
I will have another meeting with them on Thursday at noon. Can you please keep working on that, following these pieces of advice, and send me updated versions of the report by Wednesday?
Best regards,
Bruno.

