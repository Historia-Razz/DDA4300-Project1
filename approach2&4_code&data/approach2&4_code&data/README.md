# MAAIPM for Calculating Wasserstein Barycenter

This project is based on a simplified version of Jianbo Ye's package https://github.com/bobye/WBC_Matlab, which includes the propotype codes for computing the Wasserstein barycenter. See 
* Ye, J., Wu, P., Wang, J. Z., & Li, J. (2017). Fast discrete distribution clustering using Wasserstein barycenter with sparse support. IEEE Transactions on Signal Processing, 65(9), 2317-2332.
 

Besides the existing methods in Ye's package, this project contains MAAIPM for calculating Wasserstein Barycenter as well. The codes are released to reproduce the methods reported in the following paper:
*  Ge, D., Wang, H., Xiong, Z., & Ye, Y. (2019). Interior-point Methods Strike Back: Solving the Wasserstein Barycenter Problem. arXiv preprint arXiv:1905.12895.

This project contains the codes of all the mothods in the numerical experiment section. profile_centroid.m is the main comparison file and msllipses.m is an example of input format for the nested ellipses data.