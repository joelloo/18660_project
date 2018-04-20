# 18660_project

Implementation and comparison of robust PCA methods, with especial
focus on their utility in image cleaning (removal of specularities,
non-Lambertian effects) and background/foreground separation. Currently
implements:
* RPCA (using accelerated proximal gradient)
* RPCA (using exact ALM)
* RPCA (using inexact ALM)
* Fast PCP [Rodriguez,Wohlberg]

The following incremental robust PCA methods are also implemented:
* Online RPCA [Feng,Xu]
* iFrALM

Current dependencies:
* Numpy
* Scikit-learn
* Scikit-image
* Scipy
* Matplotlib
