# Multiclass_classification_schemes
Multiclass schemes for classiﬁcation of handwritten digits

Extract the ﬁles in libsvm-3.14.zip somewhere in your home directory. 
 
Start Matlab and use addpath to add the directory where you have extracted LIBSVM to the MATLAB search path (use savepath in order to add it permanently). 

Go to the subfolder matlab and type make in the Matlab prompt. (Pre-built binary ﬁles for Windows 64bit already contained in the folder).

The matlab function getKernelSVMSolution provides a nice interface to the LIBSVM pack- age (use help getKernelSVMSolution to see how it works).
 
The problem deals with the classiﬁcation of handwritten digits (10 classes). You are supposed to use the SVM with the Gaussian kernel: 
 
 k(x,y) = e−λkx−yk2
 
The training and test data is in USPSTrain.mat and USPSTest.mat. The 16×16-images of the digits are represented as 256-dimensional column vectors. Write two matlab scripts:
 
one which solves the multi-class problem using one-versus-all (save it in OneVersusAll.m)

one which solves the multi-class problem using one-versus-one (save it in OneVersusOne.m)

In both cases use C = 100 and λ = 3 γ, where γ is the median of all squared distances between training points, as parameters for the binary SVM. 

Visually inspect the digits which have been misclassiﬁed.
