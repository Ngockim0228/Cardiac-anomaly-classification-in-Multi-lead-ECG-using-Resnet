# Cardiac-anomaly-classification-in-Multi-lead-ECG-using-Resnet
The electrocardiogram (ECG) is a well-known, non-invasive method for detecting cardiac abnormalities. Resnet has shown promising results in terms of how to improve the accuracy of ECG signal classification.

The model receives an input tensor with dimension (N, 4096, 12), and returns an output tensor with dimension (N, 6), for which N is the batch size.

input: shape = (N, 4096, 12). The input tensor should contain the 4096 points of the ECG tracings sampled at 400Hz (i.e., a signal of approximately 10 seconds). Both in the training and in the test set, when the signal was not long enough, we filled the signal with zeros, so 4096 points were attained. The last dimension of the tensor contains points of the 12 different leads. 
The leads are ordered in the following order: {DI, DII, DIII, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6}. All signal are represented as 32 bits floating point numbers at the scale 1e-4V: so if the signal is in V it should be multiplied by 1000 before feeding it to the neural network model.

output: shape = (N, 6). Each entry contains a probability between 0 and 1, and can be understood as the probability of a given abnormality to be present. The abnormalities it predicts are (in that order): 1st degree AV block(1dAVb), right bundle branch block (RBBB), left bundle branch block (LBBB), sinus bradycardia (SB), atrial fibrillation (AF), sinus tachycardia (ST). 
The abnormalities are not mutually exclusive, so the probabilities do not necessarily sum to one.
