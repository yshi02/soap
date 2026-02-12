Overview:

These files represent the CFAR/Detection functionality of the JHU/APL Reference Radar Signal Processor. These files are being provided to the performers if they would like to incorporate the common functions into their radar processing pipelines. The code is being provided as-is with limited to no technical support.

Getting Started:

The “get_detection_report_SOAP.m” function accepts a radar-Doppler matrix and its metadata, along with a structure of signal processing parameters, and produces the detection report. Contained within the function are calls to CFAR, clustering, and range/Doppler measurement refinement. Note that the specific processing parameters used by the Reference Signal processor are included in the comments of this function.
