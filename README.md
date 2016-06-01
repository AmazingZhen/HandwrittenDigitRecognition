# HandwrittenDigitRecognition
A solution for handwritten digit recognition based on openCV 3.0.

### Development Environment
Win 8 64bits, VS 2013, openCV 3.0.

### Input
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- 60000 images for training, 10000 for testing.

### Models
- SVM
  - Accuracy: 0.9886 for 10000 testing images.
  - Using Hog feature as trianing input.
- Adaboost
  - Accuracy: 0.9658 for 10000 testing images.
  - Using Hog feature as trianing input. **May as a negative effect on accuracy. **
  - 'Unroll' the database to transform the multi-class problem into 2-class problem. ** Check the code for detail.

### Areas for improvement
- A better feature for this situation.
- Some other classifiers may be used for better accuracy.
