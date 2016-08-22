# HandwrittenDigitRecognition
An **incomplete** solution for handwritten digit recognition based on openCV 3.0.

### Development Environment
Win 8 64bits, VS 2013, openCV 3.0.

### Input
![input1](https://github.com/AmazingZhen/HandwrittenDigitRecognition/blob/master/handwrittenDigitRecognition/dataset/1.jpg?raw=true)
![input2](https://github.com/AmazingZhen/HandwrittenDigitRecognition/blob/master/handwrittenDigitRecognition/dataset/2.jpg?raw=true)
![input3](https://github.com/AmazingZhen/HandwrittenDigitRecognition/blob/master/handwrittenDigitRecognition/dataset/3.jpg?raw=true)

### Result
![result1](https://github.com/AmazingZhen/HandwrittenDigitRecognition/blob/master/handwrittenDigitRecognition/res/1.jpg?raw=true)
![result2](https://github.com/AmazingZhen/HandwrittenDigitRecognition/blob/master/handwrittenDigitRecognition/res/2.jpg?raw=true)
![result3](https://github.com/AmazingZhen/HandwrittenDigitRecognition/blob/master/handwrittenDigitRecognition/res/3.jpg?raw=true)

### Procedure
- How to cut digits?
  - Get gray image and do binaryzation.
  - Erode it to remove noise and enlarge digit area.
  - Do canny edge detection.
  - Get bounding rectange around the digits from edge pixel sets, each digit should has it own set.
- How to recognize digits?
  - Do pretreatment for all digits and then extract HOG features. 
  - Use SVM to predict HOG features and get predicted labels.

### Train Input
- [MNIST](http://yann.lecun.com/exdb/mnist/)
- 60000 images for training, 10000 for testing.

### Train Models
- SVM
  - Accuracy: 0.9886 for 10000 testing images.
  - Using Hog feature as trianing input.
- Adaboost
  - Accuracy: 0.9658 for 10000 testing images.
  - Using Hog feature as trianing input. *May has a negative effect on accuracy.*
  - 'Unroll' the database to transform the multi-class problem into 2-class problem. *Check the code for detail.*

### Areas for improvement
- A better feature for this situation.
- Some other classifiers may be used for better accuracy.
- A better way to cut digits.
