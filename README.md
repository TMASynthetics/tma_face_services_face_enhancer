# tma_face_services_face_enhancer
Pipeline for face enhancer.

The face enhancer calls the face feature analyser which itself calls the face detector.
**Of note**: our implementation of processors calling each other has **NOT** been defined at the moment.
This may trigger code redundancies and maintenance difficulty accross the different parts of facefusion !

For now, I circumvent this issue by cloning the face feature analyzer repo from github and running it on marie image.
I use the resulting JSON file from the face feature analyzer to build this processor.