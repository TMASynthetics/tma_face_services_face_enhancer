# tma_face_services_face_enhancer
Pipeline for face enhancer.

The face enhancer calls the face feature analyser which itself calls the face detector.
**Of note**: our implementation of processors calling each other has **NOT** been defined at the moment.
This may trigger code redundancies and maintenance difficulty accross the different parts of facefusion !

For now, I circumvent this issue by cloning the face feature analyser repo from github.
I hope to make the update of this processor easier that way.
However, once deployed, this is replaced by calling different microservices.
I don't know what is the best code architecture regarding this point at the moment.