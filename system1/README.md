# Real time face recognition tool - 01

1. requirements:
- python 3
- opencv
- numpy
- os
- tensorflow
- scipy
- sci-kit learn
- random
- re

# NOTE: 
facenet is imported from following repository:
https://github.com/davidsandberg/facenet

detect_face is also imported from following repository:
https://github.com/shanren7/real_time_face_recognition

# process:

* First, we need align face data. So, if you run 'Make_aligndata.py' first, the face data that is aligned in the 'output_dir' folder will be saved.
* Second, we need to create our own classifier with the face data we created. <br/>(In the case of me, I had a high recognition rate when I made 30 pictures for each person.)
</br>Your own classifier is a ~.pkl file that loads the previously mentioned pre-trained model ('[20170511-185253.pb](https://drive.google.com/file/d/0B5MzpY9kBtDVOTVnU3NIaUdySFE/edit)') and embeds the face for each person.<br/>All of these can be obtained by running 'Make_classifier.py'.<br/>
* Finally, we load our own 'my_classifier.pkl' obtained above and then open the sensor and start recognition.
</br> (Note that, look carefully at the paths of files and folders in all .py)
## Result

