# Tests Readme

There are two tests files:
- sklearn/_sk_test.py 
- tensorflow/_tf_images_test.py

The first one uses the file ```tester_data.csv```, the second uses the directory ```data_image_test.py```.
In case you want to test some libraries:
- put the test file and its relevant data set in the directory ```ailibrary```. It means you
need to bring it one level up in the directories tree.
- Insert the command: ```python3 TEST_FILE_NAME.py --algo=ALGORITHM_NAME```.