An implementation of package which enables the user to change the content of the data directories at the
pre-process stage.

Parameters
---
```--path``` - (string) path to directory contains the files (required parameter).

```--file_type``` - (string) the type of file (jpg, txt, png, etc.).

```--test_size``` - (float) (Default: 0.2) size of the test set, float number in [0, 1].

```--validation_size``` - (float) (Default: 0.) size of the validation set, float number in [0, 1].

```--divide_files_to_two_directories``` - (bool) (Default: False) If True -> it splits directory with files to two/three sub-directories depended by giving the --validation_size param.
1) dividing single directory to two directory:
    ex: dir -> train (x % of the files)
            -> test (100-x % of the files)

```--divide_sub_directories_to_two_directories``` - (bool) (Default: False) If True -> it splits directory with sub-directories to 
two/three sub-directories depended by giving the --validation_size param where each contains
the original sub directories with relative amount of files.
2) dividing divided directory to two sub-also-divided-directories:
    ex: dir/a           dir/train -> dir/train/a , dir/train/b , dir/train/c
        dir/b     ->    dir/test  -> dir/test/a  , dir/test/b  , dir/test/c
        dir/c

```--group_files_to_directories_by_prefix``` - (bool) (Default: False)  Groups files to directories by given prefixes. Example: ['dog','cat'] ->
would create 2 sub directories of 'dog' and 'cat' and each would contain all the images 
starts with the directory name.
3) group files by prefix to directories.
    ex: dir/dog_1.jpg       dir/dog  -> dir/dog/dog_1.jpg , dir/dog/dog_2.jpg
        dir/dog_2.jpg   ->  
        dir/cat_1.jpg       dir/cat  -> dir/cat/cat_1.jpg , dir/cat/cat_2.jpg
        dir/cat_2.jpg