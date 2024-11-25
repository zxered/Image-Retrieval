## Image Retrieval

This project aims to solve a simple image retrieval problem. \
Image retrieval involves finding the image most similar to a given query image within a dataset (or database) of multiple images. \
It is commonly used to identify the location where an image was taken or to find images containing similar objects.

### Guide

1. **Download the dataset from [this link](https://drive.google.com/file/d/16YnixlaK-hrXxyzng_qqCtgBfRorsELS/view?usp=sharing).** This dataset is gathered from [Habitat-sim](https://github.com/facebookresearch/habitat-sim) simulator developed by Meta AI.

2. **For each image in the `query` directory, find the most similar image in the `database` directory.** Generate text file `answer.txt` as below. The first line means that 001235.jpg in database is the most similar with 000000.jpg in query. :

    ```
    query/000000.jpg database/001235.jpg
    query/000001.jpg database/000274.jpg
    query/000002.jpg database/000014.jpg
    query/000003.jpg database/001973.jpg
    ...
    query/000099.jpg database/001463.jpg
    ```

3. **Use ORB features for image retrieval.** The reason for limiting features to ORB is to avoid differences caused by the presence or absence of hardware acceleration like GPU. ORB features can be efficiently computed on CPU.
    - [OpenCV tutorial for ORB features](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html)
    - [OpenCV tutorial for ORB feature matching](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)

4. **Feel free to choose the method you find most suitable.** There are various approaches to image retrieval using features, from simple distance calculations to more complex methods like [Bag of Visual Words](https://www.youtube.com/watch?v=a4cFONdc6nc). The complexity of the approach will not affect the evaluation.

### Evaluation
Put your `answer.txt` file in this directory, and run `python evaluation.py`. The script will print recall@1 score. \
You can test it with the existing `answer.txt` file with 10 query samples. Your answer should include 100 query samples.

### Submission
Upload your code to a GitHub repository and share the repository link.