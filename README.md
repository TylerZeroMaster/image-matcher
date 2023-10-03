
# Introduction

### Overview 
* This is a tool I made for automatically grouping similar images together using [tfidf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
* I only spent a few days making this, so expect some bugs 🐛


### Firsts
* First time using numpy
* First attempt at creating an image comparison algorithm


### Relevant Prior Knowledge
* None, I just looked up tfidf on wikipedia and decided to make this :D


### Performance/Benchmarks
* 🤷‍♂️ With an i7 12700K, it grouped 3600 scans in about a minute. I have not tested it further.


# Usage

```
python main.py <path_to_walk_for_images> <w> <h> [...]
```

See help for additional options.


# How does it work?


### Preparation
* All images are resized to the same width and height
    * If the aspect ratio of an image does not match the target width/height, the image is skipped
* All images are converted to RGB


### Comparison Steps

1. Each image is split into rectangles
```
[[[1,2,3], [4,5,6],
 [7,8,9], [10,11,12]],
 ...]
```

2. Images `a` and `b` are compared by
    1. Taking the delta of each rectangle
    2. Taking the mean of the delta for each rectangle
    3. Counting the means that fall within a margin of error (`raw_count`)
    4. Dividing the raw count by the total number of terms (`tf`)

These steps yield the term frequency of each rectangle; that array could look like this
```
[0.625, 0.123, 0.0, 0.0625, ...]
```

...where each value is the term frequency of a given rectangle from image `a` in image `b`


3. Once image `a` has been compared to all images in the corpus (see step 2), the term frequency arrays are concatenated together
```
[[0.625, 0.123, 0.0, 0.0625, ...],
 [0.1625, 0.1123, 0.1, 0.1625, ...],
 ...]
```

4. The concatenated array of term frequencies is rotated -90 degrees so that each column corresponds to an image, each row corresponds to a rectangle, and each value is a term frequency
```
[...,
 [0.0625, 0.1625, ...],
 [0.0, 0.1, ...],
 [0.123, 0.1123, ...],
 [0.625, 0.1625, ...]]
```

Counting nonzero values in each row yields the number of documents a rectangle appeared in (`n_docs`)

5. IDF is calculated with
    * N = len(images)
    * n_docs obtained from step 4

```python
idf = np.log((N + 1) / (n_docs + 1))
```

6. The value of `tf * idf` is saved to a file


# Limitations

1. All images must be the same aspect ratio
1. Does not work very well on images with homogenous backgrounds
    * Caused by improper use of cosine similarity when interpreting results: currently cosine similarity is used to compare all values of tfidf output. This completely nullifies the benefits of IDF since the weighting means nothing.
1. Much of the work is repeated unnecessarily (ex: the idf of a rectangle should not change, but it is calculated for each image comparison)


# Possible Improvements

1. Use [CuPy](https://cupy.dev/) instead of numpy
    * Offload math to GPU instead of using multiprocessing on CPU (or maybe do a bit of both!? 🤯)
1. Interpret results in a way that utilizes tfidf weighting more effectively
1. Instead of comparing all rectangles in an image to all rectangles in every other image, only compare unique rectangles
    * I have some prototypes that work on this principal. Thus far, however, they are slower and less effective
    * This approach might also allow comparison between images of arbitrary sizes; however, interpreting results would be a problem (since the results would be different dimensions)
1. Options for handling mismatched aspect ratios (cropping, resizing, etc.)
1. Probably could avoid some reshapes and reduce memory footprint by eliminating unnecessary dimensionality
1. Add options to use different algorithms optimized for different things (memory usage vs. precision vs. GPU/CPU utilization, etc.)
