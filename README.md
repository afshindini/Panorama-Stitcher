# panorama stitcher
This rep can stitch multi panorama images. It contains several deep and image-based analysis to do so.



## Available Methods and Preliminaries
This library offers several panorama stitching methods based on different techniques.

General options can be selected before applying the stitching methods. These options are:
- You should add the path for the image directory with the help of `-d` or `--data_path`.
- One can resize the image with `-r` or `--resize_shape`. (Default value is `None` which uses the original size of the images.)
- Define the result path with `--result_path` or `-s` (Default directory is `./`).
- Select the verbose value for logging for example as `-v` depending on what kind of logs you want to see.

The available methods are as following.
### Simple OpenCV Stitcher
This method mainly uses stitcher class from opencv to create the panorama images from multi images. It is one of the fastest and applicable methods in case of multi-image stitching.
Only one option should be set for this method as:
- Set `--stitcher_type` as "scan" if the images are scan or "panorama" if images are panorama images.

A simple code to stitch boat test images is:
```shell
panaroma_stitcher -vv -d ./test_data/boat opencv-simple --stitcher_type panorama
```
This method is recommended than other methods as it is fast, and it can stitch multi high resolution images properly.

### Kornia Stitcher
This method is based on `kornia` library, and three feature matcher as `LOFTR` deep feature matcher, `GFTTAffNetHardNet`,
and `KeyNetAffNetHardNet` matcher. The accuracy of this method is good specifically for stitching two images. However, since it uses a deep feature extractor and matcher
it requires large memory and is a bit slow. It uses a GPU automatically in cases where CUDA is installed.
Some options are:
- `--method` is used to define the matching method as "loftr", "local", "keynote".
- `--loftr_model` should be selected as "indoor" or "outdoor" depending on where the images are taken is someone uses "loftr" matcher.
- `--features` is the number of features in "local" and "keynote" methods.
- `--matcher` defines the matching algorithms in "local" or "keynote" methods. It can be "snn", "nn", "mnn", or "smnn".

Some examples of using these methods:
```shell
panaroma_stitcher -vv -d ./test_data/mountain kornia --method loftr --loftr_model outdoor
panaroma_stitcher -vv -d ./test_data/river kornia --method local --features 100 --matcher smnn
panaroma_stitcher -vv -d ./test_data/mountain kornia --method keynote --features 100 --matcher mnn
```

## How to Develop
Do the following only once after creating your project:
- Init the git repo with `git init`.
- Add files with `git add .`.
- Then `git commit -m 'initialize the project'`.
- Add remote url with `git remote add origin REPO_URL`.
- Then `git branch -M master`.
- `git push origin main`.
Then create a branch with `git checkout -b BRANCH_NAME` for further developments.
- Install poetry if you do not have it in your system from [here](https://python-poetry.org/docs/#installing-with-pipx).
- Create a virtual env preferably with virtualenv wrapper and `mkvirtualenv -p $(which python3.10) ENVNAME`.
- Then `git add poetry.lock`.
- Then `pre-commit install`.
- For applying changes use `pre-commit run --all-files`.

## Docker Container
Under development.
