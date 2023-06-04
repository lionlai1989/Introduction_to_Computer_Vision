# Introduction to Computer Vision
This github package hosts my notes and programming codes when taking the course, Introduction to Computer Vision, by Coursera and Georgia tech. 


## Description


## Getting Started
All the results can be reproduced by following the instructions below.

### Dependencies
Before you start, you need to make sure you have the following dependencies installed:
* **CMake 3.22.0 or higher:** If you don't have CMake installed, or if you need to update it, you can follow the instructions [here](https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line). To use version 3.22, you can download it from https://cmake.org/files/v3.22/cmake-3.22.6.tar.gz.
* **Eigen library:** This is a C++ library that we'll use in our project. Don't worry about installing it separately, as it's included as a git submodule in our repository.
* **stb image library:** This is a C library for loading and saving images. It's also included as a git submodule, so you don't need to do anything extra.
* **xtensor-assosiated library:** xtensor is a numpy for C++ library. All
  required code is added as git submodules. Unlike eigen and stb library, I
  can't figure out a way to just add folders to make CMake work. We need to
  first install xtensor and then use it. Please follow the instruction below:
  * Isntall xtl:
  ```
  (cd extern/xtl && cmake -D CMAKE_INSTALL_PREFIX=/tmp/xtl-install && make install)
  ```
  * Install xtensor:
  ```
  (cd extern/xtensor && cmake -D CMAKE_INSTALL_PREFIX=/tmp/xtensor-install -DCMAKE_PREFIX_PATH=/tmp/xtl-install && make install)
  ```
  * Install xsimd:
  ```
  (cd extern/xsimd && cmake -D CMAKE_INSTALL_PREFIX=/tmp/xsimd-install && make install)
  ```
  

* **Development tools for Linux and VS Code:** To develop our project, we'll be using Linux and Visual Studio Code (VS Code). To have a smoother experience, you should install the following tools and extensions for VS Code:
  * `C/C++`
  * `C/C++ Extension`
  * `CMake`
  * `CMake Extension`
  * `llvm`
  * `lldb`
  * `ninja`
  * `python3-dev` (install this with apt install python3-dev)

### Downloading
* To download this repository, run the following command:
```shell
git clone --recursive https://github.com/lionlai1989/cmake_template.git
```
If you forgot to use the `--recursive` option when cloning, you can still clone the submodules by running the command `git submodule update --init --recursive`.

### Build, Install and Execute
- Build:  
  To build the project, run the following command:
  ```
  cmake -G Ninja -S . -B build/ && cmake --build build/ -j 4 && (cd build/; ctest -V)
  ```
  If the build is successful, the tests will be run automatically and you should see the message:
  ```
  100% tests passed, 0 tests failed out of 5
  ```

- Install  
  This package can be installed in your system or a custom location in your file system. To install it, run the following command:
  ```
  cmake --install build/ --prefix /tmp/install-test/
  ```
  The above code installs the package in `/tmp/install-test/`.
  
- Execute  
  The installation can be tested with the following command:
  ```
  /tmp/install-test/bin/rgb2gray -i /tmp/install-test/bin/book_in_scene.jpg -o ./examples/files/book_in_scene_gray.jpg -m eigen
  /tmp/install-test/bin/rgb2gray -i /tmp/install-test/bin/book.png -o ./examples/files/book_gray.png -m xtensor
  ```
  It will use `eigen` to create `book_in_scene_gray.jpg` and `xtensor` to create `book_gray.png` in `./examples/files/`. Here is an example input and output images.
  <p align="left">
    <img src="./examples/files/book_in_scene.jpg" width="300" title="Input RGB Image with JPG Format">
    <img src="./examples/files/book_in_scene_gray.jpg" width="300" title="Output Grayscale Image with JPG Format">
  </p>
  <p align="left">
    <img src="./examples/files/book.png" width="300" title="Input RGB Image with PNG Format">
    <img src="./examples/files/book_gray.png" width="300" title="Output Grayscale Image with PNG Format">
  </p>

### Developing
- Using Libraries in this Package:  
  To use the libraries included in this package, it is necessary to include the appropriate headers in your code and link to the libraries in your project.

- Future Developments:  
  In the future, there are plans to expand this project to include additional example libraries to further explore their use. These libraries include libvips, CImg, terrasect, and opencv. Additionally, the code will be further improved by utilizing tools such as Clang Static Analyzer and clang-tidy to identify potential issues and enhance overall code quality.

## Contributing

Your feedback, comments, and questions about this project are welcome, as are any contributions you'd like to make. Please feel free to create an issue or a pull request in this repository. Together, let's improve this template and make life easier for C++ programmers.

## Authors

[@lionlai](https://github.com/lionlai1989)

## Version History

* 0.0.1
    * Initial Release

## Acknowledgments
Explore the inspiration and references listed here to further expand your knowledge and sharpen your skills.




Coursera: https://learn.udacity.com/courses/ud810

https://docs.google.com/spreadsheets/d/1ecUGIyhYOfQPi3HPXb-7NndrLgpX_zgkwsqzfqHPaus/pubhtml

Find time to do the assignments in the speard sheet above.

https://faculty.cc.gatech.edu/~afb/classes/CS4495-Fall2014/
