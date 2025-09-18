## Setup
- Clone Repo with this Command (run in bash, zsh, etc)
```bash
  git clone https://github.com/Keshav-writes-code/NeuralNetInCppTry.git
```

## Usage
- Create a main.cpp file in the projects root directory if doesn't already exists
- in your main.cpp file, Import the classes.h header

```cpp
#include "classes.h"
```
- Then create a NeuralNet Object with this format

```cpp
/*
1st Param : Input Layer Size (Int)
2nd Param : Hidden Layer Count (Int)
3rd Param : Hidden Layer Sizes (Array of Ints)
4th Param : Output Layer Size (Int)
5th Param : Learning Rate (float)
*/
NeuralNet* NN = new NeuralNet(2, 5, {3, 4, 6, 7, 4 }, 5, 0.03);
```

## Demo
![image](https://github.com/user-attachments/assets/0583a0df-c0f6-45f6-91b5-4e3c248d9281)
