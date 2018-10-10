
# AlphaZero
Simple and clean implementation of Alpha Zero algorithm

## Requirements
Tensorflow and Keras on Python >=3.5
Tested on Keras 2.2.0 and Tensorflow 1.8.0

## How to run
`python3 alpha-zero.py train` if you want to launch training
`python3 alpha-zero.py play` if you want to play against a trained policy. Make sure to have downloaded the .zip file in the release section and moved its contents inside `checkpoints/` folder

## Training
In the release section you can find some checkpoints of a neural network trained to play connect four game.
The training lasted about ~50 days and still it is not sufficient, because the selfplay phase is too demanding (tested with an i5-4670)

## Docs and explainations
You can find more info inside [tesi.pdf](tesi.pdf) (italian only)


This is my thesis project for Computer Engineering bachelor's degree
