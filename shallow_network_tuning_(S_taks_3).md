# shallow neural network tuning, summary task 3. 

ill write the tuning attempts I made, mostly focusing on the successful ones

normal configuration - <br/>
Accuracy: 89%

hidden neuron layer size -

| hidden neuron layer size | Accuracy | cost after 99% |
|--------------------------|:--------:|----------------|
| 4 (the original)         |   89%    | 0.281          |
| 5                        |   87%    | 0.313          |
| 6                        |   88%    | 0.314          |
| 7                        |   88%    | 0.312          |
| 10                       |   88%    | 0.312          |


activation for hidden layer -

| activation              | Accuracy | cost after 99% |
|-------------------------|:--------:|----------------|
| tanh (the original)     |   89%    | 0.281          |
| trim_tanh               |   88%    | 0.383          |
| sigmoid (just for fun)  |   88%    | 0.326          |
| trim_sigmoid            |   88%    | 0.326          |
| relu                    |   77%    | 0.510          |
| leaky_relu              |   77%    | 0.510          |


learning rate for hidden layer - 

| learning rate      | Accuracy | cost after 99% |
|--------------------|:--------:|----------------|
| 0.01               |   86%    | 0.326          |
| 0.05               |   88%    | 0.297          |
| 0.1 (the original) |   89%    | 0.281          |
| 0.15               |   89%    | 0.281          |
| 0.2                |   77%    | 0.273          |
| 1                  |   91%    | 0.261          |
| 3                  |   90%    | 0.259          |

num of iterations - 

| num of iterations    | Accuracy | cost after 99% |
|----------------------|:--------:|----------------|
| 1000                 |   85%    | 0.423          |
| 5000                 |   89%    | 0.302          |
| 10000 (the original) |   89%    | 0.287          |
| 15000                |   89%    | 0.279          |
| 30000                |   90%    | 0.266          |
| 50000                |   90%    | 0.240          |
| 100000               |   90%    | 0.216          |
| 1000000              |   91%    | 0.200          |

num of iteration change + learning rate up -  

| num of iterations    | learning rate | Accuracy | cost after 99% |
|----------------------|:-------------:|:--------:|----------------|
| 1000                 |      0.1      |   87%    | 0.423          |
| 1000                 |      0.5      |   87%    | 0.353          |
| 1000                 |       1       |   89%    | 0.324          |
| ------------         |      ---      |   ----   | -----          |
| 5000                 |      0.1      |   89%    | 0.302          |
| 5000                 |      0.5      |   90%    | 0.269          |
| 5000                 |       1       |   90%    | 0.257          |
| ------------         |      ---      |   ----   | -----          |
| 10000                | original(0.1) |   89%    | 0.287          |
| 10000 (the original) |      0.5      |   90%    | 0.238          |
| 10000 (the original) |       1       |   91%    | 0.249          |
| ------------         |      ---      |   ----   | -----          |
| 50000                |      0.1      |   90%    | 0.240          |
| 50000                |      0.5      |   91%    | 0.207          |
| 50000                |       1       |   91%    | 0.239          |


num of iteration up + learning rate down -
# didnt finish 
| num of iterations  | learning rate | Accuracy | cost after 99% |
|--------------------|:-------------:|:--------:|----------------|
| 10000              |     0.01      |   87%    | 0.313          |
| 10000              |     0.05      |   88%    | 0.290          |
| 10000              | original(0.1) |   89%    | 0.281          |
| ----------------   |      ---      |   ----   | -----          |
| 50000              |     0.01      |   88%    | 0.290          |
| 50000              |     0.05      |   88%    | 0.269          |
| 50000              |      0.1      |   90%    | 0.224          |
| ----------------   |      ---      |   ----   | -----          |
| 100000             |     0.01      |   88%    | 0.283          |
| 100000             |     0.05      |   89%    | 0.257          |
| 100000             |      0.1      |   91%    | 0.211          |
| ----------------   |      ---      |   ----   | -----          |
| 1000000            |     0.01      |   90%    | 0.248          |
| 1000000            |     0.05      |   91%    | 0.239          |
| 1000000            |      0.1      |   91%    | 0.198          |

adding another hidden layer - 

| amount of neurons in the hidden layer |  Accuracy  | accuracy |
|---------------------------------------|:----------:|----------|
| 0 (the original)                      |    89%     | 0.281    |
| 2                                     |    90%     | 0.237    |
| 3                                     |    87%     | 0.212    |





