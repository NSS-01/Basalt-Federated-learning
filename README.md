# Basalt

 ## Project

Basalt is an efficient server-client defense mechanism against Byzantine attacks.  On the client side, we design an efficient self-defense approach with model-level penalty loss that restricts local-benign divergence and decreases local-malicious correlation to prevent misclassification. Besides, on the server side, we present an efficient defense based on the manifold approximation and the maximum clique, further enhancing the capabilities to defend against malicious Byzantine attacks. We provide rigorous robustness guarantees by proving that the difference between the global model of Basalt and the optimal global model is bounded. Our extensive experiments demonstrate that Basalt outperforms existing state-of-the-art works. Especially, It achieves nearly 100% accuracy in detecting malicious clients in non-IID MNIST datasets under various Byzantine attacks.


## Citation 
```
@article{Song_2024,
title={Basalt: Server-Client Joint Defense Mechanism for Byzantine-Robust Federated Learning},
url={http://dx.doi.org/10.36227/techrxiv.171073035.50327931/v1},
DOI={10.36227/techrxiv.171073035.50327931/v1},
publisher={Institute of Electrical and Electronics Engineers (IEEE)},
author={Song, Anxiao and Li, Haoshuo and Zhang, Tao and Cheng, Ke and Shen, Yulong},
year={2024},
month=mar }
```

## Usage

This script supports a variety of command-line arguments to customize the training configuration. Here's a list of all available arguments and the descriptions:

- `--model`: The neural network model to be used during training. Default is `simple-cnn-mnist`.
- `--dataset`: The dataset used for training. Default is `mnist`.
- `--net_config`: Network configuration specified as a comma-separated list of integers, e.g., `--net_config 64,128,256`.
- `--partition`: The data partitioning strategy. Default is `noniid`.
- `--batch-size`: Input batch size for training. Default is `64`.
- `--lr`: Learning rate. Default is `0.01`.
- `--epochs`: Number of local epochs. Default is `5`.
- `--n_parties`: Number of workers in a distributed cluster. Default is `20`.
- `--n_malicious_parties`: Number of malicious parties. Default is `6`.
- `--alg`: Communication strategy, options include `fedavg`, `fedprox`, `basalt`, `Median`, `trimmed_mean`, `krum`. Default is `basalt`.
- `--comm_round`: Number of maximum communication rounds. Default is `50`.
- `--init_seed`: Random seed. Default is `0`.
- `--dropout_p`: Dropout probability. Default is `0.0`.
- `--datadir`: Data directory. Default is `"./data/"`.
- `--reg`: L2 regularization strength. Default is `1e-5`.
- `--logdir`: Log directory path. Default is `"./logs/"`.
- `--modeldir`: Model directory path. Default is `"./models/"`.
- `--alpha`: The parameter for the Dirichlet distribution for data partitioning. Default is `0.1`.
- `--beta`: The parameter for penalty constant. Default is `2`.
- `--device`: The device to run the program. Default is `cuda`.
- `--log_file_name`: The log file name. Default is `None`.
- `--optimizer`: The optimizer to be used. Default is `sgd`.
- `--mu`: The mu parameter for FedProx. Default is `1`.
- `--out_dim`: The output dimension for the projection layer. Default is `256`.
- `--local_max_epoch`: The number of epochs for local optimal training. Default is `100`.
- `--model_buffer_size`: Store how many previous models for contrastive loss. Default is `1`.
- `--pool_option`: Pooling option, either `FIFO` or `BOX`. Default is `FIFO`.
- `--sample_fraction`: How many clients are sampled in each round. Default is `1.0`.
- `--load_model_file`: The model file to load as the global model. Default is `None`.
- `--load_pool_file`: The old model pool path to load. Default is `None`.
- `--load_model_round`: How many rounds have executed for the loaded model. Default is `100`.
- `--load_first_net`: Whether to load the first net as the old net or not. Default is `1`.
- `--normal_model`: Use normal model or aggregate model. Default is `0`.
- `--save_model`: Save the trained model or not. Default is `0`.
- `--use_project_head`: Use projection head. Default is `1`.
- `--server_momentum`: The server momentum for FedAvgM. Default is `0`.
- `--attack`: Attack methods, options include `label_flipping`, `sign_flipping`, `Gaussian_attack`. Default is `label_flipping`.

To run the script with the default parameters, use the following command:

```bash
>python main.py --model simple-cnn-mnist --dataset mnist --partition noniid --batch-size 64

