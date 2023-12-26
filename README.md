# Basalt

> This is a code of Basalt.

Basalt is an efficient server-client defense mechanism against Byzantine attacks.  On the client side, we design an efficient self-defense approach with model-level penalty loss that restricts local-benign divergence and decreases local-malicious correlation to prevent misclassification. Besides, on the server side, we present an efficient defense based on the manifold approximation and the maximum clique, further enhancing the capabilities to defend against malicious Byzantine attacks. We provide rigorous robustness guarantees by proving that the difference between the global model of Basalt and the optimal global model is bounded. Our extensive experiments demonstrate that Basalt outperforms existing state-of-the-art works. Especially, It achieves nearly 100% accuracy in detecting malicious clients in non-IID MNIST datasets under various Byzantine attacks.

> Run cmd

python main.py --model simple-cnn-mnist --dataset mnist --partition noniid --batch-size 64
