import tqdm
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import matplotlib.pyplot as plt


def preprocessing(data):
    # Change labels to 0 and 1
    data[1] = data[1].map({'M': 1, 'B': 0})

    # Drop the first column which is the ID
    data.drop(0, axis=1, inplace=True)


    # Standardize the data
    scaler = MinMaxScaler()

    features = [col for col in data.columns if col != 'label']
    data[features] = scaler.fit_transform(data[features])

    return data


def load_data(filename):
    df = pd.read_csv(filename, header=None)

    return df



def split_data(data):
    y = data[1]
    X = data.drop(1, axis=1)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
    return X_train, X_val, y_train, y_val


def save_data_split(X_train, X_val, y_train, y_val, output_filename_train_data, output_filename_val_data):
    train_data = pd.concat([y_train, X_train], axis=1)
    val_data = pd.concat([y_val, X_val], axis=1)

    train_data.to_csv(output_filename_train_data, index=False)
    val_data.to_csv(output_filename_val_data, index=False)



def split_data_to_x_y(data):
    y = data[0]
    X = data.drop(0, axis=1)
    y = pd.get_dummies(y).values

    return X, y


def init(X_train, hidden_layer_nb=2, outputs_nb=2, weights_initializer='heUniform', hidden_nodes_nb=None):

    if hidden_nodes_nb is None:
        hidden_nodes_nb = int(((X_train.shape[1] + outputs_nb) / 2))

    weights = []
    biases = []

    for layer in range(hidden_layer_nb + 1):
        # Input Layer to Hidden Layer
        if layer == 0:
            nodes_in = X_train.shape[1]
            nodes_out = hidden_nodes_nb
        # Hidden Layer to Hidden Layer
        elif layer < hidden_layer_nb:
            nodes_in = hidden_nodes_nb
            nodes_out = hidden_nodes_nb
        # Hidden Layer to Output Layer
        else:
            nodes_in = hidden_nodes_nb
            nodes_out = outputs_nb

        if weights_initializer == 'xavier':
            limit = np.sqrt(6 / (nodes_in + nodes_out))
        else:
            limit = np.sqrt(6 / nodes_in)
        weights.append(np.random.uniform(-limit, limit, (nodes_out, nodes_in)))
        biases.append(np.zeros(nodes_out))

    return hidden_nodes_nb, weights, biases


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu_derivative(x):
    return (x > 0).astype(int)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def evaluate(y_train, y_pred, loss='binary_cross_entropy'):
    if loss == 'binary_cross_entropy':
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        # N = y_train.shape[0]
        N = len(y_train)

        loss = -(1/N) * np.sum(
            y_train * np.log(y_pred) +
            (1 - y_train) * np.log(1 - y_pred)
        )
    return loss


def evaluate_metrics(y_train, y_pred, loss='binary_cross_entropy'):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = evaluate(y_train, y_pred, loss)

    y_pred_class = (y_pred >= 0.5).astype(int)
    y_true_class = y_train.astype(int)

    accuracy = np.mean(y_pred_class == y_true_class)

    # Precision, Recall, F1
    true_positives = np.sum((y_true_class == 1) & (y_pred_class == 1))
    false_positives = np.sum((y_true_class == 0) & (y_pred_class == 1))
    false_negatives = np.sum((y_true_class == 1) & (y_pred_class == 0))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # Vérifie s'il y a deux classes avant de calculer l'AUC-ROC
    # if len(np.unique(y_pred)) > 1 and len(np.unique(y_train)) > 1:
    #     print("ROC_AUC")
    #     roc_auc = roc_auc_score(y_train, y_pred)
    # else:
    #     roc_auc = 0
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }
    return metrics


def forward_propagation(X, weights, biases, activation='sigmoid', output_activation='softmax'):
    layers = [X]  # List of layer activations
    Z = []        # List of pre-activation values

    for i in range(len(weights)):
        z = np.dot(layers[i], weights[i].T) + biases[i]
        Z.append(z)

        # Compute activation
        if i == len(weights) - 1:
            if output_activation == 'softmax':
                activation_output = softmax(z)
            else:
                activation_output = sigmoid(z)
        else:
            if activation == 'sigmoid':
                activation_output = sigmoid(z)
            elif activation == 'relu':
                activation_output = relu(z)

        layers.append(activation_output)

    return layers, Z


def backward_propagation(y_true, activations, Z, weights, activation='sigmoid'):
    gradients = {"dW": [], "db": []}
    num_layers = len(weights)
    m = y_true.shape[0]  # Number of samples

    delta = (activations[-1] - y_true) / m

    for i in reversed(range(num_layers)):
        dW = np.dot(delta.T, activations[i])
        db = np.sum(delta, axis=0)

        gradients["dW"].insert(0, dW)
        gradients["db"].insert(0, db)

        if i > 0:
            if activation == 'sigmoid':
                delta = np.dot(delta, weights[i]) * sigmoid_derivative(Z[i-1])
            elif activation == 'relu':
                delta = np.dot(delta, weights[i]) * relu_derivative(Z[i-1])

    return gradients


def update_parameters(weights, biases, gradients, learning_rate):
    for i in range(len(weights)):
        weights[i] -= learning_rate * gradients["dW"][i]
        biases[i] -= learning_rate * gradients["db"][i]
    return weights, biases


def train(data_train, data_predict, hidden_layer_nb=2, output_nb=2,  epochs=1000, learning_rate=0.0005, batch_size=8, patience_early_stop=30):

    # Initialize
    X_train, y_train = split_data_to_x_y(data_train)
    X_val, y_val = split_data_to_x_y(data_predict)
    hidden_nodes_nb, weights, biases = init(X_train, hidden_layer_nb, output_nb, 'xavier', 12)
    n_samples = X_train.shape[0]
    wait = 0
    best_loss = float('inf')
    metrics_train = []
    metrics_val = []


    # Training
    pbar = tqdm.tqdm(range(epochs))
    for epoch in pbar:
        # Shuffle data
        permutation = np.random.permutation(n_samples)
        X_shuffled = X_train.iloc[permutation]
        y_shuffled = y_train[permutation]

        epoch_loss = 0
        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled.iloc[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Forward pass
            activations, Z = forward_propagation(X_batch, weights, biases, 'relu')

            # Compute loss
            loss = round(evaluate(y_batch, activations[-1]), 4)

            epoch_loss += loss

            # Backward pass
            gradients = backward_propagation(y_batch, activations, Z, weights, 'relu')

            # Update parameters
            weights, biases = update_parameters(weights, biases, gradients, learning_rate)

        avg_loss = epoch_loss / (n_samples // batch_size)

        # Compute metrics
        metrics_train.append(evaluate_metrics(y_batch, activations[-1]))
        activations, _ = forward_propagation(X_val, weights, biases, 'relu')
        metrics_val.append(evaluate_metrics(y_val, activations[-1]))

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience_early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            pbar.set_description(f"Loss: {avg_loss:.4f}")

    return weights, biases, metrics_train, metrics_val


def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def display_results(metrics_train, metrics_val):
    plt.figure(figsize=(12, 6))


    loss_train = moving_average([m['loss'] for m in metrics_train])
    loss_val = moving_average([m['loss'] for m in metrics_val])
    plt.subplot(2, 2, 1)
    plt.plot(loss_train, label='Train')
    plt.plot(loss_val, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    accuracy_train = moving_average([m['accuracy'] for m in metrics_train])
    accuracy_val = moving_average([m['accuracy'] for m in metrics_val])
    plt.subplot(2, 2, 2)
    plt.plot(accuracy_train, label='Train')
    plt.plot(accuracy_val, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.show()


def predict(data, weight, bias):
    X_val, y_val = split_data_to_x_y(data)

    activations, _ = forward_propagation(X_val, weight, bias, 'relu')
    y_pred = activations[-1]
    results = evaluate_metrics(y_val, y_pred)


    print(results)


def load_weight_bias(file_weight, file_bias):
    weights = np.load(file_weight, allow_pickle=True)
    biases = np.load(file_bias, allow_pickle=True)
    return weights, biases


def save_weight_bias(weights, biases, file_weight, file_bias):
    weights = np.array(weights, dtype=object)
    biases = np.array(biases, dtype=object)
    np.save(file_weight, weights)
    np.save(file_bias, biases)



def init_args():
    parser = argparse.ArgumentParser(
        description='Multi-layer Perceptron for binary classification from malignant and benign breast cancer cells',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data-split',
        type=str,
        default='data/raw/data.csv',
        help='Path to the data file'
    )

    parser.add_argument(
        '--data-train',
        type=str,
        default='data/processed/data_train.csv',
        help='Path to the processed predict data file'
    )

    parser.add_argument(
        '--data-predict',
        type=str,
        default='data/processed/data_predict.csv',
        help='Path to the processed data file'
    )

    parser.add_argument(
        '--data-train-weight',
        type=str,
        default='data/trained/weights.npy',
        help='Path to the trained weights file'
    )

    parser.add_argument(
        '--data-train-biais',
        type=str,
        default='data/trained/biases.npy',
        help='Path to the trained biases file'
    )


    parser.add_argument(
        'mode',
                    choices=['train', 'split', 'predict', 'sklearn'],
                    help='Mode of operation: train or predict'
    )
    args = parser.parse_args()
    return args


def main():
    args = init_args()

    if args.mode == 'split':
        data = load_data(args.data_split)
        data = preprocessing(data)
        X_train, X_val, y_train, y_val = split_data(data)
        save_data_split(X_train, X_val, y_train, y_val, args.data_train, args.data_predict)
        print('Data split successfully')
    elif args.mode == 'train':
        data_train = load_data(args.data_train)
        data_predict = load_data(args.data_predict)
        weights, biases, metrics_train, metrics_val, = train(data_train, data_predict, hidden_layer_nb=2)
        display_results(metrics_train, metrics_val)
        save_weight_bias(weights, biases, args.data_train_weight, args.data_train_biais)
    elif args.mode == 'sklearn':
        data_train = load_data(args.data_train)
        X_train, y_train = split_data_to_x_y(data_train)
        data_predict = load_data(args.data_predict)
        X_val, y_val = split_data_to_x_y(data_predict)
        mlp = MLPClassifier(hidden_layer_sizes=(16, 16),
                            max_iter=1000, random_state=42)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        plt.plot(mlp.loss_curve_)
        plt.title("Évolution de la perte (log-loss), with best loss : " + str(mlp.best_loss_))
        plt.xlabel("Itérations")
        plt.ylabel("Perte")
        plt.show()
        print(f"Accuracy: {accuracy}")
    else:
        data = load_data(args.data_predict)
        weights, biases = load_weight_bias(args.data_train_weight, args.data_train_biais)
        predict(data, weights, biases)


if __name__ == '__main__':
    main()
