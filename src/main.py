import logging
from sklearn.metrics import log_loss
import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import matplotlib.pyplot as plt


def preprocessing(data):
    # Change labels to 0 and 1
    logging.info("Preprocessing data...")
    data[1] = data[1].map({'M': 1, 'B': 0})
    logging.info("Changing labels to 0 and 1")

    # Drop the first column which is the ID
    data.drop(0, axis=1, inplace=True)
    logging.info("Dropping the first column which is the ID")


    # Standardize the data
    scaler = MinMaxScaler()

    features = [col for col in data.columns if col != 'label']
    data[features] = scaler.fit_transform(data[features])
    logging.info("Standardizing the data")

    return data


def load_data(filename):
    df = pd.read_csv(filename, header=None)

    return df



def split_data(data):
    data = data.sample(frac = 1)
    y = data[1]
    X = data.drop(1, axis=1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
    )
    return X_train, X_val, y_train, y_val


def save_data_split(X_train, X_val, y_train, y_val, output_filename_train_data, output_filename_val_data):
    train_data = pd.concat([y_train, X_train], axis=1)
    val_data = pd.concat([y_val, X_val], axis=1)

    train_data.to_csv(output_filename_train_data, index=False)
    val_data.to_csv(output_filename_val_data, index=False)
    logging.info("Saving the split data")



def split_data_to_x_y(data):
    y = data[0]
    X = data.drop(0, axis=1)
    y = pd.get_dummies(y).values

    return X, y


def init(X_train, hidden_layer_nb=2, outputs_nb=2, weights_initializer='he', hidden_nodes_nb=None):

    if hidden_nodes_nb is None:
        hidden_nodes_nb = int(((X_train.shape[1] + outputs_nb) / 2))

    weights = []
    biases = []

    for layer in range(hidden_layer_nb + 1):
        if layer == 0:
            nodes_in = X_train.shape[1]
            nodes_out = hidden_nodes_nb
        elif layer < hidden_layer_nb:
            nodes_in = hidden_nodes_nb
            nodes_out = hidden_nodes_nb
        else:
            nodes_in = hidden_nodes_nb
            nodes_out = outputs_nb

        if weights_initializer == 'xavier':
            logging.info("Using Xavier initialization")
            limit = np.sqrt(6 / (nodes_in + nodes_out))
        else:
            logging.info("Using He initialization")
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


def evaluate(y_train, y_pred, loss='categorical_cross_entropy'):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    N = len(y_train)

    if loss == 'binary_cross_entropy':
        loss = -(1/N) * np.sum(
            y_train * np.log(y_pred) +
            (1 - y_train) * np.log(1 - y_pred)
        )
    elif loss == 'categorical_cross_entropy':
        loss = -(1/N) * np.sum(y_train * np.log(y_pred))

    return loss


def evaluate_metrics(y_train, y_pred, loss='categorical_cross_entropy'):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = evaluate(y_train, y_pred, loss)

    y_pred_class = (y_pred >= 0.5).astype(int)
    y_true_class = y_train.astype(int)

    accuracy = np.mean(y_pred_class == y_true_class)

    true_positives = np.sum((y_true_class == 1) & (y_pred_class == 1))
    false_positives = np.sum((y_true_class == 0) & (y_pred_class == 1))
    false_negatives = np.sum((y_true_class == 1) & (y_pred_class == 0))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }
    return metrics


def forward_propagation(X, weights, biases, activation='relu', output_activation='softmax'):
    layers = [X]
    Z = []

    for i in range(len(weights)):
        z = np.dot(layers[i], weights[i].T) + biases[i]
        Z.append(z)

        if i == len(weights) - 1:
            if output_activation == 'softmax':
                activation_output = softmax(z)
                print("Softmax")
            else:
                activation_output = sigmoid(z)
        else:
            if activation == 'sigmoid':
                activation_output = sigmoid(z)
            elif activation == 'relu':
                activation_output = relu(z)

        layers.append(activation_output)


    return layers, Z


def backward_propagation(y_true, activations, Z, weights, activation='relu'):
    gradients = {"dW": [], "db": []}
    num_layers = len(weights)
    m = y_true.shape[0]

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


def train(data_train, data_predict, args, hidden_layer_nb=2, output_nb=2,  epochs=1000, learning_rate=0.0001, batch_size=10,
          patience_early_stop=100, beta1=0.9, beta2=0.999, epsilon=1e-8):

    X_train, y_train = split_data_to_x_y(data_train)
    X_val, y_val = split_data_to_x_y(data_predict)
    hidden_nodes_nb, weights, biases = init(X_train, hidden_layer_nb, output_nb, 'he', 22)
    n_samples = X_train.shape[0]
    metrics_train = []
    metrics_val = []

    m_w = [np.zeros_like(w) for w in weights]
    v_w = [np.zeros_like(w) for w in weights]
    m_b = [np.zeros_like(b) for b in biases]
    v_b = [np.zeros_like(b) for b in biases]
    best_val_loss = float('inf')
    best_weights = None
    best_biases = None
    best_metrics = None
    wait = 0

    pbar = tqdm.tqdm(range(epochs), disable=args.disable_pbar)
    logging.info("Training the model")
    for epoch in pbar:
        np.random.seed(epoch)
        permutation = np.random.permutation(n_samples)
        X_shuffled = X_train.iloc[permutation]
        y_shuffled = y_train[permutation]

        epoch_loss = 0
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled.iloc[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            activations, Z = forward_propagation(X_batch, weights, biases)

            loss = round(evaluate(y_batch, activations[-1]), 4)

            epoch_loss += loss

            gradients = backward_propagation(y_batch, activations, Z, weights)

            for l in range(len(weights)):
                m_w[l] = beta1 * m_w[l] + (1 - beta1) * gradients['dW'][l]
                v_w[l] = beta2 * v_w[l] + (1 - beta2) * (gradients['dW'][l] ** 2)

                m_w_hat = m_w[l] / (1 - beta1 ** (epoch + 1))
                v_w_hat = v_w[l] / (1 - beta2 ** (epoch + 1))

                weights[l] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)

                m_b[l] = beta1 * m_b[l] + (1 - beta1) * gradients['db'][l]
                v_b[l] = beta2 * v_b[l] + (1 - beta2) * (gradients['db'][l] ** 2)

                m_b_hat = m_b[l] / (1 - beta1 ** (epoch + 1))
                v_b_hat = v_b[l] / (1 - beta2 ** (epoch + 1))

                biases[l] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

            # weights, biases = update_parameters(weights, biases, gradients, learning_rate)

        avg_train_loss = epoch_loss / (n_samples // batch_size)

        val_activations, _ = forward_propagation(X_val, weights, biases)
        val_metrics = evaluate_metrics(y_val, val_activations[-1])
        metrics_val.append(val_metrics)
        val_loss = val_metrics['loss']

        metrics_train.append(evaluate_metrics(y_batch, activations[-1]))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = [w.copy() for w in weights]
            best_biases = [b.copy() for b in biases]
            best_metrics = val_metrics
            wait = 0
        else:
            wait += 1
            if wait >= patience_early_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                logging.info(f"Best validation loss: {best_val_loss:.4f}")
                print(f"Early stopping at epoch {epoch}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break

        pbar.set_description(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if epoch % 100 == 0:
            pbar.set_description(f"Loss: {avg_train_loss:.4f}")
            logging.info(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

    logging.info("Training completed")
    logging.info(f"Loss: {avg_train_loss}")
    logging.info(f"Accuracy: {metrics_val[-1]['accuracy']}")
    logging.info(f"Best metrics: {best_metrics}")
    return best_weights, best_biases, metrics_train, metrics_val

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

    activations, _ = forward_propagation(X_val, weight, bias)
    y_pred = activations[-1]
    results = evaluate_metrics(y_val, y_pred)


    logging.info("Predicting the data")
    print("Predicting the data")
    for key, value in results.items():
        print(f"{key}: {value}")
        logging.info(f"{key}: {value}")


def load_weight_bias(file_weight, file_bias):
    weights = np.load(file_weight, allow_pickle=True)
    biases = np.load(file_bias, allow_pickle=True)
    return weights, biases


def save_weight_bias(weights, biases, file_weight, file_bias):
    weights = np.array(weights, dtype=object)
    biases = np.array(biases, dtype=object)
    logging.info("Saving weights and biases")
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
        default='data/processed/data.csv',
        help='Path to the data file'
    )

    parser.add_argument(
        '--data-train',
        type=str,
        default='data/processed/data_training.csv',
        help='Path to the processed predict data file'
    )

    parser.add_argument(
        '--data-predict',
        type=str,
        default='data/processed/data_test.csv',
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
        '--disable-logs',
        action='store_true',
        help='Disable logs'
    )

    parser.add_argument(
        '--disable-plot',
        action='store_true',
        help='Disable plot of the results'
    )
    parser.add_argument(
        '--disable_pbar',
        action='store_true',
        help='Disable plot of the results'
    )



    parser.add_argument(
        'mode',
                    choices=['train', 'split', 'predict', 'sklearn', 'pre',],
                    help='Mode of operation: train or predict'
    )
    args = parser.parse_args()
    return args


def init_logs(args):
    logging.basicConfig(filename='.log',
                               level=logging.INFO,
                               format='%(asctime)s - %(message)s')
    if args.disable_logs:
        logging.disable(logging.CRITICAL)


def main():
    args = init_args()
    init_logs(args)

    if args.mode == 'split':
        data = load_data(args.data_split)
        data = preprocessing(data)
        X_train, X_val, y_train, y_val = split_data(data)
        save_data_split(X_train, X_val, y_train, y_val, args.data_train, args.data_predict)
        print('Data split successfully')
        logging.info("Split data successfully")
    elif args.mode == "pre":
        data_train = preprocessing(load_data(args.data_train))
        data_predict = preprocessing(load_data(args.data_predict))
        data_train.to_csv(args.data_train, index=False)
        data_predict.to_csv(args.data_predict, index=False)
        # print('Data preprocessing successfully')
        logging.info("Data preprocessing successfully")
    elif args.mode == 'train':
        data_train = load_data(args.data_train)
        data_predict = load_data(args.data_predict)
        weights, biases, metrics_train, metrics_val, = train(data_train, data_predict, args, hidden_layer_nb=2)
        if not args.disable_plot:
            display_results(metrics_train, metrics_val)
        save_weight_bias(weights, biases, args.data_train_weight, args.data_train_biais)
    if args.mode == 'predict':
        data = load_data(args.data_predict)
        weights, biases = load_weight_bias(args.data_train_weight, args.data_train_biais)
        predict(data, weights, biases)
    if args.mode == 'sklearn':
        logging.info("Using sklearn")
        # print("Using sklearn")
        data_train = load_data(args.data_train)
        X_train = data_train.drop(0, axis=1)
        y_train = data_train[0]

        data_predict = load_data(args.data_predict)
        X_val = data_predict.drop(0, axis=1)
        y_val = data_predict[0]
        mlp = MLPClassifier(hidden_layer_sizes=(16, 16),
                            max_iter=1000, random_state=42, learning_rate_init=0.001, batch_size=10, solver="adam")
        mlp.fit(X_train, y_train)

        # bce_train = log_loss(y_train, mlp.predict_proba(X_train))
        bce_val = log_loss(y_val, mlp.predict_proba(X_val))

        # print(f"Binary Cross-Entropy sur Train : {bce_train}")
        print(f"{bce_val}")


if __name__ == '__main__':
    main()
