# APAN 5560 — Applied Generative AI

This repository contains the cumulative coursework for APAN 5560.
Assignment 2 extends the FastAPI application developed in Assignment 1.
Assignment 3 adds MNIST digit generation using a trained GAN.

## Assignment Overview

| Assignment | Implementation | API Endpoints |
| --- | --- | --- |
| Assignment 1 | Bigram text generation and spaCy word embeddings | POST /generate, POST /embedding |
| Assignment 2 | CNN trained on CIFAR-10, with saved weights and Docker deployment | POST /predict-image |
| Assignment 3 | MNIST GAN with saved generator weights and Docker inference | GET /generate-digits |

For Assignment 2, see the "Assignment 2: CIFAR-10 Image Classification"
section below for the architecture, training results, and Docker commands.

## Assignment 1

A FastAPI application demonstrating bigram text generation and spaCy word embeddings.

## Features

- Generate text using a simple bigram language model.

- Calculate a 300-dimensional embedding for an input word.

- Validate request data with Pydantic.

- Provide interactive API documentation with Swagger UI.

- Run locally or inside a Docker container.

## Project Structure

```text
APAN_5560genai/
├── app/
│   ├── __init__.py
│   ├── bigram_model.py
│   ├── embedding_model.py
│   └── main.py
├── .dockerignore
├── .gitignore
├── .python-version
├── Dockerfile
├── pyproject.toml
├── README.md
└── uv.lock
```

## Run Locally

Install the locked dependencies:

```bash
uv sync
```

Start the development server:

```bash
uv run fastapi dev app/main.py
```

Open the interactive documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

- `GET /` returns basic API information.

- `GET /health` checks whether the API is running.

- `POST /generate` generates text with the bigram model.

- `POST /embedding` returns a 300-dimensional spaCy word embedding.

### Text Generation Request

```json
{
  "start_word": "the",
  "length": 10
}
```

### Word Embedding Request

```json
{
  "word": "apple"
}
```

## Run with Docker

Build the Docker image:

```bash
docker build -t apan_5560genai .
```

Run the container:

```bash
docker run -p 8000:80 apan_5560genai
```

Then open:

```text
http://127.0.0.1:8000/docs
```

Press `Ctrl+C` in the terminal to stop the container.

## Assignment 2: CIFAR-10 Image Classification

This assignment extends the existing FastAPI application with a
CNN image classification endpoint. The text generation and word
embedding endpoints from Assignment 1 are retained.

### CNN Architecture

- Input: RGB image resized to 64 × 64
- Convolution: 3 input channels, 16 output channels, 3 × 3 kernel,
  stride 1, padding 1
- ReLU
- Max pooling: 2 × 2 kernel, stride 2
- Convolution: 16 input channels, 32 output channels, 3 × 3 kernel,
  stride 1, padding 1
- ReLU
- Max pooling: 2 × 2 kernel, stride 2
- Flatten: 32 × 16 × 16 = 8192 features
- Fully connected: 8192 → 100
- ReLU
- Fully connected: 100 → 10 class scores

### Training

The model was trained on the CIFAR-10 training set for 5 epochs,
using a batch size of 64, cross-entropy loss, and the Adam optimizer
with a learning rate of 0.001.

Preprocessing consists of resizing images to 64 × 64 and converting
them to tensors.

- Final training accuracy: 68.37%
- Test accuracy: 63.61%
- Saved weights: `models/cifar10_cnn.pth`

The trained weights are included, so retraining is not required
to run the API. Inference supports CPU execution.

### Project Files

- `helper_lib/model.py`: CNN architecture
- `helper_lib/data_loader.py`: CIFAR-10 loading and preprocessing
- `helper_lib/trainer.py`: training loop
- `helper_lib/evaluator.py`: test-set evaluation
- `train_cnn.py`: training and weight-saving script
- `app/cnn_service.py`: weight loading and image inference
- `app/main.py`: FastAPI endpoints

### Run with Docker

From the repository root, build the image:

```bash
docker build -t apan_5560genai:assignment2 .
```

Start the container:

```bash
docker run -p 127.0.0.1:8000:80 apan_5560genai:assignment2
```

Open http://127.0.0.1:8000/docs in a browser.

### Image Classification Endpoint

`POST /predict-image`

Upload an image using the multipart form field named `file`.
In Swagger UI, select **Try it out**, choose an image, and click
**Execute**.

Supported classes:

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

Example response:

```json
{
  "filename": "test_cifar10.png",
  "class_index": 3,
  "class_name": "cat",
  "confidence": 0.6714
}
```

Confidence is the softmax score of the predicted class, not a
guarantee that the prediction is correct. Images outside the ten
CIFAR-10 classes are still assigned one of these classes.

### Verification

The Dockerized endpoint was tested with the first CIFAR-10 test
image, whose true label is `cat`. It returned HTTP 200 and predicted
`cat` with a confidence score of approximately 0.6714.

## Assignment 3: MNIST Image Generation

An unconditional GAN generates 28-by-28 grayscale handwritten digit images.
The notebook follows the Module 6 practical's organization while using the
assignment's MNIST architecture and binary cross-entropy objective, rather
than the practical's CelebA Wasserstein objective.

### Architecture and Training

| Model | Layers |
| --- | --- |
| Generator | 100-dimensional noise; Linear to 128 x 7 x 7; reshape; ConvTranspose2d(128, 64, 4, 2, 1); BatchNorm; ReLU; ConvTranspose2d(64, 1, 4, 2, 1); Tanh |
| Discriminator | Conv2d(1, 64, 4, 2, 1); LeakyReLU(0.2); Conv2d(64, 128, 4, 2, 1); BatchNorm; LeakyReLU(0.2); flatten; Linear to 1; Sigmoid |

Training uses 60,000 MNIST training images normalized to [-1, 1], batch
size 64, binary cross-entropy, and separate Adam optimizers with learning
rate 0.0002 and betas (0.5, 0.999). The saved experiment contains ten epochs:
epochs 1-5 followed by resumed epochs 6-10, with one demonstration batch
update before epoch 1. Ten epochs is an experiment setting, not a claim of
convergence or optimal image quality. Some generated digits remain ambiguous.

### Included Files

- `Assignment3_MNIST_GAN_Final.ipynb`: architecture, training, all ten epoch
  grids, loss curves, fresh samples, and weight reload verification.
- `helper_lib/gan.py`: reusable generator for inference.
- `app/gan_service.py`: CPU weight loading and PNG generation.
- `app/main.py`: the added API endpoint and existing assignment endpoints.
- `models/mnist_generator_classroom_epoch_010.pth`: inference weights.
- `outputs/gan_20261027_224535/checkpoints/gan_epoch_010.pth`: final training
  checkpoint containing both models, optimizer states, fixed noise, and logs.
- `outputs/gan_20261027_224535/samples/epoch_001.png` through `epoch_010.png`:
  fixed-noise samples for every epoch.

Earlier checkpoints, notebook drafts, virtual environments, and downloaded
datasets are excluded from Git. The final training checkpoint supports the
notebook's default resume setting; the API only needs the inference weights.

### Run with Docker

From the repository root, with Docker Desktop running Linux containers:

```bash
docker build -t genai-ass3 .
docker run --rm --name genai-ass3-test -p 127.0.0.1:8001:80 genai-ass3
```

The initial build downloads all cumulative project dependencies and can take
considerable time. Wait for `Application startup complete.` and keep the
terminal open while testing.

- Documentation: http://127.0.0.1:8001/docs
- Generate 16 images: http://127.0.0.1:8001/generate-digits?num_samples=16
- Generate 4 images: http://127.0.0.1:8001/generate-digits?num_samples=4

Port 8001 on the host forwards to port 80 inside the container. It avoids
conflicting with an optional local server on port 8000. The container runs
independently and does not require the local server. Press Ctrl+C to stop it.

### API Contract

`GET /generate-digits?num_samples=16` returns an `image/png` grid.
`num_samples` is an integer from 1 to 64, defaulting to 16. Inputs such as
0 or 65 return HTTP 422. Each request uses fresh random noise; the endpoint
does not select a particular digit class. Individual images are 28-by-28
pixels, so the returned grid is small at its native display size.

Save a response using curl (use `curl.exe` in Windows PowerShell):

```bash
curl "http://127.0.0.1:8001/generate-digits?num_samples=16" --output generated_digits.png
```

### Inspect or Rerun the Notebook

Saved outputs can be inspected without executing the notebook. To execute
it, use a Python environment with the project dependencies plus Jupyter,
Matplotlib, and tqdm. One option from the repository root is:

```bash
uv run --with jupyterlab --with matplotlib --with tqdm jupyter lab
```

Select the kernel belonging to that environment. The default `resume = True`
loads the included epoch-10 checkpoint and skips further training when
`epochs = 10`; the MNIST data cell downloads the dataset if it is absent.
Set `resume = False` to train a new experiment in a new timestamped directory.
The original checkpoint does not contain every random-number state, so a
fresh or resumed run is not guaranteed to reproduce the saved images exactly.

### Submission Scope

This repository covers the implementation and deployment portions of
Assignment 3. The conceptual and calculation answers are submitted separately.

