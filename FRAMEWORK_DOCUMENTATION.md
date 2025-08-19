# DynSL: Dynamic Split Learning Framework Documentation

DynSL is an extension of SplitBud, which in turn is a Split Learning framework built on top of [Flower](https://flower.ai/). SplitBud enables distributed machine learning where models are split between clients and servers, with clients performing local computation on the initial layers and servers handling the remaining layers. DynSL, on the other hand, has the following key features:

1. Native integration with PyTorch;
2. Significantly simpler API;
3. Automatic placement of trainable layers betwee every client and server. The choice is made based on clients' compute availability and server's computational load. The developer only defines the model architecture as a `nn.Sequential` model and gives it to the framework. The framework automatically determines the number of layers to be placed on every client and, if the conditions on clients change within an epoch (for instance, the client starts an additional process slowing down its training or else some process ends, so that the client becomes faster), the layers are transferred between clients and server within the training epoch. All of this is transparent to the developer - everything is automatically handled by the framework.

## Table of Contents

1. [Key Components](#key-components)
2. [Component Interactions](#component-interactions)
3. [User Implementation Guide](#user-implementation-guide)
4. [Example Usage](#example-usage)
5. [Framework Strengths and Improvement Areas](#framework-strengths-and-improvement-areas)

## Key Components

### 1. Server Components

#### Strategy (`slbd.server.strategy.Strategy`)
The Strategy class is the central orchestrator for the split learning process. It coordinates training rounds, manages client selection, and aggregates results.

- **Key Methods**:
  - `configure_fit()`: Sets up training for both client and server model segments. Specifically, the envisioned workflow within the method is the following:
    1. The strategy samples a set of active clients out of the pool of available clients.
    2. The strategy asks the `LayerAllocator` how many layers should be given to every sampled client.
    3. The strategy extracts the parameters of the determined number of layers for every sampled client.
    4. The strategy creates a single model on the server. The number of layers on the server is determined as the minimum number of layers that can be trained by any client in the current epoch. For instance, if the global model has 10 layers, and the two selected clients specified that they have to train at least 3 and 5 layers, respectively, the server model will consist of the latest $10-3=7$ layers.
  - `aggregate_fit()`: Receives the trained weights of every sampled client and the single server model. It aggregates the weights in order to compute the weights for the next training round.
  - `configure_evaluate()`: Sets up evaluation rounds. The workflow is akin to that of the `configure_fit` method.

**PlainSlStrategy**: is a concrete implementation of the above abstract strategy. During aggregation, it averages the weights as in the FedAvg algorithm.

#### Layer Allocator (`slbd.server.layer_allocator.LayerAllocator`)

Manages dynamic allocation of model layers between client and server. More in detail, it is given detailed per-client training metrics (e.g., local computation time, server computation time, etc), and, depending on such metrics, determines if more or less layers are to be trained by the clients.
- **Key Features**:
  - Supports runtime layer reallocation (though as of now, only basic implementation is provided)
  - Tracks per-client layer distribution, i.e., how many layers are being trained by every client.
- **Current Limitations**: Basic implementation with hardcoded reallocation logic

#### Server Model (`slbd.server.server_model.ServerModel`)

Abstract base class for server-side model components.
- **Responsibilities**:
  - Maintains server-side layers of the split model
  - Handles training on activations received from clients
  - Manages parameter updates and state

Note, that a server model is a container for the server-side model. That is, the `ServerModel` class has the model as an attribute, and in addition to it, also has additional required attributes such as the optimizer for the server model, metrics, etc.

#### Server Model Manager (`slbd.server.server_model.manager`)
Is a key component of the framework. It is the component that interacts with the `Server` to configure the server-side model at the beginning of training, and it is also used by the servicer to get `ServerModel` with the given ID.

### 2. Client Components

#### Abstract Torch Client (`slbd.client.abstract_torch_client.AbstractTorchClient`)
It is the base class that users must implement for their specific use case

- **Key Methods**:
  - `get_properties()`: Returns client capabilities (min layers, etc.)
  - `fit()`: Implements local training logic using provided ModelTrainer. In this context, training logic includes data preprocessing and provisioning to the ModelTrainer -- the actual forward/backward/optimizer steps are handled by the framework.
  - `evaluate()`: Implements evaluation logic

#### Model Trainer (`slbd.client.model_trainer.ModelTrainer`)

It is the component that handles the actual forward/backward pass and communication with server.

- **Key Features**:
  - Manages client-side model layers
  - Coordinates with server model for forward/backward propagation
  - Handles dynamic layer addition/removal
  - Tracks training metrics and timing
  - Communicates with the server. Note, that it is the only component that does so - the code inside the Client cannot directly communicate with the server.


### 3. Common Components

#### Communication Layer (`slbd.proto/`, `slbd.client.grpc/`, `slbd.server.grpc/`)
- **Purpose**: gRPC-based communication between clients and server
- **Components**:
  - Protocol buffer definitions for messages
  - Connection management
  - Message handlers for training coordination

#### Utilities (`slbd.common/`, `slbd.torch/`)
- **Purpose**: Shared utilities for model manipulation, serialization, and constants
- **Key Features**:
  - Model layer extraction and manipulation
  - Parameter serialization/deserialization
  - PyTorch-specific utilities

## Servicing client requests

We now outline the flow of client requests from the moment they are received by the server to the moment they get passed to the server model for computation.

1. A batch of data is received by the `ServerModelServicer` in the `StreamRequest` method. The deserialized data is then provided to the `_compute_response` method.
2. The `update_metrics` method of the `LayerAllocator` is called by providing the latest timing metrics tracked for the considered client (client local computation time, server computation time, etc.).
3. The `route_client_request` method of the strategy is called to determine which server model should handle the present request. The strategy returns a `ClientRequestGroup` that batches requests and a callback function for cleanup. Currently, the strategy supports either processing clients individually or as a batch depending on the `process_clients_as_batch` flag.
4. Once the client request (or a group of client requests) is marked by the strategy as being ready for processing, the computation on the server begins:
    1. For every client, we determine how many layers are to be processed by calculating the offset from the base (minimum client layers)
    2. We call the training logic of the `ServerModel` by calling either `train_and_compute_gradient()` for training or `eval_get_logits()` for evaluation, specifying the offset that determines how many server layers to skip.
5. After the processing has completed, the gradient is returned to the client. In doing so, the `get_layers_diff` method of the `LayerAllocator` is also called to check if more layers are to be sent to the client. In such a case, the `LayerAllocator` updates its state so that it always knows how many layers are being processed by every client.

## Component Interactions

### Training Flow

1. **Initialization**:
   ```
   Server Strategy → Layer Allocator → Client Selection → Model Split → Client/Server Model Setup
   ```

2. **Training Round**:
   ```
   Client (local data) → ModelTrainer (forward pass) → gRPC Communication → ServerModel
                                ↓
   Client ← ModelTrainer (gradient + layer updates) ← gRPC Response ← ServerModel (backward pass)
   ```

3. **Aggregation**:
   ```
   Multiple Clients + Server Model Results → Strategy.aggregate_fit() → Weighted Parameter Averaging → Updated Global Model
   ```

### Layer Allocation Flow

1. **Client Registration**: Clients report minimum layer requirements via `get_properties()` using the `"__min_number_of_layers__"` key
2. **Epoch Setup**: Layer Allocator determines initial layer distribution based on client capabilities
3. **Runtime Adaptation**: Layer Allocator can trigger layer reallocation via `trigger_reallocation()` and `get_layers_diff()`
4. **Model Updates**: ModelTrainer handles dynamic layer changes through `_update_model()` method

### Communication Protocol

- **Stream Initialization**: Client establishes gRPC stream with `INIT_STREAM` control code
- **Forward Pass**: Client sends activations via `BatchDataForward` messages
- **Backward Pass**: Server responds with gradients via `BatchDataBackward` messages
- **Layer Transfer**: Additional layers can be sent via `additional_layers_parameters` in backward messages
- **Stream Closure**: Proper cleanup with `DO_CLOSE_STREAM` and `STREAM_CLOSED_OK` control codes

## User Implementation Guide

### What the Framework Handles

- **Model Splitting**: Automatic division of model layers between client and server based on client capabilities and layer allocator decisions
- **Communication**: gRPC-based message passing for activations and gradients using protocol buffers
- **Synchronization**: Coordination of training rounds and client-server interactions through the Strategy pattern
- **Layer Management**: Dynamic allocation and reallocation of layers via the LayerAllocator
- **Optimization**: Gradient computation and parameter updates (SGD with momentum by default)
- **Metrics Collection**: Training loss, accuracy, and timing metrics
- **Batch Processing**: Support for individual client processing or batched processing on the server
- **Parameter Aggregation**: Weighted averaging of parameters based on the number of training examples per layer

## Example Usage

The `testing/` folder provides a complete example:

- **Model**: Simple CNN for CIFAR-10 classification (`testing/model.py`)
- **Client**: Implements data loading and training logic (`testing/client.py`)
- **Server**: Sets up strategy and starts training (`testing/server.py`)
- **Usage**: Demonstrates both training and evaluation phases

**Note**: The testing example uses `"min_layers": 3` in the `get_properties()` method, but the framework expects the constant `"__min_number_of_layers__"` as defined in `slbd.common.constants`. This should be updated for consistency.

To run the example:
1. Start server: `python testing/server.py`
2. Start client(s): `python testing/client.py`

## Framework Strengths and Improvement Areas

### Strengths

1. **Flower Integration**: Leverages robust federated learning infrastructure
2. **Flexible Layer Allocation**: Supports dynamic layer distribution
3. **PyTorch Native**: Well-integrated with PyTorch ecosystem
4. **Modular Design**: Clean separation of concerns between components
5. **Extensible Strategy Pattern**: Easy to implement custom training strategies

### Areas for Improvement - What to do next?

#### 1. Layer Allocator Logic
**Current State**: So far we implemented only a very basic implementation for `LayerAllocator` with hardcoded reallocation logic (i.e., we do not really decide the number of layers based on training times).
```python
# Current simple reallocation in LayerAllocator
if self._per_client_processed_batches[client_id] == 5:
    self._client_layer_diff[client_id] = 3
```

#### 2. Model aggregation logic

Right now we simply average the provided weights, but this does not seem right as we should also consider *when* was a parameter last updated. Take a very simple example, in which we have a client training a model. At some point, the server sends one additional trainable layer to the client. After that point, the "offloaded" layer will no longer be trained on the server, but only on the client. At the end of training, we should therefore use *only* the version of the layer on the client and discard the one on the server. How should we handle such cases?

**Current Implementation**: The framework tracks the number of training examples processed by each layer (`trained_per_layer`) and uses weighted averaging based on these counts. The aggregation function in `PlainSlStrategy._aggregate_layer_parameters()` normalizes weights by total training examples, but there's an assertion that expects the total to be 1000, which seems hardcoded.

#### 3. Batch scheduling

Design a method to combine batches on the server in order to obtain the best possible utilization of the server GPU.


#### 4. Missing features

The following features still need to be implemeted:

1. Ability to configure optimizer by model owner.
2. Ability to preprocess data before sending it over the network (model owner should be able to specify a function, which manipulates data, e.g., sparsification, quantization, etc)
3. Give more flexibility in terms of model architecture. Right now, we have a nn.Sequential, but this cannot be used to train language models because it only supports a single input and output tensor, while language models may be passed additional parameters, such as attention mask.

## Conclusion

SplitBud provides a solid foundation for split learning applications with good integration with the Flower ecosystem. The modular architecture allows for extensibility, but there are opportunities to improve the sophistication of layer allocation, configuration management, and overall user experience. The framework is particularly well-suited for scenarios where data privacy is crucial and computational resources are distributed across edge devices.
