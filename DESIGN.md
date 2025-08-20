# Design of DynSL

In this file we discuss the motivations for the design decisions we made.

## Batch Scheduling for Dynamic Split Learning

### Problem Statement

In DynSL, we face a scheduling challenge due to clients dynamically offloading different portions of the computational graph to the server. This heterogeneity creates opportunities for optimization but also introduces complexity in resource utilization.

**Example Scenario**: Consider a 10-layer model with two clients:
- Client A: trains layers 1-3 locally → server processes layers 4-10 (7 layers)
- Client B: trains layers 1-6 locally → server processes layers 7-10 (4 layers)

The server must efficiently handle batches requiring different computational paths while maximizing GPU utilization and maintaining low latency.

### Scheduling Strategies Analysis

A preliminary qualitative analysis reveals three primary approaches:

#### 1. Per-Client Processing (Immediate Processing)

**Description**: Process each client batch individually as soon as it arrives, without any grouping or batching.

**Pros**:
- **Simplicity**: Straightforward implementation with minimal coordination logic
- **Low latency**: Immediate processing when no other batches are queued
- **Predictable behavior**: Each client experiences consistent processing patterns

**Cons**:
- **Poor GPU utilization**: Server GPU is constrained by client batch sizes (typically 16-128 samples), leading to underutilization on powerful server hardware
- **Incompatible with privacy mixing**: Cannot shuffle data across clients, making it possible for the server to associate embeddings with specific clients, compromising privacy
- **Suboptimal throughput**: Small batch sizes result in inefficient compute utilization

#### 2. Heterogeneous Batch Grouping (Arbitrary Batching)

**Description**: Group consecutive batches from different clients regardless of their layer requirements, then process overlapping computational paths sequentially.

**Implementation approach**:
1. Collect batches until reaching a target group size (e.g., 2 batches)
2. Process non-overlapping layers separately for each subset
3. Concatenate intermediate results for shared layers
4. Continue processing remaining shared layers on combined data

**Pros**:
- **Improved resource utilization**: Larger effective batch sizes through concatenation
- **Maintains processing for all clients**: No clients are starved waiting for similar requests

**Cons**:
- **High implementation complexity**: Requires sophisticated batch splitting, intermediate result management, and computational graph coordination
- **Partial utilization**: Some layers still process small batches during the sequential phase
- **Debugging complexity**: Error handling and profiling become significantly more challenging

#### 3. Homogeneous Batch Grouping (Layer-Aligned Batching)

**Description**: Group batches that require identical server-side layer processing, enabling direct concatenation and unified processing.

**Implementation approach**:
1. Maintain separate queues for each unique layer configuration
2. Wait until sufficient batches accumulate for a configuration
3. Concatenate batches directly (with optional shuffling for privacy)
4. Process the combined batch through the required server layers

**Pros**:
- **Optimal GPU utilization**: Full batch processing with predictable computational paths
- **Privacy preservation**: Enables cross-client data shuffling to prevent embedding-client association
- **Implementation simplicity**: Clean batch concatenation without complex graph management
- **Predictable performance**: Consistent processing patterns for each layer configuration

**Cons**:
- **Increased latency**: Clients may wait for sufficient batches to accumulate before processing begins (could be mitigated with timeouts, i.e., if a client batch is waiting for too long, process it even if it does not have the target number of client batches)
- **Potential starvation**: Uncommon layer configurations might experience longer delays


### Recommendation: Adaptive Homogeneous Batch Grouping

Overall it seems that **Strategy #3 (Homogeneous Batch Grouping)** is the most promising, thus we implement it in the framework. Additionally, we consider several enhancements to address its limitations.

#### Core Strategy
- Maintain separate queues for each layer configuration
- Group batches with identical server-side processing requirements
- Enable cross-client data shuffling for deidentification

#### Adaptive Enhancements

1. **Dynamic Timeout Mechanism**:
   - Implement configurable timeouts for each queue to prevent excessive latency
   - Process partial batches when timeout expires, balancing latency vs. utilization

   This is implemented in `ClientRequestGroup`: whenever a new `ClientRequestGroup` is created, a `asyncio` task is created which waits for a predetermined amount of time. At the end of this time period, if the request group has not been marked as ready yet, it marks the request group as ready.

2. **Intelligent Batch Size Targets**:
   - Set target batch sizes based on GPU characteristics and layer complexity
   - Smaller targets for complex layers (already computationally intensive)
   - Larger targets for lightweight layers (need more parallelism)

3. **Preemptive Processing**:
   - For rare configurations, process immediately rather than waiting
   - Use historical data to identify and prioritize common vs. rare patterns

4. **Hybrid Fallback**:
   - Fall back to per-client processing during low-activity periods
   - Maintain simple processing paths as backup when grouping queues are empty

#### Implementation Considerations

1. **Queue Management**:
   ```
   Layer Config → Queue → [Batch1, Batch2, ...] → Timeout/Size Trigger → Process
   ```

2. **Memory Management**:
   - Implement bounded queues to prevent memory exhaustion
   - Spill to disk for long-waiting batches if necessary

3. **Monitoring and Metrics**:
   - Track per-configuration queue lengths, wait times, and processing efficiency
   - Adaptive parameter tuning based on real-time performance metrics

4. **Graceful Degradation**:
   - Automatic fallback mechanisms when batching introduces excessive latency
   - Configurable maximum wait times per client or globally

### Weak Points and Concerns in Current Analysis

1. **Missing Quantitative Analysis**: The current analysis lacks specific metrics for comparison (latency percentiles, throughput numbers, GPU utilization percentages)

2. **No Discussion of Fairness**: Important to consider how different strategies affect fairness between clients with different layer configurations

3. **Limited Scalability Considerations**: The analysis doesn't address how strategies perform with varying numbers of clients or layer configurations

4. **Insufficient Memory Analysis**: Missing discussion of memory requirements and potential bottlenecks for buffering batches

5. **No Error Handling**: Real-world considerations like client failures, network issues, or processing errors are not addressed

6. **Configuration Complexity**: The relationship between batch sizes, timeouts, and system performance needs more detailed exploration

The enhanced Strategy #3 with adaptive features provides the best balance of implementation complexity, performance optimization, and system robustness for your DynSL framework.
