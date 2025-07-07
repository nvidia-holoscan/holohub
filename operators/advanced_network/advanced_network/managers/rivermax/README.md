# Rivermax Manager Implementation

This directory contains the complete implementation of the Rivermax Manager for the Advanced Network library. The Rivermax Manager provides high-performance network streaming capabilities using NVIDIA's Rivermax SDK, specifically designed for professional media and broadcast applications requiring ultra-low latency and high throughput.

## Architecture Overview

The Rivermax Manager implements a service-oriented architecture that abstracts the complexity of NVIDIA's Rivermax SDK while providing a unified interface through the Advanced Network library's Manager interface.

### High-Level Architecture

```mermaid
graph TB
    subgraph "Application Layer"
        A["Advanced Network Manager Interface<br/>(Common API)"]
    end
    
    subgraph "Rivermax Manager Implementation"
        B["RivermaxMgr<br/>(Main Manager Class)"]
        
        subgraph "Core Management Components"
            C["Configuration Management<br/>(rivermax_config_manager)"]
            D["Service Management<br/>(rivermax_mgr_service)"]
            E["Burst & Packet Processing<br/>(burst_manager)"]
        end
        
        subgraph "Implementation Details"
            F["Queue Configurations<br/>(rivermax_queue_configs)"]
            G["Data Types & Structures<br/>(rivermax_ano_data_types)"]
            H["Packet Processing<br/>(packet_processor)"]
        end
    end
    
    subgraph "Rivermax SDK (RDK) Services"
        subgraph "RX Services"
            I["IPO Receiver Service<br/>(rmax_ipo_receiver)"]
            J["RTP Receiver Service<br/>(rmax_rtp_receiver)"]
        end
        
        subgraph "TX Services"
            K["MediaSenderZeroCopyService<br/>(True Zero-Copy)"]
            L["MediaSenderService<br/>(Memory Pool)"]
            M["MediaSenderMockService<br/>(Testing)"]
        end
    end
    
    subgraph "Hardware Abstraction Layer"
        N["NVIDIA ConnectX NICs"]
        O["Rivermax Hardware Drivers"]
        P["RDMA & GPUDirect"]
    end
    
    %% Connections
    A --> B
    B --> C
    B --> D
    B --> E
    C --> F
    D --> I
    D --> J
    D --> K
    D --> L
    D --> M
    E --> G
    E --> H
    F --> I
    F --> J
    F --> K
    F --> L
    
    I --> N
    J --> N
    K --> N
    L --> N
    M --> N
    N --> O
    N --> P
    
    %% Styling
    classDef appLayer fill:#fff3e0
    classDef managerLayer fill:#e8f5e8
    classDef implLayer fill:#f3e5f5
    classDef rdkLayer fill:#e3f2fd
    classDef hardwareLayer fill:#e1f5fe
    
    class A appLayer
    class B,C,D,E managerLayer
    class F,G,H implLayer
    class I,J,K,L,M rdkLayer
    class N,O,P hardwareLayer
```

## Directory Structure and File Roles

### Root Level Files

#### `adv_network_rivermax_mgr.h/cpp`
- **Purpose**: Main manager class implementing the Advanced Network Manager interface
- **Role**: Entry point for all Advanced Network operations, coordinates service lifecycle
- **Key Responsibilities**:
  - Manager initialization and configuration
  - Service creation and management
  - Burst and packet API implementation
  - Memory management coordination

#### `rivermax_ano_data_types.h`
- **Purpose**: Core data structures and type definitions for Rivermax operations
- **Role**: Defines the fundamental data types used throughout the manager
- **Key Components**:
  - `RivermaxBurst` - Packet burst container with metadata
  - `AnoBurstsQueue` - Thread-safe queue for burst management
  - `RivermaxPacketData` - Individual packet data structure
  - Extended info structures for packet metadata

#### `CMakeLists.txt`
- **Purpose**: Build configuration for the Rivermax manager
- **Role**: Defines compilation dependencies and linking requirements
- **Dependencies**: Rivermax SDK, Advanced Network common libraries

### Implementation Directory (`rivermax_mgr_impl/`)

This directory contains the core implementation components that handle the detailed functionality of the Rivermax manager.

#### Configuration Management

##### `rivermax_config_manager.h/cpp`
- **Purpose**: Comprehensive configuration parsing and validation
- **Role**: Translates Advanced Network configurations to Rivermax-specific settings
- **Key Components**:
  - `ConfigBuilderContainer` - Manages multiple configuration builders
  - `RxConfigManager` - Handles receive queue configurations
  - `TxConfigManager` - Handles transmit queue configurations
  - Configuration validation and memory region setup

##### `rivermax_queue_configs.h/cpp`
- **Purpose**: Queue-specific configuration builders and settings
- **Role**: Creates and manages Rivermax service configurations
- **Key Components**:
  - IPO Receiver configuration builders
  - RTP Receiver configuration builders  
  - Media Sender configuration builders
  - Memory allocator configuration

#### Service Management

##### `rivermax_mgr_service.h/cpp`
- **Purpose**: Service abstraction layer for Rivermax applications
- **Role**: Manages the lifecycle of Rivermax SDK services
- **Key Components**:
  - `RivermaxManagerService` - Base service interface
  - `RivermaxManagerRxService` - Base receiver service
  - `RivermaxManagerTxService` - Base transmitter service
  - `IPOReceiverService` - IPO protocol receiver implementation
  - `RTPReceiverService` - RTP protocol receiver implementation
  - `MediaSenderService` - Media transmission service with memory pool management
  - `MediaSenderZeroCopyService` - Zero-copy media transmission service

#### Packet and Burst Processing

##### `burst_manager.h/cpp`
- **Purpose**: Manages packet bursts and memory allocation
- **Role**: Handles burst lifecycle, memory management, and packet pointer organization
- **Key Components**:
  - `RivermaxBurst` - Enhanced burst implementation with `AnoBurstExtendedInfo` metadata
  - `RxBurstsManager` - Manages received packet bursts with pointer-based architecture
  - Burst allocation and deallocation strategies
  - Packet pointer management (no data copying at burst level)

##### `packet_processor.h`
- **Purpose**: Packet-level processing and transformation
- **Role**: Handles individual packet operations and metadata processing
- **Key Components**:
  - Packet validation and filtering
  - Metadata extraction and processing
  - Protocol-specific packet handling

#### Core Implementation

##### `adv_network_rivermax_mgr.cpp`
- **Purpose**: Main manager implementation
- **Role**: Implements all Manager interface methods using Rivermax services
- **Key Functions**:
  - Service initialization and coordination
  - Burst and packet API implementations
  - Memory management and buffer allocation
  - Statistics collection and reporting

##### `rivermax_chunk_consumer_ano.h`
- **Purpose**: Chunk-based data consumption interface
- **Role**: Provides an interface for consuming data chunks from Rivermax streams
- **Integration**: Used by the burst manager for efficient data handling

### Service Directory (`rivermax_service/`)

#### `CMakeLists.txt`
- **Purpose**: Build configuration for Rivermax service components
- **Role**: Manages additional service-specific build requirements
- **Integration**: Extends the main build configuration with service-specific dependencies

## External Services and Dependencies

### NVIDIA Rivermax SDK (RDK)

The manager relies heavily on the Rivermax Development Kit, which provides:

> **Note**: For more information about the Rivermax Development Kit, see the official repository: [NVIDIA Rivermax Dev Kit](https://github.com/NVIDIA/rivermax-dev-kit). The Rivermax Dev Kit is a high-level C++ software kit designed to accelerate and simplify Rivermax application development, offering intuitive abstractions and developer-friendly services for IP-based media and data streaming use cases.

#### Core Services
- **`rmax_ipo_receiver`**: An RX service for receiving RTP data using Rivermax Inline Packet Ordering (IPO) feature
- **`rmax_rtp_receiver`**: An RX service for receiving RTP data using Rivermax Striding protocol
- **`rmax_xstream_media_sender`**: High-performance media transmission service with multiple variants:
  - `MediaSenderZeroCopyService`: True zero-copy path using application frame buffers directly
  - `MediaSenderService`: Single-copy path with internal memory pool for generated data

#### Key Capabilities
- **Hardware Acceleration**: Direct NIC hardware access for minimal latency
- **Memory Management**: Advanced memory allocation strategies (huge pages, GPU memory)
- **Protocol Support**: SMPTE 2110, RTP, custom protocols
- **Timing Control**: Precise packet timing for broadcast applications

### Hardware Dependencies

#### NVIDIA ConnectX NICs
- **ConnectX-6 or later**: Required for Rivermax functionality
- **Hardware Features**: RDMA, GPUDirect, hardware timestamping
- **Driver Requirements**: MOFED drivers for full functionality

#### GPU Integration
- **GPUDirect Support**: Zero-copy GPU-to-NIC data paths
- **Memory Management**: Device memory allocation and management
- **Compute Integration**: GPU-based packet processing capabilities

## Operational Flow

### Initialization Sequence

The Rivermax manager follows a carefully orchestrated initialization process to ensure optimal performance and proper resource allocation:

```mermaid
flowchart TD
    A["Application Startup"] --> B["RivermaxMgr Construction"]
    B --> C["YAML Configuration Loading"]
    
    C --> D["Configuration Parsing<br/>(rivermax_config_manager)"]
    D --> E["Queue Configuration Building<br/>(rivermax_queue_configs)"]
    E --> F["Memory Region Validation"]
    
    F --> G{{"Service Type Detection"}}
    G -->|RX Queue| H["RX Service Configuration"]
    G -->|TX Queue| I["TX Service Configuration"]
    
    H --> J["RX Service Creation"]
    I --> K["TX Service Creation"]
    
    J --> L{{"RX Service Type"}}
    L -->|settings_type: ipo_receiver| M["IPOReceiverService<br/>Instantiation"]
    L -->|settings_type: rtp_receiver| N["RTPReceiverService<br/>Instantiation"]
    
    K --> O{{"TX Service Type"}}
    O -->|use_internal_memory_pool: false| P["MediaSenderZeroCopyService<br/>Instantiation"]
    O -->|use_internal_memory_pool: true| Q["MediaSenderService<br/>Instantiation"]
    O -->|dummy_sender: true| R["MediaSenderMockService<br/>Instantiation"]
    
    M --> S["Memory Region Allocation<br/>(CPU/GPU Buffers)"]
    N --> S
    P --> S
    Q --> S
    R --> S
    
    S --> T["Hardware Initialization<br/>(ConnectX NIC Setup)"]
    T --> U["Service Launch<br/>(Background Threads)"]
    U --> V["Ready for Operations<br/>(Burst Processing)"]
    
    %% Styling
    classDef initPhase fill:#e3f2fd
    classDef configPhase fill:#f3e5f5
    classDef servicePhase fill:#e8f5e8
    classDef hardwarePhase fill:#e1f5fe
    classDef readyPhase fill:#e8f5e8
    
    class A,B,C initPhase
    class D,E,F configPhase
    class G,H,I,J,K,L,M,N,O,P,Q,R servicePhase
    class S,T hardwarePhase
    class U,V readyPhase
```

#### Detailed Initialization Steps

1. **Configuration Parsing**: Parse YAML configuration into Rivermax-specific settings using ConfigBuilderContainer
2. **Service Creation**: Instantiate appropriate Rivermax services based on queue configuration and service types
3. **Memory Setup**: Allocate and configure memory regions (CPU/GPU) with proper alignment and access patterns
4. **Hardware Initialization**: Initialize NIC hardware, establish RDMA connections, and configure hardware queues
5. **Service Launch**: Start background services for packet processing with proper CPU affinity and threading

### Receive Path (RX)

The RX path implements a sophisticated pointer-based architecture for zero-copy packet processing:

```mermaid
flowchart TD
    A["Network Packets<br/>(ConnectX NIC)"] --> B["Hardware DMA<br/>(Direct Memory Access)"]
    B --> C["Pre-allocated Memory Regions<br/>(Host/Device Memory)"]
    
    C --> D{{"RDK Service Selection"}}
    D -->|settings_type: ipo_receiver| E["IPO Receiver Service<br/>(rmax_ipo_receiver)"]
    D -->|settings_type: rtp_receiver| F["RTP Receiver Service<br/>(rmax_rtp_receiver)"]
    
    E --> G["Packet Reception<br/>(Hardware Acceleration)"]
    F --> G
    
    G --> H["RivermaxMgr Processing<br/>(burst_manager)"]
    H --> I["Burst Assembly<br/>(Packet Pointers Only)"]
    
    I --> J["AnoBurstExtendedInfo<br/>Metadata Creation"]
    J --> K["RivermaxBurst Container<br/>(Pointers + Metadata)"]
    
    K --> L{{"Header-Data Split<br/>(HDS) Configuration"}}
    L -->|hds_on: true| M["Headers: CPU Memory<br/>burst->pkts[0]<br/>Payloads: GPU Memory<br/>burst->pkts[1]"]
    L -->|hds_on: false| N["Headers + Payloads<br/>CPU Memory<br/>burst->pkts[0] + offset"]
    
    M --> O["AnoBurstsQueue<br/>(Thread-safe Distribution)"]
    N --> O
    
    O --> P["Application Consumption<br/>(get_rx_burst)"]
    P --> Q["Burst Processing<br/>(Advanced Network Media)"]
    Q --> R["Pointer Lifecycle Tracking<br/>(free_all_packets_and_burst_rx)"]
    
    %% Zero-copy emphasis
    S["Zero-Copy Architecture<br/>Emphasis"] -.-> I
    S -.-> K
    S -.-> O
    T["No Data Copying<br/>Only Pointer Management"] -.-> S
    
    %% Styling
    classDef hardwareLayer fill:#e1f5fe
    classDef rdkLayer fill:#e3f2fd
    classDef managerLayer fill:#f3e5f5
    classDef burstLayer fill:#e8f5e8
    classDef appLayer fill:#fff3e0
    classDef zeroCopyEmphasis fill:#e8f5e8,stroke:#4caf50,stroke-width:3px
    
    class A,B,C hardwareLayer
    class D,E,F,G rdkLayer
    class H,I,J,K,L,M,N,O managerLayer
    class P,Q,R appLayer
    class S,T zeroCopyEmphasis
```

#### Key RX Path Characteristics

1. **Packet Reception**: Rivermax service receives packets from NIC hardware directly into pre-allocated memory regions
2. **Burst Assembly**: Packet **pointers** are aggregated into `RivermaxBurst` containers with `AnoBurstExtendedInfo` metadata (no data copying)
3. **Metadata Extraction**: Extract timing, flow, and protocol information from packet headers
4. **HDS Configuration**: Headers and payloads placed in CPU/GPU memory based on Header-Data Split settings
5. **Queue Management**: Bursts (containing pointers) are queued for application consumption via `AnoBurstsQueue`
6. **Memory Management**: Coordinate buffer allocation, cleanup, and pointer lifecycle tracking

### Transmit Path (TX)

The TX path implements frame-level processing with configurable memory management strategies:

```mermaid
flowchart TD
    A["Application<br/>(MediaFrame Input)"] --> B["BurstParams Creation<br/>(custom_pkt_data attachment)"]
    B --> C["RivermaxMgr Processing<br/>(Service Coordination)"]
    
    C --> D{{"Service Selection<br/>(Configuration Driven)"}}
    D -->|use_internal_memory_pool: false| E["MediaSenderZeroCopyService<br/>(True Zero-Copy Path)"]
    D -->|use_internal_memory_pool: true| F["MediaSenderService<br/>(Single Copy + Pool Path)"]
    D -->|dummy_sender: true| G["MediaSenderMockService<br/>(Testing Mode)"]
    
    %% Zero-Copy Path
    E --> H["No Internal Memory Pool"]
    H --> I["Direct Frame Reference<br/>(custom_pkt_data → RDK)"]
    I --> J["RDK MediaSenderApp<br/>(Zero-Copy Processing)"]
    J --> K["Frame Ownership Transfer<br/>& Release After Processing"]
    
    %% Memory Pool Path
    F --> L["Pre-allocated MediaFramePool<br/>(MEDIA_FRAME_POOL_SIZE)"]
    L --> M["Single Memory Copy<br/>(Frame → Pool Buffer)"]
    M --> N["RDK MediaSenderApp<br/>(Pool Buffer Processing)"]
    N --> O["Pool Buffer Reuse<br/>(Returned to Pool)"]
    
    %% Mock Path
    G --> P["Mock Processing<br/>(Testing/Development)"]
    
    %% Common RDK Processing
    J --> Q["RDK Internal Processing<br/>(All Packet Operations)"]
    N --> Q
    P --> Q
    
    Q --> R["RTP Packetization<br/>(SMPTE 2110 Standard)"]
    R --> S["Protocol Headers<br/>& Metadata Addition"]
    S --> T["Precise Timing Control<br/>& Scheduling"]
    T --> U["Hardware Submission<br/>(ConnectX NIC)"]
    U --> V["Network Transmission"]
    
    %% Performance Paths
    W["Zero-Copy Performance<br/>Minimum Latency"] -.-> E
    X["Memory Pool Performance<br/>Sustained Throughput"] -.-> F
    
    %% Styling
    classDef appLayer fill:#fff3e0
    classDef managerLayer fill:#f3e5f5
    classDef zeroCopyPath fill:#e8f5e8,stroke:#4caf50,stroke-width:3px
    classDef poolPath fill:#fff8e1,stroke:#ff9800,stroke-width:3px
    classDef mockPath fill:#f5f5f5,stroke:#757575,stroke-width:2px
    classDef rdkLayer fill:#e3f2fd
    classDef networkLayer fill:#e1f5fe
    classDef performanceEmphasis fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    
    class A,B appLayer
    class C,D managerLayer
    class E,H,I,J,K zeroCopyPath
    class F,L,M,N,O poolPath
    class G,P mockPath
    class Q,R,S,T rdkLayer
    class U,V networkLayer
    class W,X performanceEmphasis
```

#### Key TX Path Characteristics

1. **Burst Preparation**: Applications provide frame data in burst format with MediaFrame references via `custom_pkt_data`
2. **Service Selection**: Route to appropriate MediaSender service based on configuration:
   - `MediaSenderZeroCopyService`: Direct frame buffer access (zero-copy, minimum latency)
   - `MediaSenderService`: Copy to internal memory pool (single-copy, sustained throughput)
   - `MediaSenderMockService`: Testing mode with minimal functionality
3. **RDK Processing**: All packet formatting, RTP headers, and protocol handling performed by RDK services
4. **Timing Control**: Apply precise transmission timing within RDK for SMPTE 2110 compliance
5. **Hardware Submission**: Submit formatted packets to NIC hardware via Rivermax acceleration
6. **Resource Cleanup**: Free frame buffers and update statistics based on service type


### Memory Architecture Components

#### Memory Region Types
- **Host Memory**: Standard system memory for headers and control data
- **Huge Pages**: Large page allocations for improved performance and reduced TLB misses
- **Device Memory**: GPU memory for zero-copy operations and GPU-accelerated processing
- **Pinned Memory**: Host memory accessible to GPU and NIC for efficient DMA transfers

#### Buffer Management Strategy
- **Pre-allocation**: Buffers are allocated during initialization to avoid runtime allocation overhead
- **Pool Management**: Efficient buffer reuse through pool allocation reduces memory fragmentation
- **Lifecycle Tracking**: Careful buffer ownership and cleanup management prevents memory leaks
- **Burst Metadata**: `AnoBurstExtendedInfo` structure carries configuration details:
  - **HDS Configuration**: `hds_on`, `header_stride_size`, `payload_stride_size`
  - **Memory Location Flags**: `header_on_cpu`, `payload_on_cpu`
  - **Segment Indices**: `header_seg_idx`, `payload_seg_idx` for memory region mapping

#### HDS Memory Layout Optimization
- **HDS Enabled**: Headers stored in CPU memory, payloads in GPU memory for optimal processing
- **HDS Disabled**: Headers and payloads stored together in CPU memory with calculated offsets

## Performance Optimizations

The Rivermax manager implements multiple layers of performance optimization for ultra-low latency streaming:

### Multi-Layer Optimization Strategy

#### Zero-Copy Architecture
- **Pointer-Based Bursts**: Bursts contain only pointers to packet data, no copying at burst level
- **Direct Memory Access**: Minimize data copying between components through DMA and pointer management
- **GPU Integration**: Direct GPU-to-NIC data paths where supported through GPUDirect
- **Frame Reference Management**: MediaFrame objects reference original application buffers (TX zero-copy path)
- **Buffer Sharing**: Efficient buffer sharing between services and applications through pointer management

#### Memory Copy Optimization
- **Strategy-Based Processing**: RX path uses optimal copy strategies (contiguous/strided) based on memory layout
- **Adaptive Strategy Selection**: Runtime detection of optimal copy strategy based on buffer alignment
- **Single Copy Principle**: When copying is necessary, ensure only one copy operation per data path

### Thread Management
- **Service Threads**: Dedicated threads for each Rivermax service
- **CPU Affinity**: Thread pinning to isolated CPU cores
- **Lock-Free Queues**: High-performance inter-thread communication

### Hardware Optimization
- **Batch Processing**: Process multiple packets per operation through burst containers
- **Hardware Queues**: Utilize multiple NIC queues for parallelism
- **Interrupt Mitigation**: Reduce interrupt overhead through batching
- **Memory Copy Optimization**: 
  - Contiguous packets: Single `cudaMemcpy` operation
  - Strided packets: Optimized `cudaMemcpy2D` with detected stride parameters
  - Adaptive strategy selection based on runtime memory layout analysis

## Configuration Integration

The Rivermax manager integrates seamlessly with the Advanced Network configuration system:

### Supported Configurations
- **Network Interfaces**: Multiple NIC configuration support
- **Memory Regions**: Flexible memory allocation strategies
- **Queue Settings**: Per-queue configuration with protocol-specific options
- **Protocol Parameters**: RTP, IPO, and media streaming parameters

### Validation and Error Handling
- **Configuration Validation**: Comprehensive parameter validation
- **Runtime Monitoring**: Service health monitoring and error reporting
- **Graceful Degradation**: Fallback strategies for partial failures

## Future Extensibility

The architecture is designed for future enhancements:

### Protocol Support
- Additional protocol implementations can be added through the service interface
- New Rivermax SDK features can be integrated without changing the manager interface

### Hardware Support
- Support for new ConnectX generations through configuration updates
- Enhanced GPU integration as hardware capabilities evolve

### Performance Enhancements
- Additional optimization strategies can be implemented in the service layer
- New memory management techniques can be integrated transparently 