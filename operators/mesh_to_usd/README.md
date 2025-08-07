# Mesh to USD Operator

The `SendMeshToUSDOp` operator converts 3D meshes in STL format to USD (Universal Scene Description) mesh format for integration into OpenUSD scenes.

## Overview

This operator takes STL mesh data (either from a file path or byte stream) and converts it to USD geometry format. It supports both file-based and in-memory STL data processing, making it flexible for various pipeline configurations.

## Features

- **STL to USD Conversion**: Converts STL mesh files to USD mesh format
- **Dual Input Support**: Accepts STL data from file paths or byte streams
- **OpenUSD Integration**: Seamlessly integrates with existing USD scenes
- **Mesh Properties**: Preserves mesh points, normals, and face information
- **Component Hierarchy**: Sets appropriate USD component kind for assembly selection

## Usage

### Basic Usage with File Path

```python
from holoscan.operators import SendMeshToUSDOp

# Create operator with STL file path
mesh_op = SendMeshToUSDOp(
    fragment=fragment,
    stl_file_path="/path/to/mesh.stl",
    g_stage=usd_stage
)
```

### Usage with Byte Stream Input

```python
# Create operator for byte stream input
mesh_op = SendMeshToUSDOp(
    fragment=fragment,
    g_stage=usd_stage
)

# Connect to upstream operator that provides STL bytes
# The input name is "stl_bytes"
```

## Parameters

- **`stl_file_path`** (optional): Path to STL file to convert
- **`g_stage`**: Existing USD stage where the mesh will be added

## Input/Output

### Input
- **`stl_bytes`** (optional): Byte stream containing STL mesh data
  - Required when `stl_file_path` is not provided
  - Condition: `ConditionType.NONE` (optional input)

### Output
- **USD Mesh**: Creates a USD mesh primitive in the specified stage
  - Mesh name: "mesh" (automatically made valid identifier)
  - Location: Under the stage's default prim path
  - Properties: Points, normals, face vertex indices, face vertex counts
  - Kind: Component (for assembly hierarchy support)

## Mesh Processing

The operator performs the following conversions:

1. **STL Parsing**: Reads STL file or byte stream using numpy-stl library
2. **Point Conversion**: Extracts vertex coordinates and converts to USD format
3. **Normal Processing**: Preserves surface normals from STL data
4. **Face Generation**: Creates triangular faces with proper indexing
5. **Extent Calculation**: Computes bounding box for the mesh
6. **USD Integration**: Adds mesh to existing USD stage with proper metadata

## File Format Support

- **Input**: STL (Stereolithography) format
- **Output**: USD (Universal Scene Description) mesh format
- **Compatibility**: Works with OpenUSD pipeline and tools

## Integration

The operator is designed to work within Holoscan pipelines and can be connected to:
- STL file readers or generators
- USD stage management operators
- Visualization and rendering systems
- 3D processing workflows 