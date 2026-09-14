# Data Distribution Service Operators

These operators connect Holoscan graphs to RTI Connext DDS. Start with the
[RTI Connext DDS Module overview](../../modules/holoscan-connext-dds/README.md)
for the Connext introduction, supported versions, container workflow, and
license setup.

Each operator keeps its ports, parameters, behavior, and limitations in its
own README:

| Operator | Language | Documentation |
| --- | --- | --- |
| `ConnextDDSPublisherOp` | Python | [Connext DDS publisher](dds_pubsub/publisher/README.md) |
| `ConnextDDSSubscriberOp` | Python | [Connext DDS subscriber](dds_pubsub/subscriber/README.md) |
| `DDSShapesSubscriberOp` | C++ | [DDS Shapes subscriber](dds_shapes_subscriber/README.md) |
| `DDSVideoPublisherOp` | C++ | [DDS video publisher](video/dds_video_publisher/README.md) |
| `DDSVideoSubscriberOp` | C++ | [DDS video subscriber](video/dds_video_subscriber/README.md) |
| `DDSOperatorBase` | C++ | [Shared DDS base](base/README.md) |

The [DDS applications](../../applications/dds/README.md) demonstrate the
operators in generic Python and C++ video workflows. Build and run them through
HoloHub so that Connext, Holoscan, and all supporting dependencies remain
inside the module container.
