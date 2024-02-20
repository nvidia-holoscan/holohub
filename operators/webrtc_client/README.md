### WebRTC Client Operator

The `webrtc_client` operator receives video frames through a WebRTC connection. The application using this operator needs to call the `offer` method of the operator when a new WebRTC connection is available.

### Methods

- **`async def offer(self, sdp, type) -> (local_sdp, local_type)`**
  Start a connection between the local computer and the peer.

  **Parameters**
  - **sdp** peer Session Description Protocol object
  - **type** peer session type

  **Return values**
  - **sdp** local Session Description Protocol object
  - **type** local session type

### Outputs

- **`output`**: Tensor with 8 bit per component RGB data
  - type: `Tensor`
