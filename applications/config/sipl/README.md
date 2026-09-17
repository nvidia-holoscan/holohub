# SIPL platform configs

Vendor JSON platform configs for the Leopard Imaging VB1940 Eagle camera,
shared across the SIPL apps (`--json-config` / `SIPL_JSON_CONFIG`) and the
SIPL operator's hardware tests. Ingested directly by the NvSIPL camera stack
(`SIPLCaptureService`'s `json_config` argument) — this project does not parse
or transform them. Each file adheres to the SIPL Query JSON schema; see the
[SIPL JSON Query Guide](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/CameraDevelopment/SIPLFramework/SIPL-for-L4T/SIPL-JSON-Query-Guide.html#coe-camera-development).

| File | Cameras | Source |
| --- | --- | --- |
| `vb1940_single.json` | One, independent | Holoscan Sensor Bridge, `examples/sipl_config/vb1940_single.json` at `7b12310` |
| `vb1940_dual.json` | Two, independent (not hardware-synced) | Holoscan Sensor Bridge, `examples/sipl_config/vb1940_dual.json` at `7b12310` |
| `vb1940_stereo.json` | Two, hardware-synced (`isStereo`, `sensorGroup`, `sync_sensors`) | Derived here from `vb1940_dual.json` for [sipl_stereo_monitor](../../sipl_stereo_monitor) |

`vb1940_single.json` and `vb1940_dual.json` are byte-identical to their
originals in [Holoscan Sensor Bridge](https://gitlab-master.nvidia.com/holoscan/hololink)
(`examples/sipl_config/`) as of commit `7b12310d54cef00c759beaa02fbdd168bffb4bdb`
("SIPL JP7.2 and backward compatibility with JP7.1"). `vb1940_stereo.json` has
no HSB counterpart as of that commit; it adds the stereo-sync fields
`vb1940_dual.json` doesn't set. HSB is expected to ship its own stereo config
in a future release (2.8/3.0) — if so, prefer that as the source of truth and
update this file to match rather than maintaining it independently.
