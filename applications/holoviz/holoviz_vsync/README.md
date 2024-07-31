# Holoviz vsync

![](holoviz_vsync.png)<br>
This application demonstrates the capability of the Holoviz operator to wait for the vertical blank of the display before updating the current image. It prints the displayed frames per second to the console, if sync to vertical blank is enabled the frames per second are capped to the display refresh rate.

To enable syncing to vertical blank set the `vsync` parameter of the Holoviz operator to `true`:

```cpp
    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz",
        // enable synchronization to vertical blank
        Arg("vsync", true));
```

By default, the Holoviz operator is not syncing to the vertical blank of the display.

## Run Instructions

To build and start the application:

```bash
./dev_container build_and_run holoviz_vsync
```
