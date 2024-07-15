# HoloHub Release Process

HoloHub is a constantly expanding collection of community projects. This guide aims to clarify how the HoloHub rolling release process relates to Holoscan SDK formal releases.

This guide is targeted at HoloHub maintainers or any community developers who intend to consume any version of HoloHub that does not use the latest Holoscan SDK version.

## HoloHub Releases

HoloHub accepts continuous contributions from community developers. The latest commit to the `main` branch is also considered the latest HoloHub "release". We do not tag or branch HoloHub for general purpose releases, with only narrow exceptions for long-term support as described later in this document.

HoloHub versioning is typically maintained at the community project level. Each community project is required to have a [`metadata.json`](/CONTRIBUTING.md#metadata-description) file that specifies a project version, among other attributes. Project maintainers are encouraged to advance their project versioning over time to reflect the addition of new features or fixes. We recommend users refer to Git history for a given [`metadata.json`](/CONTRIBUTING.md#metadata-description) file in order to review versions and updates over time for any given project.

Software Quality Assurance (SQA) is not available for any given HoloHub release. A limited set of HoloHub applications currently undergo regular nightly testing. We plan to make more details available on nightly testing at a later date.

## Holoscan SDK and HoloHub

[Holoscan SDK](https://github.com/nvidia-holoscan/holoscan-sdk) follows a regular release process. At the time of writing Holoscan SDK follows an approximately monthly release cadence.

We encourage HoloHub community members to use the latest Holoscan SDK version in new contributions and development. HoloHub has no general requirement for backwards compatibility with any specific Holoscan SDK version, though contributors may choose to provide it. Contributors should maintain tested and targeted Holoscan SDK versions in their application `metadata.json` file.

We generally request that applications are up to date within no greater than three major versions of the latest Holoscan SDK release. For instance, if the latest Holoscan SDK is release `4.0.0` and the HoloHub application complies only with Holoscan SDK release `0.5.0`, we may consider the application out of compliance.

## Holoscan SDK Long-Term Compatibility

Holoscan SDK provides long-term production branch support as part of the [NVIDIA AI Enterprise (NVAIE)](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/) program. Refer to NVIDIA for the latest information regarding available software programs.

HoloHub does not directly participate in the NVIDIA AI Enterprise program. However, we pursue two strategies for long-term HoloHub maintenance in reflection of that upstream program.

### Curated Backwards Compatible Applications

NVIDIA maintains a curated subset of first-party applications in HoloHub that represent a good basis for getting started developing with Holoscan SDK. The list of applications specifically maintained for backwards compatibility extends to the following:
- [`body_pose_estimation`](/applications/body_pose_estimation/README.md)
- [`endoscopy_tool_tracking`](/applications/endoscopy_tool_tracking/README.md)
- [`multiai_ultrasound`](/applications/multiai_ultrasound/README.md)
- [`volume_rendering`](/applications/volume_rendering/README.md)

### Short Term Support

While not guaranteed, we generally aim to maintain backwards compatibility for the set of applications listed above for up to nine months. That means that you can use the HoloHub applications listed above with any Holoscan SDK release within the past nine months.

The sample applications above currently support backwards compatibility with the following versions:
- Holoscan SDK v1.0.3 (general availability)
- Holoscan SDK v2.0.0 (general availability)

### Long Term Support

We may test various long-term supported Holoscan SDK branch versions with HoloHub. We will publicly tag the HoloHub version used to assure long-term whole stack compliance.

If you are an NVIDIA AI Enterprise participant, we recommend that you refer to the tagged commit in Git for your program. Check the `metadata.json` file for a given application to determine compatibility.
