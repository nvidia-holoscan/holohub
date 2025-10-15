# Copilot Instructions for HoloHub Repository

## Category Validation

When reviewing Pull Requests that modify or add `metadata.json` files, verify that the **category** (first tag in the `tags` array) of those `metadata.json` files is correctly set and matches one of the existing categories below.

### Validation Steps

1. **Locate metadata.json files**: Check if the PR adds or modifies any `metadata.json` files
2. **Extract tags**: Look for the `tags` field within the `application`, `operator`, `tutorial`, `benchmark`, `workflow`, or `gxf_extension` object, and extract the first tag as the category.
3. **Compare against approved list**: Verify the category exists in the "Approved Category List" below
4. **Flag discrepancies**: If the category is not in the approved list:
   - Comment on the PR indicating which tags are not recognized
   - If the there is a similar category in the approved list, suggest that the contributor use that category instead.
   - If there is any README.md file in the same directory as the `metadata.json` file, suggest an approved category based on the content of the README.md file.
   - Ask the contributor to either use an existing category or justify why a new category should be added.
   - Use "Example Review Comment" as a guideline on how to provide feedback.

### Approved Category List

Below is the complete list of categories currently approved for use in `metadata.json` files:

- Benchmarking
- Camera
- Computer Vision and Perception
- Converter
- Deployment
- Development
- Extended Reality
- Healthcare AI
- Image Processing
- Inference
- Interoperability
- Medical Imaging
- Natural Language and Conversational AI
- Networking and Distributed Computing
- Optimization
- Quantum Computing
- Rendering
- Robotics
- Scheduler
- Signal Processing
- Streaming
- Threading
- Video
- Video Capture
- Visualization
- XR

### Example Review Comment

If a PR contains invalid tags, provide feedback like:

> ❌ **Metadata Category Validation Failed**
>
> The following category in `path/to/metadata.json` is not in the approved category list: `NewCategory`
>
> Please replace it with an existing category from [the approved category list](https://github.com/nvidia-holoscan/holohub/tree/main/.github/copilot-instructions.md#approved-category-list). Here are some existing categories that might be suitable alternatives:
>
> - For `NewCategory`, consider: `Similar Category A`, `Similar Category B`
