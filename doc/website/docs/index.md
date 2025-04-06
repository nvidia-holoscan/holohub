---
title: Holoscan Reference Applications # Used for the header when scrolling down
hide:
  - navigation
  - footer
---

<!-- Hide the site name in the header when scrolled to the top
to avoid redundancy with the page title -->
<style>
  /* Hide site/page name text when the title container is not 'active' by making it transparent */
  .md-header__title:not(.md-header__title--active) .md-header__topic {
    opacity: 0;
  }

  /* Override default transition: fade-in only, no slide */
  .md-header__topic {
    transition: opacity 0.2s ease !important;
    transform: none !important;
  }

  /* Hide the permalink anchor within the page title (from mkdocs / toc / permalink: true) */
  .md-content h1 > a.headerlink {
    display: none;
  }
</style>

<!-- We need it despite 'title:' above or the .nav title value ('Home') would be used-->
# Holoscan Reference Applications

**Holoscan Reference Applications** is a central repository for the [NVIDIA Holoscan](https://www.nvidia.com/en-us/clara/holoscan/) AI sensor processing community
to share reference applications, operators, tutorials and benchmarks.
The repository hosts a variety of applications that demonstrate how to use Holoscan for streaming, imaging, and other AI-driven tasks across embedded, edge, and cloud environments. These applications serve as reference implementations, providing developers with examples of best practices and efficient coding techniques to build high-performance, low-latency AI applications. The repository is open to contributions from the community, encouraging developers to share their own applications and extensions to enhance the Holoscan ecosystem.

<div class="grid cards" markdown>

-   :material-merge:{ .lg } __Workflows__ (#workflows)

    ---

    Reference workflows demonstrate how capabilities from applications and operators can be combined
    to achieve complex tasks.

    [Browse Workflows](workflows){ .md-button .md-button }


-   :material-apps:{ .lg } __Applications__ (#applications)

    ---

    Reference applications demonstrate a specific capability of Holoscan or how a specific operator
    can be used to perform an optimize task.

    [Browse Applications](applications){ .md-button .md-button }

-   :material-code-greater-than:{ .lg } __Operators__ (#operators)

    ---

    Operators perform a specific task.

    [Browse Operators](operators){ .md-button .md-button }

-   :material-book-open-variant-outline:{ .lg } __Tutorials__ (#tutorials)

    ---

    Tutorials provide hands-on experience.

    [Browse Tutorials](tutorials){ .md-button .md-button }


-   :material-chart-box-outline:{ .lg } __Benchmarks__ (#benchmarks)

    ---

    Benchmarks provide tools for assessing performance of Holoscan pipelines as well as reference benchmarks for specific releases

    [Browse Benchmarks](benchmarks){ .md-button .md-button }




</div>
