from conan import ConanFile


class DeployDepsConan(ConanFile):
    python_requires = "pcpd_dependencies/[>=0.4.0 <1.0]@artekmed/stable"

    options = {
    }

    default_options = {
        "cuda_dev_config/*:cuda_version": "12.2",
        "cuda_dev_config/*:cuda_archs": "75",
        "qt/*:with_libjpeg": "libjpeg-turbo",
        "opencv/*:with_jpeg": "libjpeg-turbo",
        "opencv/*:with_cuda": True,
        "opencv/*:with_ipp": False,
        "opencv/*:with_gtk": False,
        "libsndfile/*:with_mpeg": False,
        "yuv/*:with_jpeg": "libjpeg-turbo",
        "libtiff/*:jpeg": "libjpeg-turbo",
        "jasper/*:with_libjpeg": "libjpeg-turbo",
        "zstd/*:shared": True,
    }

    settings = "os",
    generators = "VirtualRunEnv",

    @property
    def pcpd_dependencies(self):
        return self.python_requires["pcpd_dependencies"].module

    def init(self):
        d_opts = self.pcpd_dependencies.get_default_options(self)
        d_opts.update(self.default_options)
        self.default_options = d_opts

    def get_dependencies(self):
        dependencies = ["kinect-azure-sensor-sdk", "capnproto", "eigen", "fast-cdr", "fmt", "iceoryx", "libjpeg-turbo", "opencv", "openssl", "zenoh-cpp", "zlib", "zdepth"]

        return dependencies

    def configure(self):
        self.pcpd_dependencies.configure_dependencies(self, True, self.get_dependencies())


    def requirements(self):
        self.pcpd_dependencies.add_dependencies(self, self.get_dependencies())
