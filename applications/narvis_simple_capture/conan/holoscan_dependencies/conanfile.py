from conan import ConanFile
from conan.tools.files import apply_conandata_patches, collect_libs, copy, export_conandata_patches, get, rename, replace_in_file, rmdir, save


def mkdep(pkg_name:str, **kw):
    return (pkg_name, kw)

HOLOSCAN_DEPENDENCIES = {
    "arrow": mkdep("arrow/12.0.1@camposs/stable", transitive_libs=True),
    "assimp": mkdep("assimp/5.2.2", transitive_libs=True),
    "bitset2": mkdep("bitset2/0.1@camposs/stable"),
    "boost": mkdep("boost/1.82.0", transitive_headers=True, transitive_libs=True, force=True),
    "capnproto": mkdep("capnproto/0.10.3", transitive_headers=True, transitive_libs=True, visible=True),
    "cereal": mkdep("cereal/1.3.2", transitive_libs=True),
    "corrade": mkdep("corrade/2020.06@camposs/stable", transitive_libs=True),
    "cppfs": mkdep("cppfs/1.3.0@camposs/stable", transitive_libs=True),
    "cpptrace": mkdep("cpptrace/0.6.1", transitive_headers=True),
    "cuda_dev_config": mkdep("cuda_dev_config/2.2@camposs/stable", force=True),
    "draco": mkdep("draco/1.5.6", transitive_headers=True, transitive_libs=True),
    "ebml": mkdep("ebml/1.4.4@camposs/stable", transitive_headers=True, transitive_libs=True),
    "eigen": mkdep("eigen/3.4.0"),
    "enet": mkdep("enet/1.3.17", transitive_libs=True),
    "eventbus": mkdep("eventbus/3.0.0-r2@camposs/stable"),
    "fast-cdr": mkdep("fast-cdr/2.0.0@camposs/stable", force=True, transitive_headers=True, transitive_libs=True),
    "fast-dds": mkdep("fast-dds/2.10.1@camposs/stable", transitive_headers=True),
    "fast-dds-gen": mkdep("fast-dds-gen/3.0.0@camposs/stable"),
    "fiberpool": mkdep("fiberpool/0.1@camposs/stable"),
    "ffmpeg": mkdep("ffmpeg/6.1@camposs/stable", transitive_libs=True, force=True),
    "fmt": mkdep("fmt/9.1.0", force=True),
    "freetype": mkdep("freetype/2.13.2@camposs/stable", transitive_libs=True, force=True),
    "ftxui": mkdep("ftxui/5.0.0", transitive_libs=True, force=True),
    "glfw": mkdep("glfw/3.3.8", force=True),
    "gstreamer": mkdep("gstreamer/1.22.0@camposs/stable", transitive_libs=True),
    "gtest": mkdep("gtest/1.14.0", force=True),
    "h264nal": mkdep("h264nal/0.17@camposs/stable"),
    "iceoryx": mkdep("iceoryx/2.0.6@camposs/stable", transitive_libs=True),
    "kinect-azure-bodytracking-sdk": mkdep("kinect-azure-bodytracking-sdk/1.1.0@vendor/stable", transitive_libs=True, run=True),
    "kinect-azure-sensor-sdk": mkdep("kinect-azure-sensor-sdk/1.4.1-r3@camposs/stable", transitive_libs=True, run=True),
    "libalsa": mkdep("libalsa/1.2.10"),
    "libe57format": mkdep("libe57format/2.3.0", transitive_libs=True),
    "libjpeg": mkdep("libjpeg/9e", force=True),
    "libjpeg-turbo": mkdep("libjpeg-turbo/3.0.1", transitive_headers=True, transitive_libs=True, force=True),
    "librealsense": mkdep("librealsense/2.53.1", transitive_headers=True, transitive_libs=True, force=True),
    "libnuma": mkdep("libnuma/2.0.14"),
    "libwebp": mkdep("libwebp/1.3.2", force=True),
    "libpng": mkdep("libpng/1.6.40", override=True),
    "libxml2": mkdep("libxml2/2.12.4", override=True),
    "live555": mkdep("live555/1.26.0@camposs/stable", transitive_libs=True),
    "magnum": mkdep("magnum/2020.06@camposs/stable", transitive_libs=True),
    "magnum-integration": mkdep("magnum-integration/2020.06@camposs/stable"),
    "magnum-plugins": mkdep("magnum-plugins/2020.06@camposs/stable"),
    "matroska": mkdep("matroska/1.7.1@camposs/stable", transitive_libs=True, transitive_headers=True),
    "ncurses": mkdep("ncurses/6.4"),
    "nvidia-video-codec-sdk": mkdep("nvidia-video-codec-sdk/12.1.14.0@vendor/stable", transitive_libs=True),
    "newtekndi": mkdep("newtekndi/5.0@vendor/stable", transitive_libs=True),
    "open3d": mkdep("open3d/0.17.0@camposs/stable"),
    "opencv": mkdep("opencv/4.10.0@camposs/stable", transitive_libs=True, transitive_headers=True),
    "opengl": mkdep("opengl/system"),
    "openssl": mkdep("openssl/1.1.1t", force=True),
    "optick": mkdep("optick/1.4.0.0@camposs/stable"),
    "orbbec-sdk": mkdep("orbbec-sdk/1.10.12@vendor/stable", transitive_headers=True, transitive_libs=True, run=True),
    "pcl": mkdep("pcl/1.14.1@camposs/stable", transitive_libs=True),
    "pcre2": mkdep("pcre2/10.42", transitive_libs=True),
    "pybind11": mkdep("pybind11/2.10.1"),
    "python_dev_config": mkdep("python_dev_config/1.0@camposs/stable"),
    "qt": mkdep("qt/6.6.1@camposs/stable", force=True, run=True, visible=True),
    "qt-propertybrowser": mkdep("qt-propertybrowser/2.0@camposs/stable"),
    "rapidjson": mkdep("rapidjson/cci.20230929@camposs/stable", force=True),
    "rtmidi": mkdep("rtmidi/5.0.0@camposs/stable", transitive_libs=True),
    "rttr": mkdep("rttr/0.9.7-dev@camposs/stable", transitive_headers=True, transitive_libs=True),
    "sdl": mkdep("sdl/2.26.1", force=True),
    "spdlog": mkdep("spdlog/1.11.0", transitive_libs=True),
    "st_tree": mkdep("st_tree/1.2.2", transitive_headers=True),
    "tcn_schema": mkdep("tcn_schema/0.0.1@artekmed/stable", transitive_libs=True, transitive_headers=True),
    "util-linux-libuuid": mkdep("util-linux-libuuid/2.39", force=True),
    "uvgrtp": mkdep("uvgrtp/2.1.2@camposs/stable", transitive_libs=True),
    "wayland": mkdep("wayland-protocols/1.31"),
    "xkbcommon": mkdep("xkbcommon/1.5.0", force=True),
    "xorg": mkdep("xorg/system"),
    "xz_utils": mkdep("xz_utils/5.4.5", force=True),
    "yaml-cpp": mkdep("yaml-cpp/0.7.0", transitive_libs=True),
    "yuv": mkdep("yuv/1749@camposs/stable", transitive_libs=True, transitive_headers=True),
    "zdepth": mkdep("zdepth/0.2@camposs/stable", transitive_libs=True),
    "zenoh-cpp": mkdep("zenoh-cpp/0.11.0@camposs/stable", transitive_libs=True, transitive_headers=True),
    "zlib": mkdep("zlib/1.3@camposs/stable", transitive_libs=True, force=True, run=True, visible=True),
    "zstd": mkdep("zstd/1.5.5", transitive_headers=True, transitive_libs=True),    
    "zulu-openjdk": mkdep("zulu-openjdk/11.0.15"),
}


def get_default_options(self:ConanFile):
    defaults = {
        "gstreamer/*:with_libjpeg": "libjpeg-turbo",
        "jasper/*:with_libjpeg": "libjpeg-turbo",
        "libsndfile/*:with_mpeg": False,
        "libtiff/*:jpeg": "libjpeg-turbo",
        "magnum-integration/*:with_bullet": False,  # does not build on windows debug for the moment ...
        "magnum-integration/*:with_eigen": True,
        "magnum-integration/*:with_imgui": True,
        "magnum-plugins/*:with_stbimageconverter": True,
        "magnum-plugins/*:with_stbimageimporter": True,
        "magnum/*:target_gles": False,
        "magnum/*:with_anyimageimporter": True,
        "magnum/*:with_anysceneimporter": True,
        "magnum/*:with_eglcontext": False,
        "magnum/*:with_gl_info": True,
        "magnum/*:with_imageconverter": True,
        "magnum/*:with_objimporter": True,
        "magnum/*:with_opengltester": True,
        "magnum/*:with_sdl2application": True,
        "magnum/*:with_tgaimageconverter": True,
        "magnum/*:with_tgaimporter": True,
        "magnum/*:with_windowlesseglapplication": False,
        "opencv/*:with_jpeg": "libjpeg-turbo",
        "qt/*:with_libjpeg": "libjpeg-turbo",
        "qt/*:with_mysql": False,
        "qt/*:with_odbc": False,
        "qt/*:with_pq": False,
        "qt/*:with_sqlite3": False,
        "yuv/*:with_jpeg": "libjpeg-turbo",
        "tcn_schema/*:with_dds": False,
    }
    return defaults


def available_packages() -> list:
    return HOLOSCAN_DEPENDENCIES.keys()

def add_build_tools(self:ConanFile, tools:list, **kwargs):
    
    deps_ok = True
    for d in tools:
        if d not in HOLOSCAN_DEPENDENCIES:
            self.output.warning("Missing external tool: {}".format(d))
            deps_ok = False
        else:
            pkg_name, kw = HOLOSCAN_DEPENDENCIES[d]
            self.tool_requires(pkg_name, **kwargs)

    if not deps_ok:
        raise ValueError("Missing External Dependencies.")

def add_dependencies(self:ConanFile, deps:list):
    
    deps_ok = True
    for d in deps:
        if d not in HOLOSCAN_DEPENDENCIES:
            self.output.warning("Missing external dependency: {}".format(d))
            deps_ok = False
        else:
            pkg_name, kw = HOLOSCAN_DEPENDENCIES[d]
            self.requires(pkg_name, **kw)

    if not deps_ok:
        raise ValueError("Missing External Dependencies.")

def configure_dependencies(self:ConanFile, shared_lib:bool, deps:list):
    if shared_lib:
        if "bzip2" in deps:
            self.options["bzip2"].shared = True

        if "zlib" in deps:
            self.options["zlib"].shared = True

        if "zstd" in deps:
            self.options["zstd"].shared = True

        if "boost" in deps:
            self.options['boost'].shared = True

        if "fast-dds" in deps:
            self.options['fast-dds'].shared = True

        if "ffmpeg" in deps:
            self.options['ffmpeg'].shared = True
            self.options['ffmpeg'].with_ssl = False
            self.options['ffmpeg'].with_cuda = True

        if "opencv" in deps:
            self.options['opencv'].shared = True
    
        if "openssl" in deps:
            self.options['openssl'].shared = True
    
        if "gstreamer" in deps:
            self.options['gstreamer'].shared = True
            self.options['glib'].shared = True

        # if "tcn_schema" in deps:
        #     self.options['tcn_schema'].shared = True

        if "libjpeg-turbo" in deps:
            self.options["libjpeg-turbo"].shared = True


        if self.settings.os == "Linux":
            if "corrade" in deps:
                self.options['corrade'].shared = True

            if "magnum" in deps:
                self.options['magnum'].shared = True

            if "capnproto" in deps:
                self.options['capnproto'].shared = False

            if "rtmidi" in deps:
                self.options['rtmidi'].shared = True


        if self.settings.os == "Windows":
            if "spdlog" in deps:
                self.options['spdlog'].shared = True

            if "draco" in deps:
                self.options['draco'].shared = True

            if "yuv" in deps:
                self.options['yuv'].shared = True


    if "iceoryx" in deps:
        self.options['iceoryx'].with_introspection = True

    if "magnum" in deps:
        self.options["magnum"].with_gl_info = True
        self.options["magnum"].with_eglcontext = False
        self.options["magnum"].with_windowlesseglapplication = False
        self.options["magnum"].target_gles = False
        self.options["magnum"].with_opengltester = True

    if "opencv" in deps:
        self.options['opencv'].with_cuda = True

    if "pcl" in deps:
        self.options['pcl'].with_cuda = True



    if self.settings.os == "Linux":
        if "magnum" in deps:
            self.options['magnum'].with_windowlessglxapplication = True

        if "opencv" in deps:
            self.options['opencv'].with_gtk = True

        if "capnproto" in deps:
            self.options['capnproto'].shared = True


    if self.settings.os == "Windows":

        if "magnum" in deps:
            self.options['magnum'].with_windowlesswglapplication = True
            self.options['magnum'].shared = False

        if "corrade" in deps:
            self.options['corrade'].shared = False

        if "enet" in deps:
            self.options['enet'].shared = False

        if "capnproto" in deps:
            # use static version of capnproto and include all logic of network transfer into pcpd
            self.options['capnproto'].shared = False

        if "eventbus" in deps:
            self.options['eventbus'].shared = False



class PcpdDependenciesConan(ConanFile):
    name = "holoscan_dependencies"
    version = "0.4.0"
    package_type = "python-require"

    description = "Holoscan 3rdparty dependencies collected"
    url = "https://github.com/TUM-CAMP-NARVIS/holohub-narvis"
    license = "internal"
