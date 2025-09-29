/*
 * GStreamer Custom Simple Sink Element Implementation
 * A basic video sink that demonstrates how to create a native GStreamer sink
 */

#include "gstsimplecustomsink.h"
#include <gst/video/video.h>
#include <stdio.h>
#include <string.h>

GST_DEBUG_CATEGORY_STATIC(gst_simple_custom_sink_debug);
#define GST_CAT_DEFAULT gst_simple_custom_sink_debug

/* Properties */
enum
{
  PROP_0,
  PROP_SAVE_FRAMES,
  PROP_OUTPUT_DIR,
  PROP_FPS
};

/* Default values */
#define DEFAULT_SAVE_FRAMES FALSE
#define DEFAULT_OUTPUT_DIR "/tmp"
#define DEFAULT_FPS 30.0

/* Pad templates */
static GstStaticPadTemplate sink_pad_template = GST_STATIC_PAD_TEMPLATE("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("ANY")
);

/* Function prototypes */
static void gst_simple_custom_sink_set_property(GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec);
static void gst_simple_custom_sink_get_property(GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec);
static void gst_simple_custom_sink_finalize(GObject *object);

static gboolean gst_simple_custom_sink_set_caps(GstBaseSink *sink, GstCaps *caps);
static GstFlowReturn gst_simple_custom_sink_render(GstBaseSink *sink, GstBuffer *buffer);
static gboolean gst_simple_custom_sink_start(GstBaseSink *sink);
static gboolean gst_simple_custom_sink_stop(GstBaseSink *sink);

/* Class initialization */
#define gst_simple_custom_sink_parent_class parent_class
G_DEFINE_TYPE(GstSimpleCustomSink, gst_simple_custom_sink, GST_TYPE_BASE_SINK);

static void
gst_simple_custom_sink_class_init(GstSimpleCustomSinkClass *klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseSinkClass *gstbasesink_class;

  gobject_class = G_OBJECT_CLASS(klass);
  gstelement_class = GST_ELEMENT_CLASS(klass);
  gstbasesink_class = GST_BASE_SINK_CLASS(klass);

  /* Set up object methods */
  gobject_class->set_property = gst_simple_custom_sink_set_property;
  gobject_class->get_property = gst_simple_custom_sink_get_property;
  gobject_class->finalize = gst_simple_custom_sink_finalize;

  /* Install properties */
  g_object_class_install_property(gobject_class, PROP_SAVE_FRAMES,
      g_param_spec_boolean("save-frames", "Save Buffers",
          "Save received data buffers to files", DEFAULT_SAVE_FRAMES,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobject_class, PROP_OUTPUT_DIR,
      g_param_spec_string("output-dir", "Output Directory",
          "Directory to save data buffers", DEFAULT_OUTPUT_DIR,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobject_class, PROP_FPS,
      g_param_spec_double("fps", "Data Rate",
          "Target data rate per second (for display)", 0.0, G_MAXDOUBLE,
          DEFAULT_FPS, G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  /* Set element metadata */
  gst_element_class_set_static_metadata(gstelement_class,
      "Simple Custom Universal Sink",
      "Sink/Generic",
      "A universal custom sink that accepts any data type for demonstration purposes",
      "Your Name <your.email@example.com>");

  /* Add pad template */
  gst_element_class_add_static_pad_template(gstelement_class, &sink_pad_template);

  /* Set up base sink methods */
  gstbasesink_class->set_caps = GST_DEBUG_FUNCPTR(gst_simple_custom_sink_set_caps);
  gstbasesink_class->render = GST_DEBUG_FUNCPTR(gst_simple_custom_sink_render);
  gstbasesink_class->start = GST_DEBUG_FUNCPTR(gst_simple_custom_sink_start);
  gstbasesink_class->stop = GST_DEBUG_FUNCPTR(gst_simple_custom_sink_stop);

  /* Initialize debug category */
  GST_DEBUG_CATEGORY_INIT(gst_simple_custom_sink_debug, "simplecustomsink", 0,
      "Simple Custom Video Sink");
}

static void
gst_simple_custom_sink_init(GstSimpleCustomSink *sink)
{
  /* Initialize properties */
  sink->save_frames = DEFAULT_SAVE_FRAMES;
  sink->output_dir = g_strdup(DEFAULT_OUTPUT_DIR);
  sink->fps = DEFAULT_FPS;
  sink->frame_count = 0;
  sink->width = 0;
  sink->height = 0;
  sink->info_set = FALSE;

  /* Enable QoS and lateness handling */
  gst_base_sink_set_qos_enabled(GST_BASE_SINK(sink), TRUE);
}

static void
gst_simple_custom_sink_finalize(GObject *object)
{
  GstSimpleCustomSink *sink = GST_SIMPLE_CUSTOM_SINK(object);

  g_free(sink->output_dir);

  G_OBJECT_CLASS(parent_class)->finalize(object);
}

static void
gst_simple_custom_sink_set_property(GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec)
{
  GstSimpleCustomSink *sink = GST_SIMPLE_CUSTOM_SINK(object);

  switch (prop_id) {
    case PROP_SAVE_FRAMES:
      sink->save_frames = g_value_get_boolean(value);
      break;
    case PROP_OUTPUT_DIR:
      g_free(sink->output_dir);
      sink->output_dir = g_value_dup_string(value);
      break;
    case PROP_FPS:
      sink->fps = g_value_get_double(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static void
gst_simple_custom_sink_get_property(GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec)
{
  GstSimpleCustomSink *sink = GST_SIMPLE_CUSTOM_SINK(object);

  switch (prop_id) {
    case PROP_SAVE_FRAMES:
      g_value_set_boolean(value, sink->save_frames);
      break;
    case PROP_OUTPUT_DIR:
      g_value_set_string(value, sink->output_dir);
      break;
    case PROP_FPS:
      g_value_set_double(value, sink->fps);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static gboolean
gst_simple_custom_sink_set_caps(GstBaseSink *sink, GstCaps *caps)
{
  GstSimpleCustomSink *custom_sink = GST_SIMPLE_CUSTOM_SINK(sink);
  GstStructure *structure;
  const gchar *media_type;
  
  GST_DEBUG_OBJECT(sink, "Setting caps: %" GST_PTR_FORMAT, caps);

  /* Get the first structure from caps to determine media type */
  structure = gst_caps_get_structure(caps, 0);
  media_type = gst_structure_get_name(structure);

  GST_INFO_OBJECT(sink, "Accepting media type: %s", media_type);

  /* Try to parse video info if it's a video stream */
  if (g_str_has_prefix(media_type, "video/")) {
    if (gst_video_info_from_caps(&custom_sink->video_info, caps)) {
      custom_sink->width = GST_VIDEO_INFO_WIDTH(&custom_sink->video_info);
      custom_sink->height = GST_VIDEO_INFO_HEIGHT(&custom_sink->video_info);
      GST_INFO_OBJECT(sink, "Video format: %s, %dx%d",
          gst_video_format_to_string(GST_VIDEO_INFO_FORMAT(&custom_sink->video_info)),
          custom_sink->width, custom_sink->height);
    } else {
      GST_WARNING_OBJECT(sink, "Could not parse video info, but continuing anyway");
      custom_sink->width = 0;
      custom_sink->height = 0;
    }
  } else {
    /* For non-video data, just set dimensions to 0 */
    custom_sink->width = 0;
    custom_sink->height = 0;
    GST_INFO_OBJECT(sink, "Non-video data accepted: %s", media_type);
  }

  custom_sink->info_set = TRUE;
  return TRUE;
}

static GstFlowReturn
gst_simple_custom_sink_render(GstBaseSink *sink, GstBuffer *buffer)
{
  GstSimpleCustomSink *custom_sink = GST_SIMPLE_CUSTOM_SINK(sink);
  GstMapInfo map;
  gchar *filename;
  FILE *file;

  if (!custom_sink->info_set) {
    GST_ERROR_OBJECT(sink, "Caps not set");
    return GST_FLOW_NOT_NEGOTIATED;
  }

  custom_sink->frame_count++;

  /* Log buffer information */
  GST_DEBUG_OBJECT(sink, "Received buffer %u, size: %" G_GSIZE_FORMAT " bytes",
      custom_sink->frame_count, gst_buffer_get_size(buffer));

  /* Print buffer info to console - format depends on whether we have video info */
  if (custom_sink->width > 0 && custom_sink->height > 0) {
    g_print("Buffer %u (Video): %dx%d, size: %" G_GSIZE_FORMAT " bytes\n",
        custom_sink->frame_count, custom_sink->width, custom_sink->height,
        gst_buffer_get_size(buffer));
  } else {
    g_print("Buffer %u (Generic): size: %" G_GSIZE_FORMAT " bytes\n",
        custom_sink->frame_count, gst_buffer_get_size(buffer));
  }

  /* Save buffer to file if requested */
  if (custom_sink->save_frames) {
    if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
      filename = g_strdup_printf("%s/buffer_%06u.raw", 
          custom_sink->output_dir, custom_sink->frame_count);
      
      file = fopen(filename, "wb");
      if (file) {
        fwrite(map.data, 1, map.size, file);
        fclose(file);
        GST_DEBUG_OBJECT(sink, "Saved buffer to: %s", filename);
      } else {
        GST_WARNING_OBJECT(sink, "Failed to open file: %s", filename);
      }
      
      g_free(filename);
      gst_buffer_unmap(buffer, &map);
    } else {
      GST_ERROR_OBJECT(sink, "Failed to map buffer");
    }
  }

  return GST_FLOW_OK;
}

static gboolean
gst_simple_custom_sink_start(GstBaseSink *sink)
{
  GstSimpleCustomSink *custom_sink = GST_SIMPLE_CUSTOM_SINK(sink);
  
  GST_DEBUG_OBJECT(sink, "Starting");
  
  custom_sink->frame_count = 0;
  
  /* Create output directory if saving buffers */
  if (custom_sink->save_frames) {
    g_mkdir_with_parents(custom_sink->output_dir, 0755);
  }
  
  return TRUE;
}

static gboolean
gst_simple_custom_sink_stop(GstBaseSink *sink)
{
  GstSimpleCustomSink *custom_sink = GST_SIMPLE_CUSTOM_SINK(sink);
  
  GST_DEBUG_OBJECT(sink, "Stopping");
  GST_INFO_OBJECT(sink, "Processed %u buffers total", custom_sink->frame_count);
  
  custom_sink->info_set = FALSE;
  
  return TRUE;
}

/* Element registration function for direct use */
gboolean 
gst_simple_custom_sink_plugin_init(GstPlugin *plugin)
{
  return gst_element_register(plugin, "simplecustomsink", GST_RANK_NONE,
      GST_TYPE_SIMPLE_CUSTOM_SINK);
}
