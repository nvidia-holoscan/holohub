/*
 * GStreamer Custom Simple Sink Element
 * A basic video sink that demonstrates how to create a native GStreamer sink
 */

#ifndef __GST_SIMPLE_CUSTOM_SINK_H__
#define __GST_SIMPLE_CUSTOM_SINK_H__

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <gst/video/video.h>

G_BEGIN_DECLS

/* Standard macros for defining the GStreamer element */
#define GST_TYPE_SIMPLE_CUSTOM_SINK \
  (gst_simple_custom_sink_get_type())
#define GST_SIMPLE_CUSTOM_SINK(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_SIMPLE_CUSTOM_SINK,GstSimpleCustomSink))
#define GST_SIMPLE_CUSTOM_SINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_SIMPLE_CUSTOM_SINK,GstSimpleCustomSinkClass))
#define GST_IS_SIMPLE_CUSTOM_SINK(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_SIMPLE_CUSTOM_SINK))
#define GST_IS_SIMPLE_CUSTOM_SINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_SIMPLE_CUSTOM_SINK))

typedef struct _GstSimpleCustomSink GstSimpleCustomSink;
typedef struct _GstSimpleCustomSinkClass GstSimpleCustomSinkClass;

/**
 * GstSimpleCustomSink:
 * @parent: the parent object
 * @width: video width
 * @height: video height
 * @frame_count: number of frames processed
 * @fps: frames per second (for demonstration)
 * @save_frames: whether to save frames to files
 * @output_dir: directory to save frames
 *
 * The custom sink object structure
 */
struct _GstSimpleCustomSink 
{
  GstBaseSink parent;

  /* Properties */
  gint width;
  gint height;
  guint frame_count;
  gdouble fps;
  gboolean save_frames;
  gchar *output_dir;
  
  /* Private data */
  GstVideoInfo video_info;
  gboolean info_set;
};

/**
 * GstSimpleCustomSinkClass:
 * @parent_class: the parent class
 *
 * The custom sink class structure
 */
struct _GstSimpleCustomSinkClass
{
  GstBaseSinkClass parent_class;
};

/* Function declarations */
GType gst_simple_custom_sink_get_type(void);
gboolean gst_simple_custom_sink_plugin_init(GstPlugin *plugin);

G_END_DECLS

#endif /* __GST_SIMPLE_CUSTOM_SINK_H__ */
