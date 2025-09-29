#include <gst/gst.h>
#include <iostream>
#include <signal.h>

// External declaration of our custom sink plugin initialization
extern "C" {
  gboolean gst_simple_custom_sink_plugin_init(GstPlugin *plugin);
}

static GMainLoop *loop = nullptr;
static GstElement *pipeline = nullptr;

static void
sigint_handler(int sig)
{
  g_print("\nCaught signal %d, stopping pipeline...\n", sig);
  if (loop) {
    g_main_loop_quit(loop);
  }
}

static gboolean
bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *)data;

  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
      g_print("End of stream\n");
      g_main_loop_quit(loop);
      break;
      
    case GST_MESSAGE_ERROR: {
      gchar *debug;
      GError *error;
      
      gst_message_parse_error(msg, &error, &debug);
      g_printerr("Error: %s\n", error->message);
      if (debug) {
        g_printerr("Debug: %s\n", debug);
      }
      g_error_free(error);
      g_free(debug);
      
      g_main_loop_quit(loop);
      break;
    }
    
    case GST_MESSAGE_WARNING: {
      gchar *debug;
      GError *error;
      
      gst_message_parse_warning(msg, &error, &debug);
      g_printerr("Warning: %s\n", error->message);
      if (debug) {
        g_printerr("Debug: %s\n", debug);
      }
      g_error_free(error);
      g_free(debug);
      break;
    }
    
    case GST_MESSAGE_STATE_CHANGED: {
      if (GST_MESSAGE_SRC(msg) == GST_OBJECT(pipeline)) {
        GstState old_state, new_state;
        gst_message_parse_state_changed(msg, &old_state, &new_state, NULL);
        g_print("Pipeline state changed from %s to %s\n",
                gst_element_state_get_name(old_state),
                gst_element_state_get_name(new_state));
      }
      break;
    }
    
    default:
      break;
  }
  
  return TRUE;
}

extern "C" {
  GType gst_simple_custom_sink_get_type(void);
}

static void
register_custom_plugin()
{
  // Register the element factory directly
  gst_element_register(NULL, "simplecustomsink", GST_RANK_NONE, 
                      gst_simple_custom_sink_get_type());
}

int main(int argc, char *argv[])
{
  GstElement *source, *convert, *sink;
  GstBus *bus;
  guint bus_watch_id;
  gboolean save_frames = FALSE;
  gchar *output_dir = nullptr;
  
  // Handle command line arguments
  if (argc > 1) {
    if (g_strcmp0(argv[1], "--save-frames") == 0) {
      save_frames = TRUE;
      if (argc > 2) {
        output_dir = argv[2];
      }
    }
  }

  // Initialize GStreamer
  gst_init(&argc, &argv);

  // Register our custom plugin
  register_custom_plugin();

  loop = g_main_loop_new(NULL, FALSE);

  // Set up signal handler
  signal(SIGINT, sigint_handler);

  // Create pipeline and elements
  pipeline = gst_pipeline_new("custom-sink-pipeline");
  source = gst_element_factory_make("videotestsrc", "source");
  convert = gst_element_factory_make("videoconvert", "convert");
  sink = gst_element_factory_make("simplecustomsink", "custom_sink");

  if (!pipeline || !source || !convert || !sink) {
    g_printerr("One or more elements could not be created. Exiting.\n");
    return -1;
  }

  // Configure the source
  g_object_set(G_OBJECT(source), 
               "pattern", 0,  // SMPTE color bars
               "num-buffers", 300,  // Generate 300 frames (10 seconds at 30 fps)
               NULL);

  // Configure our custom sink
  g_object_set(G_OBJECT(sink),
               "save-frames", save_frames,
               "fps", 30.0,
               NULL);
  
  if (output_dir) {
    g_object_set(G_OBJECT(sink), "output-dir", output_dir, NULL);
  }

  // Set up the pipeline
  gst_bin_add_many(GST_BIN(pipeline), source, convert, sink, NULL);
  
  if (!gst_element_link_many(source, convert, sink, NULL)) {
    g_printerr("Elements could not be linked. Exiting.\n");
    gst_object_unref(pipeline);
    return -1;
  }

  // Add a message handler
  bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
  gst_object_unref(bus);

  // Print usage information
  g_print("Starting pipeline with custom sink...\n");
  g_print("The sink will process video frames and print information to console.\n");
  if (save_frames) {
    g_print("Frames will be saved to: %s\n", output_dir ? output_dir : "/tmp");
  }
  g_print("Press Ctrl+C to stop.\n\n");

  // Start playing
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  // Iterate
  g_main_loop_run(loop);

  // Out of the main loop, clean up nicely
  g_print("Returned, stopping pipeline\n");
  gst_element_set_state(pipeline, GST_STATE_NULL);

  gst_object_unref(GST_OBJECT(pipeline));
  g_source_remove(bus_watch_id);
  g_main_loop_unref(loop);

  g_print("Custom sink example finished.\n");
  
  return 0;
}
