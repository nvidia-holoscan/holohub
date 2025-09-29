/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gst_sink_resource.hpp"
#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <gst/video/video.h>

extern "C" {

// ============================================================================
// GStreamer Custom Simple Sink Element Implementation (embedded in C++)
// ============================================================================

/* Standard macros for defining the GStreamer element */
#define GST_TYPE_HOLOSCAN_SINK \
  (gst_holoscan_sink_get_type())
#define GST_HOLOSCAN_SINK(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_HOLOSCAN_SINK,GstHoloscanSink))
#define GST_HOLOSCAN_SINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_HOLOSCAN_SINK,GstHoloscanSinkClass))
#define GST_IS_HOLOSCAN_SINK(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_HOLOSCAN_SINK))
#define GST_IS_HOLOSCAN_SINK_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_HOLOSCAN_SINK))

typedef struct _GstHoloscanSink GstHoloscanSink;
typedef struct _GstHoloscanSinkClass GstHoloscanSinkClass;

/**
 * GstHoloscanSink:
 * @parent: the parent object
 * @buffer_count: number of buffers processed (for monitoring)
 * @caps_set: whether caps have been negotiated
 *
 * The Holoscan sink object structure (internal GStreamer element for data bridging)
 */
struct _GstHoloscanSink 
{
  GstBaseSink parent;

  /* Processing state */
  guint buffer_count;
  gboolean caps_set;
};

/**
 * GstHoloscanSinkClass:
 * @parent_class: the parent class
 *
 * The Holoscan sink class structure (internal GStreamer element class)
 */
struct _GstHoloscanSinkClass
{
  GstBaseSinkClass parent_class;
};

GST_DEBUG_CATEGORY_STATIC(gst_holoscan_sink_debug);
#define GST_CAT_DEFAULT gst_holoscan_sink_debug

/* Properties */
enum
{
  PROP_0
};

/* Pad templates */
static GstStaticPadTemplate sink_pad_template = GST_STATIC_PAD_TEMPLATE("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("ANY")
);

/* Function prototypes */
static void gst_holoscan_sink_set_property(GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec);
static void gst_holoscan_sink_get_property(GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec);
static void gst_holoscan_sink_finalize(GObject *object);

static gboolean gst_holoscan_sink_set_caps(GstBaseSink *sink, GstCaps *caps);
static GstFlowReturn gst_holoscan_sink_render(GstBaseSink *sink, GstBuffer *buffer);
static gboolean gst_holoscan_sink_start(GstBaseSink *sink);
static gboolean gst_holoscan_sink_stop(GstBaseSink *sink);

/* Class initialization */
#define gst_holoscan_sink_parent_class parent_class
G_DEFINE_TYPE(GstHoloscanSink, gst_holoscan_sink, GST_TYPE_BASE_SINK);

static void
gst_holoscan_sink_class_init(GstHoloscanSinkClass *klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseSinkClass *gstbasesink_class;

  gobject_class = G_OBJECT_CLASS(klass);
  gstelement_class = GST_ELEMENT_CLASS(klass);
  gstbasesink_class = GST_BASE_SINK_CLASS(klass);

  /* Set up object methods */
  gobject_class->set_property = gst_holoscan_sink_set_property;
  gobject_class->get_property = gst_holoscan_sink_get_property;
  gobject_class->finalize = gst_holoscan_sink_finalize;

  /* No properties needed for basic data bridging */

  /* Set element metadata */
  gst_element_class_set_static_metadata(gstelement_class,
      "Holoscan Bridge Sink",
      "Sink/Generic",
      "A GStreamer sink element that bridges data to Holoscan operators",
      "NVIDIA Corporation <holoscan@nvidia.com>");

  /* Add pad template */
  gst_element_class_add_static_pad_template(gstelement_class, &sink_pad_template);

  /* Set up base sink methods */
  gstbasesink_class->set_caps = GST_DEBUG_FUNCPTR(gst_holoscan_sink_set_caps);
  gstbasesink_class->render = GST_DEBUG_FUNCPTR(gst_holoscan_sink_render);
  gstbasesink_class->start = GST_DEBUG_FUNCPTR(gst_holoscan_sink_start);
  gstbasesink_class->stop = GST_DEBUG_FUNCPTR(gst_holoscan_sink_stop);

  /* Initialize debug category */
  GST_DEBUG_CATEGORY_INIT(gst_holoscan_sink_debug, "holoscansink", 0,
      "Holoscan Sink Element");
}

static void
gst_holoscan_sink_init(GstHoloscanSink *sink)
{
  /* Initialize state */
  sink->buffer_count = 0;
  sink->caps_set = FALSE;

  /* Enable QoS and lateness handling for better performance */
  gst_base_sink_set_qos_enabled(GST_BASE_SINK(sink), TRUE);
}

static void
gst_holoscan_sink_finalize(GObject *object)
{
  GstHoloscanSink *sink = GST_HOLOSCAN_SINK(object);

  /* No dynamic resources to clean up in the simplified version */
  GST_DEBUG_OBJECT(sink, "Finalizing Holoscan sink");

  G_OBJECT_CLASS(parent_class)->finalize(object);
}

static void
gst_holoscan_sink_set_property(GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec)
{
  GstHoloscanSink *sink = GST_HOLOSCAN_SINK(object);

  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static void
gst_holoscan_sink_get_property(GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec)
{
  GstHoloscanSink *sink = GST_HOLOSCAN_SINK(object);

  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static gboolean
gst_holoscan_sink_set_caps(GstBaseSink *sink, GstCaps *caps)
{
  GstHoloscanSink *holoscan_sink = GST_HOLOSCAN_SINK(sink);
  GstStructure *structure;
  const gchar *media_type;
  
  GST_DEBUG_OBJECT(sink, "Setting caps: %" GST_PTR_FORMAT, caps);

  /* Get the first structure from caps to determine media type */
  structure = gst_caps_get_structure(caps, 0);
  media_type = gst_structure_get_name(structure);

  GST_INFO_OBJECT(sink, "Accepting caps for bridging: %s", media_type);

  /* Mark caps as successfully negotiated */
  holoscan_sink->caps_set = TRUE;
  return TRUE;
}

static GstFlowReturn
gst_holoscan_sink_render(GstBaseSink *sink, GstBuffer *buffer)
{
  GstHoloscanSink *holoscan_sink = GST_HOLOSCAN_SINK(sink);

  if (!holoscan_sink->caps_set) {
    GST_ERROR_OBJECT(sink, "Caps not negotiated");
    return GST_FLOW_NOT_NEGOTIATED;
  }

  holoscan_sink->buffer_count++;

  /* Log buffer information for monitoring */
  GST_DEBUG_OBJECT(sink, "Bridging buffer %u, size: %" G_GSIZE_FORMAT " bytes",
      holoscan_sink->buffer_count, gst_buffer_get_size(buffer));

  /* Print basic info to console for demo purposes */
  g_print("Buffer %u: size: %" G_GSIZE_FORMAT " bytes\n",
      holoscan_sink->buffer_count, gst_buffer_get_size(buffer));

  /* TODO: This is where Holoscan operators would retrieve the buffer data */
  /* Future extension point for data bridging to Holoscan operators */

  return GST_FLOW_OK;
}

static gboolean
gst_holoscan_sink_start(GstBaseSink *sink)
{
  GstHoloscanSink *holoscan_sink = GST_HOLOSCAN_SINK(sink);
  
  GST_DEBUG_OBJECT(sink, "Starting Holoscan bridge sink");
  
  holoscan_sink->buffer_count = 0;
  holoscan_sink->caps_set = FALSE;
  
  return TRUE;
}

static gboolean
gst_holoscan_sink_stop(GstBaseSink *sink)
{
  GstHoloscanSink *holoscan_sink = GST_HOLOSCAN_SINK(sink);
  
  GST_DEBUG_OBJECT(sink, "Stopping Holoscan bridge sink");
  GST_INFO_OBJECT(sink, "Processed %u buffers total", holoscan_sink->buffer_count);
  
  holoscan_sink->caps_set = FALSE;
  
  return TRUE;
}

/* Element registration function for direct use */
gboolean 
gst_holoscan_sink_plugin_init(GstPlugin *plugin)
{
  return gst_element_register(plugin, "holoscansink", GST_RANK_NONE,
      GST_TYPE_HOLOSCAN_SINK);
}

}  // extern "C"

// ============================================================================
// Holoscan GstSinkResource Implementation (C++)
// ============================================================================

namespace holoscan {

GstSinkResource::~GstSinkResource() {
  HOLOSCAN_LOG_DEBUG("Destroying GstSinkResource");
  if (sink_element_ && GST_IS_ELEMENT(sink_element_)) {
    gst_element_set_state(sink_element_, GST_STATE_NULL);
    gst_object_unref(sink_element_);
    sink_element_ = nullptr;
  }
  HOLOSCAN_LOG_DEBUG("GstSinkResource destroyed");
}

void GstSinkResource::initialize() {
  HOLOSCAN_LOG_INFO("Initializing GstSinkResource for data bridging");

  // Initialize GStreamer if not already done
  if (!gst_is_initialized()) {
    gst_init(nullptr, nullptr);
  }

  // Register our bridge sink element type
  gst_element_register(nullptr, "holoscansink", GST_RANK_NONE,
                      gst_holoscan_sink_get_type());

  // Create the sink element
  sink_element_ = gst_element_factory_make("holoscansink",
                                         sink_name_.empty() ? nullptr : sink_name_.c_str());

  if (!sink_element_) {
    HOLOSCAN_LOG_ERROR("Failed to create Holoscan bridge sink element");
    return;
  }

  HOLOSCAN_LOG_INFO("GstSinkResource initialized successfully for data bridging");
}


}  // namespace holoscan
