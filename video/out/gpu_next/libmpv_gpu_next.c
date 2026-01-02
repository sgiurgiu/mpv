/*
 * This file is part of mpv.
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "config.h"

#include <libplacebo/options.h>
#include <libplacebo/renderer.h>
#include <libplacebo/shaders/lut.h>
#include <libplacebo/utils/libav.h>
#include <libplacebo/utils/frame_queue.h>


#ifdef PL_HAVE_VULKAN
#include <libplacebo/vulkan.h>
#endif

#ifdef PL_HAVE_OPENGL
#include <libplacebo/opengl.h>
#endif

#include "common/common.h"
#include "common/msg.h"
#include "options/m_config.h"
#include "video/out/libmpv.h"
#include "video/out/gpu/video.h"
#include "video/out/gpu/video_shaders.h"
#include "video/out/gpu/hwdec.h"
#include "video/out/placebo/ra_pl.h"
#include "video/out/placebo/utils.h"
#include "video/mp_image.h"
#include "video/fmt-conversion.h"
#include "sub/osd.h"
#include "sub/draw_bmp.h"

#include "mpv/render_next.h"

struct osd_entry {
    pl_tex tex;
    struct pl_overlay_part *parts;
    int num_parts;
};

struct overlay_state {
    struct osd_entry entries[MAX_OSD_PARTS];
    struct pl_overlay overlays[MAX_OSD_PARTS];
};

struct frame_priv {
    struct overlay_state subs;
    uint64_t osd_sync;
    struct ra_hwdec *hwdec;
};

struct priv {
    struct mp_log *log;
    struct mpv_global *global;

    // libplacebo objects
    pl_log pllog;
    pl_gpu gpu;
    pl_renderer rr;
    pl_queue queue;
    pl_swapchain swapchain;

    // Backend type
    enum {
        BACKEND_NONE,
        BACKEND_VULKAN,
        BACKEND_OPENGL,
    } backend_type;

#ifdef PL_HAVE_VULKAN
    pl_vulkan vulkan;
#endif
#ifdef PL_HAVE_OPENGL
    pl_opengl opengl;
#endif

    // OSD state
    pl_fmt osd_fmt[SUBBITMAP_COUNT];
    pl_tex *sub_tex;
    int num_sub_tex;
    struct overlay_state osd_state;
    uint64_t osd_sync;

    // Frame state
    struct mp_rect src, dst;
    struct mp_osd_res osd_res;
    struct mp_image_params img_params;
    struct osd_state *osd;  // OSD source from vo
    uint64_t last_id;
    double last_pts;
    bool want_reset;

    // Rendering options
    pl_options pars;
    struct m_config_cache *opts_cache;

    // Hardware decoding
    struct ra_hwdec_ctx hwdec_ctx;
    struct ra_hwdec_mapper *hwdec_mapper;
    struct mp_hwdec_devices *hwdec_devs;

    // For ra interop (needed for hwdec)
    struct ra *ra;
};

static void update_options(struct priv *p);

static int plane_data_from_imgfmt(struct pl_plane_data out_data[4],
                                  struct pl_bit_encoding *out_bits,
                                  enum mp_imgfmt imgfmt)
{
    struct mp_imgfmt_desc desc = mp_imgfmt_get_desc(imgfmt);
    if (!desc.num_planes || !(desc.flags & MP_IMGFLAG_HAS_COMPS))
        return 0;

    if (desc.flags & MP_IMGFLAG_HWACCEL)
        return 0;

    if (!(desc.flags & MP_IMGFLAG_NE))
        return 0;

    if (desc.flags & MP_IMGFLAG_PAL)
        return 0;

    if ((desc.flags & MP_IMGFLAG_TYPE_FLOAT) && (desc.flags & MP_IMGFLAG_YUV))
        return 0;

    bool has_bits = false;

    for (int p = 0; p < desc.num_planes; p++) {
        struct pl_plane_data *data = &out_data[p];
        struct mp_imgfmt_comp_desc sorted[MP_NUM_COMPONENTS];
        int num_comps = 0;
        if (desc.bpp[p] % 8)
            return 0;

        for (int c = 0; c < mp_imgfmt_desc_get_num_comps(&desc); c++) {
            if (desc.comps[c].plane != p)
                continue;

            data->component_map[num_comps] = c;
            sorted[num_comps] = desc.comps[c];
            num_comps++;

            for (int i = num_comps - 1; i > 0; i--) {
                if (sorted[i].offset >= sorted[i - 1].offset)
                    break;
                MPSWAP(struct mp_imgfmt_comp_desc, sorted[i], sorted[i - 1]);
                MPSWAP(int, data->component_map[i], data->component_map[i - 1]);
            }
        }

        uint64_t total_bits = 0;
        memset(data->component_size, 0, sizeof(data->component_size));
        for (int c = 0; c < num_comps; c++) {
            data->component_size[c] = sorted[c].size;
            data->component_pad[c] = sorted[c].offset - total_bits;
            total_bits += data->component_pad[c] + data->component_size[c];

            if (!out_bits || data->component_map[c] == PL_CHANNEL_A)
                continue;

            struct pl_bit_encoding bits = {
                .sample_depth = data->component_size[c],
                .color_depth = sorted[c].size - abs(sorted[c].pad),
                .bit_shift = MPMAX(sorted[c].pad, 0),
            };

            if (!has_bits) {
                *out_bits = bits;
                has_bits = true;
            } else {
                if (!pl_bit_encoding_equal(out_bits, &bits)) {
                    *out_bits = (struct pl_bit_encoding) {0};
                    out_bits = NULL;
                }
            }
        }

        data->pixel_stride = desc.bpp[p] / 8;
        data->type = (desc.flags & MP_IMGFLAG_TYPE_FLOAT)
                            ? PL_FMT_FLOAT
                            : PL_FMT_UNORM;
    }

    return desc.num_planes;
}

static bool format_supported(struct priv *p, int format)
{
    struct pl_bit_encoding bits;
    struct pl_plane_data data[4] = {0};
    int planes = plane_data_from_imgfmt(data, &bits, format);
    if (!planes)
        return false;

    for (int i = 0; i < planes; i++) {
        if (!pl_plane_find_fmt(p->gpu, NULL, &data[i]))
            return false;
    }

    return true;
}

static int init(struct render_backend *ctx, mpv_render_param *params)
{
    ctx->priv = talloc_zero(NULL, struct priv);
    struct priv *p = ctx->priv;
    p->log = ctx->log;
    p->global = ctx->global;

    char *api = get_mpv_render_param(params, MPV_RENDER_PARAM_API_TYPE, NULL);
    if (!api || strcmp(api, MPV_RENDER_API_TYPE_NEXT) != 0)
        return MPV_ERROR_NOT_IMPLEMENTED;

    // Get optional pl_log
    pl_log external_log = get_mpv_render_param(params, MPV_RENDER_PARAM_NEXT_PL_LOG, NULL);

    // Get required swapchain
    p->swapchain = get_mpv_render_param(params, MPV_RENDER_PARAM_NEXT_SWAPCHAIN, NULL);
    if (!p->swapchain) {
        MP_ERR(p, "MPV_RENDER_PARAM_NEXT_SWAPCHAIN is required\n");
        return MPV_ERROR_INVALID_PARAMETER;
    }

    // Try Vulkan init
    mpv_next_vk_init_params *vk_params = get_mpv_render_param(params,
        MPV_RENDER_PARAM_NEXT_VK_INIT_PARAMS, NULL);

    // Try OpenGL init
    mpv_next_gl_init_params *gl_params = get_mpv_render_param(params,
        MPV_RENDER_PARAM_NEXT_GL_INIT_PARAMS, NULL);

    if (vk_params && gl_params) {
        MP_ERR(p, "Cannot specify both Vulkan and OpenGL init params\n");
        return MPV_ERROR_INVALID_PARAMETER;
    }

    if (!vk_params && !gl_params) {
        MP_ERR(p, "Must specify either Vulkan or OpenGL init params\n");
        return MPV_ERROR_INVALID_PARAMETER;
    }

    // Create or use pl_log
    if (external_log) {
        p->pllog = external_log;
    } else {
        p->pllog = mppl_log_create(p, p->log);
        if (!p->pllog) {
            MP_ERR(p, "Failed to create libplacebo log\n");
            return MPV_ERROR_GENERIC;
        }
    }

#ifdef PL_HAVE_VULKAN
    if (vk_params) {
        p->backend_type = BACKEND_VULKAN;

        //p->vulkan = pl_vulkan_import(p->pllog, &import_params);
        //if (!p->vulkan) {
        //    MP_ERR(p, "Failed to import Vulkan context\n");
        //    return MPV_ERROR_GENERIC;
        //}

        p->gpu = p->swapchain->gpu;
        p->ra = ra_create_pl(p->gpu, p->log);
    }
#else
    if (vk_params) {
        MP_ERR(p, "Vulkan support not compiled in libplacebo\n");
        return MPV_ERROR_NOT_IMPLEMENTED;
    }
#endif

#ifdef PL_HAVE_OPENGL
    if (gl_params) {
        p->backend_type = BACKEND_OPENGL;
        
        p->gpu = p->swapchain->gpu;
        p->ra = ra_create_pl(p->gpu, p->log);
    }
#else
    if (gl_params) {
        MP_ERR(p, "OpenGL support not compiled in libplacebo\n");
        return MPV_ERROR_NOT_IMPLEMENTED;
    }
#endif

    if (!p->gpu) {
        MP_ERR(p, "Failed to obtain GPU context\n");
        return MPV_ERROR_GENERIC;
    }

    // Create renderer and queue
    p->rr = pl_renderer_create(p->pllog, p->gpu);
    if (!p->rr) {
        MP_ERR(p, "Failed to create libplacebo renderer\n");
        return MPV_ERROR_GENERIC;
    }

    p->queue = pl_queue_create(p->gpu);
    if (!p->queue) {
        MP_ERR(p, "Failed to create frame queue\n");
        return MPV_ERROR_GENERIC;
    }

    // Initialize OSD formats
    p->osd_fmt[SUBBITMAP_LIBASS] = pl_find_named_fmt(p->gpu, "r8");
    p->osd_fmt[SUBBITMAP_BGRA] = pl_find_named_fmt(p->gpu, "bgra8");
    p->osd_sync = 1;

    // Initialize render options
    p->pars = pl_options_alloc(p->pllog);
    p->opts_cache = m_config_cache_alloc(p, p->global, &gl_video_conf);

    // Initialize hwdec context
    ctx->hwdec_devs = hwdec_devices_create();
    p->hwdec_devs = ctx->hwdec_devs;

    // Set capabilities
    ctx->driver_caps = VO_CAP_ROTATE90 | VO_CAP_FILM_GRAIN | VO_CAP_VFLIP;

    return 0;
}

static bool check_format(struct render_backend *ctx, int imgfmt)
{
    struct priv *p = ctx->priv;

    // Check hwdec formats
    if (ra_hwdec_get(&p->hwdec_ctx, imgfmt))
        return true;

    return format_supported(p, imgfmt);
}

static int set_parameter(struct render_backend *ctx, mpv_render_param param)
{
    // struct priv *p = ctx->priv;
    (void)ctx;

    switch (param.type) {
    case MPV_RENDER_PARAM_ICC_PROFILE: {
        // ICC profile handling would go here
        // For now, we don't support dynamic ICC profile changes
        return MPV_ERROR_NOT_IMPLEMENTED;
    }
    default:
        return MPV_ERROR_NOT_IMPLEMENTED;
    }
}

static void reconfig(struct render_backend *ctx, struct mp_image_params *params)
{
    struct priv *p = ctx->priv;
    p->img_params = *params;
    p->want_reset = true;
}

static void reset(struct render_backend *ctx)
{
    struct priv *p = ctx->priv;
    p->want_reset = true;
}

static void update_external(struct render_backend *ctx, struct vo *vo)
{
    struct priv *p = ctx->priv;
    p->osd = vo ? vo->osd : NULL;
}

static void resize(struct render_backend *ctx, struct mp_rect *src,
                   struct mp_rect *dst, struct mp_osd_res *osd)
{
    struct priv *p = ctx->priv;
    p->src = *src;
    p->dst = *dst;
    p->osd_res = *osd;

    // Notify swapchain of the new size
    int w = mp_rect_w(*dst);
    int h = mp_rect_h(*dst);
    if (w > 0 && h > 0) {
        if (!pl_swapchain_resize(p->swapchain, &w, &h)) {
            MP_WARN(p, "Failed to resize swapchain to %dx%d\n", w, h);
        }
        // Update dst to actual swapchain size (may differ from requested)
        p->dst.x0 = 0;
        p->dst.y0 = 0;
        p->dst.x1 = w;
        p->dst.y1 = h;
    }
}

static int get_target_size(struct render_backend *ctx, mpv_render_param *params,
                           int *out_w, int *out_h)
{
    struct priv *p = ctx->priv;

    // Get size from swapchain - pass 0,0 to query current size
    int w = 0, h = 0;
    if (!pl_swapchain_resize(p->swapchain, &w, &h) || w <= 0 || h <= 0) {
        // If resize fails or returns invalid size, try to get size from frame
        struct pl_swapchain_frame frame;
        if (pl_swapchain_start_frame(p->swapchain, &frame)) {
            w = frame.fbo->params.w;
            h = frame.fbo->params.h;
            // We started a frame just to get size, need to cancel it
            // Unfortunately there's no clean way to do this, so we submit empty
            pl_swapchain_submit_frame(p->swapchain);
        } else {
            return MPV_ERROR_GENERIC;
        }
    }

    *out_w = w;
    *out_h = h;
    return 0;
}

static void update_overlays(struct priv *p, struct mp_osd_res res,
                            struct pl_frame *frame, struct overlay_state *state,
                            struct osd_state *osd, double pts)
{
    if (!osd)
        return;

    struct sub_bitmap_list *subs = osd_render(osd, res, pts, 0, mp_draw_sub_formats);

    frame->overlays = state->overlays;
    frame->num_overlays = 0;

    for (int n = 0; n < subs->num_items; n++) {
        const struct sub_bitmaps *item = subs->items[n];
        if (!item->num_parts || !item->packed)
            continue;

        struct osd_entry *entry = &state->entries[item->render_index];
        pl_fmt tex_fmt = p->osd_fmt[item->format];

        if (!entry->tex)
            MP_TARRAY_POP(p->sub_tex, p->num_sub_tex, &entry->tex);

        bool ok = pl_tex_recreate(p->gpu, &entry->tex, &(struct pl_tex_params) {
            .format = tex_fmt,
            .w = MPMAX(item->packed_w, entry->tex ? entry->tex->params.w : 0),
            .h = MPMAX(item->packed_h, entry->tex ? entry->tex->params.h : 0),
            .host_writable = true,
            .sampleable = true,
        });

        if (!ok) {
            MP_ERR(p, "Failed recreating OSD texture!\n");
            break;
        }

        ok = pl_tex_upload(p->gpu, &(struct pl_tex_transfer_params) {
            .tex        = entry->tex,
            .rc         = { .x1 = item->packed_w, .y1 = item->packed_h, },
            .row_pitch  = item->packed->stride[0],
            .ptr        = item->packed->planes[0],
        });

        if (!ok) {
            MP_ERR(p, "Failed uploading OSD texture!\n");
            break;
        }

        entry->num_parts = 0;
        for (int i = 0; i < item->num_parts; i++) {
            const struct sub_bitmap *b = &item->parts[i];
            if (b->dw == 0 || b->dh == 0)
                continue;

            uint32_t c = b->libass.color;
            struct pl_overlay_part part = {
                .src = { b->src_x, b->src_y, b->src_x + b->w, b->src_y + b->h },
                .dst = { b->x, b->y, b->x + b->dw, b->y + b->dh },
                .color = {
                    (c >> 24) / 255.0f,
                    ((c >> 16) & 0xFF) / 255.0f,
                    ((c >> 8) & 0xFF) / 255.0f,
                    (255 - (c & 0xFF)) / 255.0f,
                }
            };
            MP_TARRAY_APPEND(p, entry->parts, entry->num_parts, part);
        }

        struct pl_overlay *ol = &state->overlays[frame->num_overlays++];
        *ol = (struct pl_overlay) {
            .tex = entry->tex,
            .parts = entry->parts,
            .num_parts = entry->num_parts,
            .color = {
                .primaries = PL_COLOR_PRIM_BT_709,
                .transfer = PL_COLOR_TRC_SRGB,
            },
            .coords = PL_OVERLAY_COORDS_DST_FRAME,
        };

        switch (item->format) {
        case SUBBITMAP_BGRA:
            ol->mode = PL_OVERLAY_NORMAL;
            ol->repr.alpha = PL_ALPHA_PREMULTIPLIED;
            break;
        case SUBBITMAP_LIBASS:
            ol->mode = PL_OVERLAY_MONOCHROME;
            ol->repr.alpha = PL_ALPHA_INDEPENDENT;
            break;
        }
    }

    talloc_free(subs);
}

static bool map_frame(pl_gpu gpu, pl_tex *tex, const struct pl_source_frame *src,
                      struct pl_frame *frame)
{
    struct mp_image *mpi = src->frame_data;
    struct frame_priv *fp = mpi->priv;
    struct priv *p = fp->hwdec ? NULL : mpi->priv; // Simplified - would need proper context
    (void)p;

    if (!mpi)
        return false;

    struct mp_image_params par = mpi->params;
    mp_image_params_guess_csp(&par);

    *frame = (struct pl_frame) {
        .color = par.color,
        .repr = par.repr,
        .rotation = par.rotate / 90,
        .user_data = mpi,
    };

    // Software decoding path
    struct pl_plane_data data[4] = {0};
    frame->num_planes = plane_data_from_imgfmt(data, &frame->repr.bits, mpi->imgfmt);

    for (int n = 0; n < frame->num_planes; n++) {
        struct pl_plane *plane = &frame->planes[n];
        data[n].width = mp_image_plane_w(mpi, n);
        data[n].height = mp_image_plane_h(mpi, n);

        if (mpi->stride[n] < 0) {
            data[n].pixels = mpi->planes[n] + (data[n].height - 1) * mpi->stride[n];
            data[n].row_stride = -mpi->stride[n];
            plane->flipped = true;
        } else {
            data[n].pixels = mpi->planes[n];
            data[n].row_stride = mpi->stride[n];
        }

        if (!pl_upload_plane(gpu, plane, &tex[n], &data[n])) {
            talloc_free(mpi);
            return false;
        }
    }

    pl_frame_set_chroma_location(frame, par.chroma_location);

    if (mpi->film_grain)
        pl_film_grain_from_av(&frame->film_grain, (AVFilmGrainParams *) mpi->film_grain->data);

    return true;
}

static void unmap_frame(pl_gpu gpu, struct pl_frame *frame,
                        const struct pl_source_frame *src)
{
    struct mp_image *mpi = src->frame_data;
    talloc_free(mpi);
}

static void discard_frame(const struct pl_source_frame *src)
{
    struct mp_image *mpi = src->frame_data;
    talloc_free(mpi);
}

static int render(struct render_backend *ctx, mpv_render_param *params,
                  struct vo_frame *frame)
{
    struct priv *p = ctx->priv;

    update_options(p);

    // Handle reset
    if (p->want_reset) {
        pl_renderer_flush_cache(p->rr);
        pl_queue_reset(p->queue);
        p->last_pts = 0.0;
        p->last_id = 0;
        p->want_reset = false;
    }

    // Push incoming frames to queue
    for (int n = 0; n < frame->num_frames; n++) {
        int id = frame->frame_id + n;
        if (id <= p->last_id)
            continue;

        struct mp_image *mpi = mp_image_new_ref(frame->frames[n]);
        struct frame_priv *fp = talloc_zero(mpi, struct frame_priv);
        mpi->priv = fp;

        pl_queue_push(p->queue, &(struct pl_source_frame) {
            .pts = mpi->pts,
            .duration = frame->approx_duration,
            .frame_data = mpi,
            .map = map_frame,
            .unmap = unmap_frame,
            .discard = discard_frame,
        });

        p->last_id = id;
    }

    // Start swapchain frame
    struct pl_swapchain_frame swframe;
    if (!pl_swapchain_start_frame(p->swapchain, &swframe)) {
        MP_ERR(p, "Failed to start swapchain frame\n");
        return MPV_ERROR_GENERIC;
    }

    struct pl_frame target;
    pl_frame_from_swapchain(&target, &swframe);

    // Set up target crop - use full framebuffer if dst is not set
    if (p->dst.x1 > p->dst.x0 && p->dst.y1 > p->dst.y0) {
        target.crop = (struct pl_rect2df) {
            .x0 = p->dst.x0,
            .y0 = p->dst.y0,
            .x1 = p->dst.x1,
            .y1 = p->dst.y1,
        };
    } else {
        // Use full framebuffer size
        target.crop = (struct pl_rect2df) {
            .x0 = 0,
            .y0 = 0,
            .x1 = swframe.fbo->params.w,
            .y1 = swframe.fbo->params.h,
        };
    }

    // Get frame mix from queue
    struct pl_frame_mix mix = {0};
    if (frame->current) {
        struct pl_queue_params qparams = *pl_queue_params(
            .pts = frame->current->pts,
            .radius = pl_frame_mix_radius(&p->pars->params),
        );

        switch (pl_queue_update(p->queue, &mix, &qparams)) {
        case PL_QUEUE_ERR:
            MP_ERR(p, "Failed updating frame queue!\n");
            pl_tex_clear(p->gpu, swframe.fbo, (float[4]){ 0.5, 0.0, 1.0, 1.0 });
            goto submit;
        case PL_QUEUE_EOF:
            break;
        case PL_QUEUE_MORE:
        case PL_QUEUE_OK:
            break;
        }

        // Apply source crop to all frames
        for (int i = 0; i < mix.num_frames; i++) {
            struct pl_frame *image = (struct pl_frame *) mix.frames[i];
            struct mp_image *mpi = image->user_data;
            // Use configured crop if set, otherwise use full image
            if (p->src.x1 > p->src.x0 && p->src.y1 > p->src.y0) {
                image->crop = (struct pl_rect2df) {
                    .x0 = p->src.x0,
                    .y0 = p->src.y0,
                    .x1 = p->src.x1,
                    .y1 = p->src.y1,
                };
            } else if (mpi) {
                image->crop = (struct pl_rect2df) {
                    .x0 = 0,
                    .y0 = 0,
                    .x1 = mpi->params.w,
                    .y1 = mpi->params.h,
                };
            }
        }
    }

    // Update OSD overlays on target frame
    double pts = frame->current ? frame->current->pts : 0;
    update_overlays(p, p->osd_res, &target, &p->osd_state, p->osd, pts);

    // Render
    struct pl_render_params render_params = p->pars->params;
    if (!pl_render_image_mix(p->rr, &mix, &target, &render_params)) {
        MP_ERR(p, "Failed rendering frame!\n");
        pl_tex_clear(p->gpu, swframe.fbo, (float[4]){ 0.5, 0.0, 1.0, 1.0 });
    }

submit:
    pl_gpu_flush(p->gpu);

    if (!pl_swapchain_submit_frame(p->swapchain)) {
        MP_ERR(p, "Failed submitting frame to swapchain!\n");
        return MPV_ERROR_GENERIC;
    }

    return 0;
}

static struct mp_image *get_image(struct render_backend *ctx, int imgfmt,
                                  int w, int h, int stride_align, int flags)
{
    struct priv *p = ctx->priv;
    pl_gpu gpu = p->gpu;

    if (!gpu->limits.thread_safe || !gpu->limits.max_mapped_size)
        return NULL;

    // DR (direct rendering) support could be added here
    return NULL;
}

static void screenshot(struct render_backend *ctx, struct vo_frame *frame,
                       struct voctrl_screenshot *args)
{
    // Screenshot support would be implemented here
    args->res = NULL;
}

static void perfdata(struct render_backend *ctx,
                     struct voctrl_performance_data *out)
{
    // Performance data collection would be implemented here
    memset(out, 0, sizeof(*out));
}

static void destroy(struct render_backend *ctx)
{
    struct priv *p = ctx->priv;

    if (!p)
        return;

    // Destroy queue first
    pl_queue_destroy(&p->queue);

    // Destroy OSD textures
    for (int i = 0; i < MP_ARRAY_SIZE(p->osd_state.entries); i++)
        pl_tex_destroy(p->gpu, &p->osd_state.entries[i].tex);
    for (int i = 0; i < p->num_sub_tex; i++)
        pl_tex_destroy(p->gpu, &p->sub_tex[i]);

    // Destroy hwdec
    if (p->hwdec_mapper)
        ra_hwdec_mapper_free(&p->hwdec_mapper);
    ra_hwdec_ctx_uninit(&p->hwdec_ctx);
    hwdec_devices_destroy(ctx->hwdec_devs);

    // Destroy renderer
    pl_renderer_destroy(&p->rr);

    // Destroy options
    pl_options_free(&p->pars);

    // Destroy RA
    if (p->ra)
        p->ra->fns->destroy(p->ra);

    // Destroy backend-specific resources
#ifdef PL_HAVE_VULKAN
    if (p->vulkan)
        pl_vulkan_destroy(&p->vulkan);
#endif

#ifdef PL_HAVE_OPENGL
    if (p->opengl)
        pl_opengl_destroy(&p->opengl);
#endif

    // Destroy pl_log if we created it
    if (p->pllog)
        pl_log_destroy(&p->pllog);

    talloc_free(p);
    ctx->priv = NULL;
}

static void update_options(struct priv *p)
{
    m_config_cache_update(p->opts_cache);

    // Apply basic render options from config
    pl_options pars = p->pars;
    const struct gl_video_opts *opts = p->opts_cache->opts;

    pars->params.background_color[0] = opts->background_color.r / 255.0;
    pars->params.background_color[1] = opts->background_color.g / 255.0;
    pars->params.background_color[2] = opts->background_color.b / 255.0;
    pars->params.background_transparency = 1 - opts->background_color.a / 255.0;

    // Deband params
    pars->params.deband_params = opts->deband ? &pars->deband_params : NULL;
    if (opts->deband && opts->deband_opts) {
        pars->deband_params.iterations = opts->deband_opts->iterations;
        pars->deband_params.radius = opts->deband_opts->range;
        pars->deband_params.threshold = opts->deband_opts->threshold / 16.384;
        pars->deband_params.grain = opts->deband_opts->grain / 8.192;
    }

    // Sigmoid params
    pars->params.sigmoid_params = opts->sigmoid_upscaling ? &pars->sigmoid_params : NULL;
    if (opts->sigmoid_upscaling) {
        pars->sigmoid_params.center = opts->sigmoid_center;
        pars->sigmoid_params.slope = opts->sigmoid_slope;
    }
}

const struct render_backend_fns render_backend_gpu_next = {
    .init = init,
    .check_format = check_format,
    .set_parameter = set_parameter,
    .reconfig = reconfig,
    .reset = reset,
    .update_external = update_external,
    .resize = resize,
    .get_target_size = get_target_size,
    .render = render,
    .get_image = get_image,
    .screenshot = screenshot,
    .perfdata = perfdata,
    .destroy = destroy,
};
