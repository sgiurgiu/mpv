/* Copyright (C) 2024 the mpv developers
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef MPV_CLIENT_API_RENDER_NEXT_H_
#define MPV_CLIENT_API_RENDER_NEXT_H_

#include "render.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * GPU-Next backend (libplacebo-based)
 * ------------------------------------
 *
 * This header contains definitions for using the gpu-next renderer with the
 * render.h API. The gpu-next renderer uses libplacebo directly and provides
 * advanced rendering features like HDR tone mapping, frame interpolation,
 * and high-quality scaling.
 *
 * Supported graphics APIs:
 * - Vulkan: Provide raw Vulkan handles via mpv_next_vk_init_params
 * - OpenGL: Provide get_proc_address callback via mpv_next_gl_init_params
 *
 * The client is responsible for:
 * - Creating and managing the graphics context (Vulkan instance/device or
 *   OpenGL context)
 * - Creating and providing a pl_swapchain from libplacebo
 * - Calling pl_swapchain_swap_buffers() after mpv_render_context_render()
 *
 * Basic usage (Vulkan):
 *
 *   1. Create Vulkan instance, device, and queues
 *   2. Create pl_vulkan using pl_vulkan_import() with your handles
 *   3. Create pl_swapchain using pl_vulkan_create_swapchain()
 *   4. Call mpv_render_context_create() with:
 *      - MPV_RENDER_PARAM_API_TYPE = MPV_RENDER_API_TYPE_NEXT
 *      - MPV_RENDER_PARAM_NEXT_VK_INIT_PARAMS = &vk_init_params
 *      - MPV_RENDER_PARAM_NEXT_SWAPCHAIN = pl_swapchain
 *   5. Render loop: mpv_render_context_render() then swap buffers
 *
 * Basic usage (OpenGL):
 *
 *   1. Create and make current your OpenGL context
 *   2. Create pl_opengl using pl_opengl_create()
 *   3. Create pl_swapchain using pl_opengl_create_swapchain()
 *   4. Call mpv_render_context_create() with:
 *      - MPV_RENDER_PARAM_API_TYPE = MPV_RENDER_API_TYPE_NEXT
 *      - MPV_RENDER_PARAM_NEXT_GL_INIT_PARAMS = &gl_init_params
 *      - MPV_RENDER_PARAM_NEXT_SWAPCHAIN = pl_swapchain
 *   5. Render loop: mpv_render_context_render() then swap buffers
 */


/**
 * Initialization parameters for the Vulkan backend.
 * Used with MPV_RENDER_PARAM_NEXT_VK_INIT_PARAMS.
 *
 * The client must provide valid Vulkan handles. mpv will use these to
 * import a pl_vulkan context via pl_vulkan_import().
 *
 * Required extensions: The Vulkan device should be created with extensions
 * suitable for video playback. At minimum, swapchain support is needed.
 * For hardware decoding, additional extensions may be required depending
 * on the hwdec method used.
 */
typedef struct mpv_next_vk_init_params {
    /**
     * Valid VkInstance handle. Must remain valid for the lifetime of the
     * render context.
     */
    /* VkInstance */ void *instance;

    /**
     * Valid VkPhysicalDevice handle.
     */
    /* VkPhysicalDevice */ void *physical_device;

    /**
     * Valid VkDevice handle. Must remain valid for the lifetime of the
     * render context.
     */
    /* VkDevice */ void *device;

    /**
     * Queue family index for graphics operations.
     */
    uint32_t graphics_queue_family;

    /**
     * Queue index within the graphics queue family.
     */
    uint32_t graphics_queue_index;

    /**
     * Queue family index for async compute operations.
     * Set to UINT32_MAX if not available or not desired.
     */
    uint32_t compute_queue_family;

    /**
     * Queue index within the compute queue family.
     * Ignored if compute_queue_family is UINT32_MAX.
     */
    uint32_t compute_queue_index;

    /**
     * Queue family index for async transfer operations.
     * Set to UINT32_MAX if not available or not desired.
     */
    uint32_t transfer_queue_family;

    /**
     * Queue index within the transfer queue family.
     * Ignored if transfer_queue_family is UINT32_MAX.
     */
    uint32_t transfer_queue_index;

    /**
     * Array of instance extension names that were enabled when creating
     * the VkInstance. Can be NULL if num_instance_extensions is 0.
     */
    const char **enabled_instance_extensions;

    /**
     * Number of entries in enabled_instance_extensions.
     */
    int num_instance_extensions;

    /**
     * Array of device extension names that were enabled when creating
     * the VkDevice. Can be NULL if num_device_extensions is 0.
     */
    const char **enabled_device_extensions;

    /**
     * Number of entries in enabled_device_extensions.
     */
    int num_device_extensions;

    /**
     * Vulkan API version (e.g., VK_API_VERSION_1_2).
     * This should match the version used when creating the instance.
     */
    uint32_t api_version;

    /**
     * Function pointer for loading Vulkan functions.
     * This is typically vkGetInstanceProcAddr.
     * Type: PFN_vkGetInstanceProcAddr
     */
    void *(*get_instance_proc_addr)(void *instance, const char *name);
} mpv_next_vk_init_params;

/**
 * Initialization parameters for the OpenGL backend.
 * Used with MPV_RENDER_PARAM_NEXT_GL_INIT_PARAMS.
 *
 * Similar to mpv_opengl_init_params but for the gpu-next renderer.
 */
typedef struct mpv_next_gl_init_params {
    /**
     * Callback for loading OpenGL function pointers.
     *
     * This is called by libplacebo to resolve OpenGL functions. You can
     * typically use the GL context API's GetProcAddress function here
     * (e.g., glXGetProcAddressARB, wglGetProcAddress, eglGetProcAddress).
     *
     * @param ctx   The get_proc_address_ctx value
     * @param name  The name of the OpenGL function to load
     * @return      Function pointer, or NULL if not found
     */
    void *(*get_proc_address)(void *ctx, const char *name);

    /**
     * Arbitrary context pointer passed to get_proc_address.
     */
    void *get_proc_address_ctx;

    /**
     * Set to true to enable OpenGL debug mode.
     * This enables additional error checking and debug callbacks.
     */
    int debug;

    /**
     * Set to true to allow software rendering (e.g., Mesa llvmpipe).
     * By default, software renderers may be rejected.
     */
    int allow_software;
} mpv_next_gl_init_params;

#ifdef __cplusplus
}
#endif

#endif
