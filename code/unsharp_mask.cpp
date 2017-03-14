#include <chrono>
#include <climits>
#include <sstream>
#include "cl/cl.h"
#include "err_code.h"

#include "ppm.hpp"

//
//
//
static bool use_gaussian_blur = true;

struct OpenCLStuff {
    bool success;
    cl_context context;
    cl_kernel ko_blur, ko_weighted;
    cl_command_queue commands;
};

static size_t global_item_size[2];

static int arg_cnt = 0;
template<typename T> static void set_argument_helper(cl_kernel kernel, T var) {
    int err = clSetKernelArg(kernel, arg_cnt++, sizeof(var), &var);
    checkError(err, "clSetKernelArg");
}

static void run_kernel(cl_command_queue cmds, cl_kernel ker) {
    int err = 0;

    err = clEnqueueNDRangeKernel(cmds, ker, 2, 0, global_item_size, 0, 0, 0, 0);
    checkError(err, "Running kernel");

    err = clFinish(cmds);
    checkError(err, "Waiting for kernel to finish");

    arg_cnt = 0;
}

static void do_blur(OpenCLStuff res, cl_mem dst, cl_mem src, int blur_radius, int w, int h, int nchannels) {
    int err = 0;

    set_argument_helper(res.ko_blur, dst);
    set_argument_helper(res.ko_blur, src);
    set_argument_helper(res.ko_blur, blur_radius);
    set_argument_helper(res.ko_blur, w);
    set_argument_helper(res.ko_blur, h);
    set_argument_helper(res.ko_blur, nchannels);

    run_kernel(res.commands, res.ko_blur);
}

static void unsharp_mask(char unsigned  *out, char unsigned *in, int blur_radius,
                         int w, int h, int nchannels, OpenCLStuff res, int blur_times) {
    size_t size = w * h * nchannels;
    global_item_size[0] = w * nchannels;
    global_item_size[1] = h * nchannels;

    int err = 0;

    //
    // Calculate Blur
    //

    // Create a copy of the original image in GPU memory.
    cl_mem gpu_in = clCreateBuffer(res.context, CL_MEM_READ_WRITE, size, 0, &err);  checkError(err, "Error allocating memory.");
    err = clEnqueueWriteBuffer(res.commands, gpu_in, CL_TRUE, 0, size, in, 0, 0, 0); checkError(err, "Error copying original buffer from gpu to cpu");

    // Allocate memory for the swap buffers.
    cl_mem blur1 = clCreateBuffer(res.context, CL_MEM_READ_WRITE, size, 0, &err); checkError(err, "Creating blur buffer 1");
    cl_mem blur2 = clCreateBuffer(res.context, CL_MEM_READ_WRITE, size, 0, &err); checkError(err, "Creating blur buffer 2");

    // Copy original to blur 1.
    err = clEnqueueCopyBuffer(res.commands, gpu_in, blur1, 0, 0, size, 0, 0, 0); checkError(err, "Copying original image to blur1");

    cl_mem final_blur = 0, gpu_out = 0;

    // Do the blur n times.
    for(int i = 0, n = blur_times; (i < n); ++i) {
        bool is_even = !(i &1);

        cl_mem src = (is_even) ? blur1 : blur2;
        cl_mem dst = (is_even) ? blur2 : blur1;

        // Set arguments then run kernel.
        set_argument_helper(res.ko_blur, dst);
        set_argument_helper(res.ko_blur, src);
        set_argument_helper(res.ko_blur, blur_radius);
        set_argument_helper(res.ko_blur, w);
        set_argument_helper(res.ko_blur, h);
        set_argument_helper(res.ko_blur, nchannels);
        run_kernel(res.commands, res.ko_blur);

        // Set the final blur to 1 or 2, depending on whether we finish on even or odd.
        // But also set the final output memory buffer to the opposite one, to avoid the
        // extra memory allocation later.
        if(i == n - 1) {
            final_blur = (is_even) ? blur2 : blur1;
            gpu_out    = (is_even) ? blur1 : blur2;
        }
    }

    //
    // Add weighted
    //

    // Set arguments then run kernel.
    set_argument_helper(res.ko_weighted, gpu_out);
    set_argument_helper(res.ko_weighted, gpu_in);
    set_argument_helper(res.ko_weighted, 1.5f);
    set_argument_helper(res.ko_weighted, final_blur);
    set_argument_helper(res.ko_weighted, -0.5f);
    set_argument_helper(res.ko_weighted, 0.0f);
    set_argument_helper(res.ko_weighted, w);
    set_argument_helper(res.ko_weighted, h);
    set_argument_helper(res.ko_weighted, nchannels);
    run_kernel(res.commands, res.ko_weighted);

    // Copy memory from GPU to CPU.
    err = clEnqueueReadBuffer(res.commands, gpu_out, CL_TRUE, 0, size, out, 0, 0, 0); checkError(err, "Reading back.");

    // TODO(Jonny): Free opencl memory.
}

static char const *opencl_original_blur =
    "__kernel void pixel_average(__global unsigned char *out,\n"
    "                            __global const unsigned char *in,\n"
    "                            const int x, const int y, const int blur_radius,\n"
    "                            const int w, const int h, const int nchannels) {\n"
    "    float red_total = 0, green_total = 0, blue_total = 0;\n"
    "    const unsigned nsamples = (blur_radius*2-1) * (blur_radius*2-1);\n"
    "    for(int j = y-blur_radius+1; j < y+blur_radius; ++j) {\n"
    "        for(int i = x-blur_radius+1; i < x+blur_radius; ++i) {\n"
    "            const unsigned r_i = i < 0 ? 0 : i >= w ? w-1 : i;\n"
    "            const unsigned r_j = j < 0 ? 0 : j >= h ? h-1 : j;\n"
    "            unsigned byte_offset = (r_j*w+r_i)*nchannels;\n"
    "            red_total   += in[byte_offset + 0];\n"
    "            green_total += in[byte_offset + 1];\n"
    "            blue_total  += in[byte_offset + 2];\n"
    "        }\n"
    "    }\n"
    "\n"
    "    unsigned byte_offset = (y*w+x)*nchannels;\n"
    "    out[byte_offset + 0] = (char unsigned)(red_total   / nsamples);\n"
    "    out[byte_offset + 1] = (char unsigned)(green_total / nsamples);\n"
    "    out[byte_offset + 2] = (char unsigned)(blue_total  / nsamples);\n"
    "}\n"
    "\n"
    "__kernel void blur(__global unsigned char *out, __global const unsigned char *in,\n"
    "                   const int blur_radius,\n"
    "                   const unsigned w, const unsigned h, const unsigned nchannels) {\n"
    "    int x = get_global_id(0);\n"
    "    int y = get_global_id(1);\n"
    "    if((x <= w) && (y <= h)) {\n"
    "        pixel_average(out, in, x, y, blur_radius, w, h, nchannels);\n"
    "    }\n"
    "}\n";

#if 0
From: http://blog.ivank.net/fastest-gaussian-blur.html
// source channel, target channel, width, height, radius
function gaussBlur_1 (scl, tcl, w, h, r) {
    var rs = Math.ceil(r * 2.57);     // significant radius
    for(var i=0; i<h; i++)
        for(var j=0; j<w; j++) {
            var val = 0, wsum = 0;
            for(var iy = i-rs; iy<i+rs+1; iy++)
                for(var ix = j-rs; ix<j+rs+1; ix++) {
                    var x = Math.min(w-1, Math.max(0, ix));
                    var y = Math.min(h-1, Math.max(0, iy));
                    var dsq = (ix-j)*(ix-j)+(iy-i)*(iy-i);
                    var wght = Math.exp( -dsq / (2*r*r) ) / (Math.PI*2*r*r);
                    val += scl[y*w+x] * wght;  wsum += wght;
                }
            tcl[i*w+j] = Math.round(val/wsum);
        }
}
#endif
static char const *opencl_gaussian_blur =
    "__kernel void blur(__global char unsigned *out, __global char unsigned *in,\n"
    "                   int const blur_radius, int const w, int const h, int const nchannels) {\n"
    "    int const significant_radius = ceil(blur_radius * 2.57f);     // significant radius\n"
    "    int const i = get_global_id(0);\n"
    "    int const j = get_global_id(1);\n"
    "    float val = 0, wsum = 0;\n"
    "    if((i < w) && (j < h)) {"
    "        for(int channel_count = 0; (channel_count < nchannels); ++channel_count) {\n"
    "            for(int iter = 0; (iter < nchannels); ++iter) {\n"
    "                for(int iy = (j - significant_radius); (iy < j + significant_radius + 1); ++iy) {\n"
    "                    for(int ix = (i - significant_radius); (ix < i + significant_radius + 1); ++ix) {\n"
    "                        int x = min(w - 1, max(0, ix));\n"
    "                        int y = min(h - 1, max(0, iy));\n"
    "                        float dsq = (ix - i) * (ix - i) + (iy - j) * (iy - j);\n"
    "                        float wght = exp( -dsq / (2 * blur_radius*blur_radius) ) / (M_PI * 2 * blur_radius*blur_radius);\n"
    "                        val += in[y * w + x * nchannels] * wght;\n"
    "                        wsum += wght;\n"
    "                    }\n"
    "                }\n"
    "\n"
    "                out[j * w + j * nchannels] = round(val / wsum);\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "}\n";

// Calculates the weighted sum of two arrays, in1 and in2 according
// to the formula: out(I) = saturate(in1(I)*alpha + in2(I)*beta + gamma)
static char const *opencl_add_weighted =
    "__kernel void add_weighted(__global unsigned char *out,\n"
    "                           __global const unsigned char *in1, const float alpha,\n"
    "                           __global const unsigned char *in2, const float  beta, const float gamma,\n"
    "                           const unsigned w, const unsigned h, const unsigned nchannels) {\n"
    "    int x = get_global_id(0);\n"
    "    int y = get_global_id(1);\n"
    "    if((x < w) && (y < h)) {\n"
    "        unsigned byte_offset = (y * w + x) * nchannels;\n"
    "\n"
    "        float tmp = in1[byte_offset + 0] * alpha + in2[byte_offset + 0] * beta + gamma;\n"
    "        out[byte_offset + 0] = (char unsigned)(tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp);\n"
    "\n"
    "        tmp = in1[byte_offset + 1] * alpha + in2[byte_offset + 1] * beta + gamma;\n"
    "        out[byte_offset + 1] = (char unsigned)(tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp);\n"
    "\n"
    "        tmp = in1[byte_offset + 2] * alpha + in2[byte_offset + 2] * beta + gamma;\n"
    "        out[byte_offset + 2] = (char unsigned)(tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp);\n"
    "    }\n"
    "}\n";

static OpenCLStuff setup_opencl() {
    int err, i;
    cl_uint numPlatforms;
    cl_device_id device_id = 0;
    cl_program blur_program;
    cl_program add_weighted_program;
    // compute kernel

    OpenCLStuff res = {};

    // Find number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");
    if (numPlatforms == 0) {
        printf("Found 0 platforms!\n");
        return(res);
    }

    // Get all platforms
    cl_platform_id *Platform = new cl_platform_id[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    checkError(err, "Getting platforms");

    // Secure a GPU
    for (i = 0; i < numPlatforms; i++) {
        err = clGetDeviceIDs(Platform[i], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
        if (err == CL_SUCCESS) {
            break;
        }
    }

    if (device_id == NULL)
        checkError(err, "Finding a device");

    //err = output_device_info(device_id);
    //checkError(err, "Printing device output");

    // Create a compute context
    res.context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create a command queue
    res.commands = clCreateCommandQueue(res.context, device_id, 0, &err);
    checkError(err, "Creating command queue");

    // Create the compute program from the source buffer
    blur_program = clCreateProgramWithSource(res.context, 1, (use_gaussian_blur) ? (const char **)&opencl_gaussian_blur : (const char **) &opencl_original_blur, NULL, &err);
    checkError(err, "Creating blur_program");

    // Build the blur_program
    err = clBuildProgram(blur_program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build blur_program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(blur_program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return(res);
    }

    // Create the compute program from the source buffer
    add_weighted_program = clCreateProgramWithSource(res.context, 1, (const char **) &opencl_add_weighted, NULL, &err);
    checkError(err, "Creating add_weighted_program");

    // Build the add_weighted_program
    err = clBuildProgram(add_weighted_program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build add_weighted_program executable!\n%s\n", err_code(err));
        clGetProgramBuildInfo(add_weighted_program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return(res);
    }

    // Create the compute kernel from the program
    res.ko_blur     = clCreateKernel(blur_program,         "blur",         &err);
    checkError(err, "Creating kernel");

    res.ko_weighted = clCreateKernel(add_weighted_program, "add_weighted", &err);
    checkError(err, "Creating kernel");

    res.success = true;
    return(res);
}

int main(int argc, char *argv[]) {
    OpenCLStuff res = setup_opencl();
    if(res.success) {
        int i, blur_radius, blur_times;
        char const *ifilename = "lena.ppm";

        i = 0;
        blur_radius = 0;
        blur_times = 3;

        /*for(i = 0; (i < 5); )*/ {
            /*for(blur_times = 1; (blur_times < 20); blur_times += 1)*/ {
                /*for(blur_radius = 5; (blur_radius < 50); blur_radius += 5)*/ {
                    ppm img;
                    std::vector<char unsigned> data_in, data_sharp;

                    img.read(ifilename, data_in);
                    data_sharp.resize(img.w * img.h * img.nchannels);

                    auto t1 = std::chrono::steady_clock::now();

                    unsharp_mask(data_sharp.data(), data_in.data(), blur_radius,
                                 img.w, img.h, img.nchannels, res, blur_times);

                    auto t2 = std::chrono::steady_clock::now();
                    std::cout << "Image " << std::to_string(i)                                                          << std::endl;
                    std::cout << "    " << "Number of times blurred " << blur_times                                     << std::endl;
                    std::cout << "    " << "Radius "                  << blur_radius                                    << std::endl;
                    std::cout << "    " << "Time (seconds) "          << std::chrono::duration<double>(t2 - t1).count() << std::endl;
                    std::cout << std::endl << std::endl;

                    std::string ofilename = "out" + std::to_string(i) + ".ppm";
                    img.write(ofilename.c_str(), data_sharp);

                    ++i;
                }
            }
        }
    }

    return(0);
}
