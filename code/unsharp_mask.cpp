#include <chrono>
#include <climits>
#include "cl/cl.h"
#include "err_code.h"

#include "ppm.hpp"

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
                         int w, int h, int nchannels, OpenCLStuff res) {
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
    for(int i = 0, n = 3; (i < n); ++i) {
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
    err = clEnqueueReadBuffer(res.commands, gpu_out, CL_TRUE, 0, size, out, 0, 0, 0);   checkError(err, "Reading back.");

    // TODO(Jonny): Free opencl memory.
}

static char const *opencl_blur =
    R"(

__kernel void pixel_average(__global unsigned char *out,
                            __global const unsigned char *in,
                            const int x, const int y, const int blur_radius,
                            const int w, const int h, const int nchannels) {
    float red_total = 0, green_total = 0, blue_total = 0;
    const unsigned nsamples = (blur_radius*2-1) * (blur_radius*2-1);
    for(int j = y-blur_radius+1; j < y+blur_radius; ++j) {
        for(int i = x-blur_radius+1; i < x+blur_radius; ++i) {
            const unsigned r_i = i < 0 ? 0 : i >= w ? w-1 : i;
            const unsigned r_j = j < 0 ? 0 : j >= h ? h-1 : j;
            unsigned byte_offset = (r_j*w+r_i)*nchannels;
            red_total   += in[byte_offset + 0];
            green_total += in[byte_offset + 1];
            blue_total  += in[byte_offset + 2];
        }
    }

    unsigned byte_offset = (y*w+x)*nchannels;
    out[byte_offset + 0] = (char unsigned)(red_total   / nsamples);
    out[byte_offset + 1] = (char unsigned)(green_total / nsamples);
    out[byte_offset + 2] = (char unsigned)(blue_total  / nsamples);
}

__kernel void blur(__global unsigned char *out, __global const unsigned char *in,
                   const int blur_radius,
                   const unsigned w, const unsigned h, const unsigned nchannels) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if((x <= w) && (y <= h)) {
        pixel_average(out, in, x, y, blur_radius, w, h, nchannels);
    }
}

)";


// Calculates the weighted sum of two arrays, in1 and in2 according
// to the formula: out(I) = saturate(in1(I)*alpha + in2(I)*beta + gamma)
static char const *opencl_add_weighted =
    R"(
__kernel void add_weighted(__global unsigned char *out,
                           __global const unsigned char *in1, const float alpha,
                           __global const unsigned char *in2, const float  beta, const float gamma,
                           const unsigned w, const unsigned h, const unsigned nchannels) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if((x < w) && (y < h)) {
        unsigned byte_offset = (y * w + x) * nchannels;

        float tmp = in1[byte_offset + 0] * alpha + in2[byte_offset + 0] * beta + gamma;
        out[byte_offset + 0] = (char unsigned)(tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp);

        tmp = in1[byte_offset + 1] * alpha + in2[byte_offset + 1] * beta + gamma;
        out[byte_offset + 1] = (char unsigned)(tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp);

        tmp = in1[byte_offset + 2] * alpha + in2[byte_offset + 2] * beta + gamma;
        out[byte_offset + 2] = (char unsigned)(tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp);
    }
}

)";

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
    blur_program = clCreateProgramWithSource(res.context, 1, (const char **) &opencl_blur, NULL, &err);
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

        char const *ifilename = (argc > 1) ? argv[1] : "ghost-town-8k.ppm";
        char const *ofilename = (argc > 2) ? argv[2] : "out.ppm";
        int blur_radius       = (argc > 3) ? std::atoi(argv[3]) : 50;

        ppm img;
        std::vector<char unsigned> data_in, data_sharp;

        img.read(ifilename, data_in);
        data_sharp.resize(img.w * img.h * img.nchannels);

        auto t1 = std::chrono::steady_clock::now();

        unsharp_mask(data_sharp.data(), data_in.data(), blur_radius,
                     img.w, img.h, img.nchannels, res);

        auto t2 = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration<double>(t2-t1).count() << " seconds.\n";

        img.write(ofilename, data_sharp);
    }

    return(0);
}
