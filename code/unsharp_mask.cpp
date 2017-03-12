#include <chrono>
#include <climits>
#include "cl/cl.h"
#include "err_code.h"

// Apply an unsharp mask to the 24-bit PPM loaded from the file path of
// the first input argument; then write the sharpened output to the file path
// of the second argument. The third argument provides the blur radius.

#include "ppm.hpp"

#ifndef DEVICE
    #define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

struct OpenCLRes {
    bool success;
    cl_context context;
    cl_kernel ko_blur;
    cl_kernel ko_weighted;
    cl_command_queue commands;
};

void add_weighted(unsigned char *out,
                  const unsigned char *in1, const float alpha,
                  const unsigned char *in2, const float  beta, const float gamma,
                  const unsigned w, const unsigned h, const unsigned nchannels) {
    for(int x = 0; (x < w); ++x) {
        for(int y = 0; (y < h); ++y) {
            unsigned byte_offset = (y * w + x) * nchannels;

            float tmp = in1[byte_offset + 0] * alpha + in2[byte_offset + 0] * beta + gamma;
            out[byte_offset + 0] = (char unsigned)(tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp);

            tmp = in1[byte_offset + 1] * alpha + in2[byte_offset + 1] * beta + gamma;
            out[byte_offset + 1] = (char unsigned)(tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp);

            tmp = in1[byte_offset + 2] * alpha + in2[byte_offset + 2] * beta + gamma;
            out[byte_offset + 2] = (char unsigned)(tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp);
        }
    }
}
#if 0
// Averages the nsamples pixels within blur_radius of (x,y). Pixels which
// would be outside the image, replicate the value at the image border.
void pixel_average(unsigned char *out,
                   const unsigned char *in,
                   const int x, const int y, const int blur_radius,
                   const unsigned w, const unsigned h, const unsigned nchannels) {
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

void blur(unsigned char *out, const unsigned char *in,
          const int blur_radius,
          const unsigned w, const unsigned h, const unsigned nchannels) {
    for(int y = 0; y < h; ++y) {
        for(int x = 0; x < w; ++x) {
            pixel_average(out,in,x,y,blur_radius,w,h,nchannels);
        }
    }
}
#endif

void unsharp_mask(unsigned char *out, const unsigned char *in,
                  const int blur_radius,
                  const unsigned w, const unsigned h, const unsigned nchannels, OpenCLRes res) {
    int err = 0;

    //
    // Calculate Blur
    //
    char unsigned *host_blur3 = (char unsigned *)malloc(w * h * nchannels);

    cl_mem blur1   = clCreateBuffer(res.context,  CL_MEM_READ_ONLY,  w * h * nchannels, NULL, &err); checkError(err, "Creating blur buffers");
    cl_mem blur2   = clCreateBuffer(res.context,  CL_MEM_READ_ONLY,  w * h * nchannels, NULL, &err); checkError(err, "Creating blur buffers");
    cl_mem blur3   = clCreateBuffer(res.context,  CL_MEM_READ_ONLY,  w * h * nchannels, NULL, &err); checkError(err, "Creating blur buffers");

    cl_mem gpu_in  = clCreateBuffer(res.context,  CL_MEM_READ_ONLY,  w * h * nchannels, NULL, &err);
    checkError(err, "Creating blur buffers");

    err = clEnqueueWriteBuffer(res.commands, gpu_in, CL_TRUE, 0, w * h * nchannels, in, 0, NULL, NULL);
    checkError(err, "Copying");

    size_t global_item_size[2] = { w * nchannels, h * nchannels};

    {
        err = clSetKernelArg(res.ko_blur, 0, sizeof(cl_mem),   &blur1);       checkError(err, "Argument 0");
        err = clSetKernelArg(res.ko_blur, 1, sizeof(cl_mem),   &gpu_in);      checkError(err, "Argument 1");
        err = clSetKernelArg(res.ko_blur, 2, sizeof(int),      &blur_radius); checkError(err, "Argument 2");
        err = clSetKernelArg(res.ko_blur, 3, sizeof(unsigned), &w);           checkError(err, "Argument 3");
        err = clSetKernelArg(res.ko_blur, 4, sizeof(unsigned), &h);           checkError(err, "Argument 4");
        err = clSetKernelArg(res.ko_blur, 5, sizeof(unsigned), &nchannels);   checkError(err, "Argument 5");

        err = clEnqueueNDRangeKernel(res.commands, res.ko_blur, 2, NULL, global_item_size, NULL, 0, NULL, NULL);
        checkError(err, "Running kernel");

        err = clFinish(res.commands);
        checkError(err, "Waiting for kernel to finish");
    }

    {
        err = clSetKernelArg(res.ko_blur, 0, sizeof(cl_mem),   &blur2);       checkError(err, "Argument 0");
        err = clSetKernelArg(res.ko_blur, 1, sizeof(cl_mem),   &blur1);       checkError(err, "Argument 1");
        err = clSetKernelArg(res.ko_blur, 2, sizeof(int),      &blur_radius); checkError(err, "Argument 2");
        err = clSetKernelArg(res.ko_blur, 3, sizeof(unsigned), &w);           checkError(err, "Argument 3");
        err = clSetKernelArg(res.ko_blur, 4, sizeof(unsigned), &h);           checkError(err, "Argument 4");
        err = clSetKernelArg(res.ko_blur, 5, sizeof(unsigned), &nchannels);   checkError(err, "Argument 5");

        err = clEnqueueNDRangeKernel(res.commands, res.ko_blur, 2, NULL, global_item_size, NULL, 0, NULL, NULL);
        checkError(err, "Running kernel");

        err = clFinish(res.commands);
        checkError(err, "Waiting for kernel to finish");
    }

    {
        err = clSetKernelArg(res.ko_blur, 0, sizeof(cl_mem),   &blur3);       checkError(err, "Argument 0");
        err = clSetKernelArg(res.ko_blur, 1, sizeof(cl_mem),   &blur2);       checkError(err, "Argument 1");
        err = clSetKernelArg(res.ko_blur, 2, sizeof(int),      &blur_radius); checkError(err, "Argument 2");
        err = clSetKernelArg(res.ko_blur, 3, sizeof(unsigned), &w);           checkError(err, "Argument 3");
        err = clSetKernelArg(res.ko_blur, 4, sizeof(unsigned), &h);           checkError(err, "Argument 4");
        err = clSetKernelArg(res.ko_blur, 5, sizeof(unsigned), &nchannels);   checkError(err, "Argument 5");

        err = clEnqueueNDRangeKernel(res.commands, res.ko_blur, 2, NULL, global_item_size, NULL, 0, NULL, NULL);
        checkError(err, "Running kernel");

        err = clFinish(res.commands);
        checkError(err, "Waiting for kernel to finish");
    }

    //
    // Add weighted
    //
    {
        cl_mem gpu_out = clCreateBuffer(res.context,  CL_MEM_READ_ONLY,  w * h * nchannels, NULL, &err);
        checkError(err, "Allocating out");

        float a = 1.5f, b = -0.5f, c = 0.0f;
        err = clSetKernelArg(res.ko_weighted, 0, sizeof(cl_mem), &gpu_out);   checkError(err, "Argument 0");
        err = clSetKernelArg(res.ko_weighted, 1, sizeof(cl_mem), &gpu_in);    checkError(err, "Argument 1");
        err = clSetKernelArg(res.ko_weighted, 2, sizeof(float),  &a);         checkError(err, "Argument 2");
        err = clSetKernelArg(res.ko_weighted, 3, sizeof(cl_mem), &blur3);     checkError(err, "Argument 3");
        err = clSetKernelArg(res.ko_weighted, 4, sizeof(float),  &b);         checkError(err, "Argument 4");
        err = clSetKernelArg(res.ko_weighted, 5, sizeof(float),  &c);         checkError(err, "Argument 5");
        err = clSetKernelArg(res.ko_weighted, 6, sizeof(int),    &w);         checkError(err, "Argument 6");
        err = clSetKernelArg(res.ko_weighted, 7, sizeof(int),    &h);         checkError(err, "Argument 7");
        err = clSetKernelArg(res.ko_weighted, 8, sizeof(int),    &nchannels); checkError(err, "Argument 8");

        err = clEnqueueNDRangeKernel(res.commands, res.ko_weighted, 2, NULL, global_item_size, NULL, 0, NULL, NULL);
        checkError(err, "Running kernel");

        err = clFinish(res.commands);
        checkError(err, "Waiting for kernel to finish");

        err = clEnqueueReadBuffer(res.commands, gpu_out, CL_TRUE, 0, w * h * nchannels, out, 0, NULL, NULL);
        checkError(err, "Reading back.");
    }
}

static char const *opencl_blur =
    R"(

__kernel void pixel_average(__global unsigned char *out,
                            __global const unsigned char *in,
                            const int x, const int y, const int blur_radius,
                            const unsigned w, const unsigned h, const unsigned nchannels) {
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
                   const unsigned w, const unsigned h, const unsigned nchannels)
{
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

static OpenCLRes setup_opencl() {
    int err, i;
    cl_uint numPlatforms;
    cl_device_id device_id = 0;
    cl_program blur_program;
    cl_program add_weighted_program;
    // compute kernel

    OpenCLRes res = {};

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
        err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
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
    OpenCLRes res = setup_opencl();
    if(res.success) {

        const char *ifilename = argc > 1 ?           argv[1] : "lena.ppm";
        const char *ofilename = argc > 2 ?           argv[2] : "out.ppm";
        const int blur_radius = argc > 3 ? std::atoi(argv[3]) : 5;

        ppm img;
        std::vector<unsigned char> data_in, data_sharp;

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
