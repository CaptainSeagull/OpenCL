#ifndef _ADD_WEIGHTED_HPP_
#define _ADD_WEIGHTED_HPP_

#include <climits>

// Calculates the weighted sum of two arrays, in1 and in2 according
// to the formula: out(I) = saturate(in1(I)*alpha + in2(I)*beta + gamma)
template <typename T>
void add_weighted(unsigned char *out,
                  const unsigned char *in1, const T alpha,
                  const unsigned char *in2, const T  beta, const T gamma,
                  const unsigned w, const unsigned h, const unsigned nchannels)
{
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      unsigned byte_offset = (y*w+x)*nchannels;
      
      T tmp = in1[byte_offset+0] * alpha + in2[byte_offset+0] * beta + gamma;
      out[byte_offset+0] = tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp;

        tmp = in1[byte_offset+1] * alpha + in2[byte_offset+1] * beta + gamma;
      out[byte_offset+1] = tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp;

        tmp = in1[byte_offset+2] * alpha + in2[byte_offset+2] * beta + gamma;
      out[byte_offset+2] = tmp < 0 ? 0 : tmp > UCHAR_MAX ? UCHAR_MAX : tmp;
    }
  }
}

#endif // _ADD_WEIGHTED_HPP_
